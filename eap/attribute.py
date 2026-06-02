import os.path
import pickle
from typing import Callable, List, Union, Optional, Literal, Tuple
from functools import partial
import time
import cProfile
import pstats
import io

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, load_ablations, \
    compute_mean_activations_per_class
from .evaluate import evaluate_graph, evaluate_graph_per_sample, evaluate_baseline, evaluate_baseline_per_sample
from .graph import Graph

import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import partial
from einops import einsum

Tensor = torch.Tensor

def differentiable_eap_ig_to_logits_archive(
    model,
    graph,
    clean_images: Tensor,
    corrupted_images: Optional[Tensor],
    label,  # keep same structure your metric expects
    metric: Callable[[Tensor, Tensor, object], Tensor],  # metric(logits, clean_logits, label)->scalar
    steps: int = 5,
    intervention: str = "mean-positional",  # "patching", "mean", "mean-positional", "optimal"
    intervention_means: Optional[Tensor] = None,  # precomputed means in *activation_difference* space
    optimal_ablations: Optional[Tensor] = None,    # preloaded optimal ablations tensor (already broadcastable)
    layers_src: Optional[List[int]] = None,        # restrict which layers contribute src nodes
    create_graph: bool = True,                     # True => differentiable wrt model params (2nd-order)
    amp_bf16: bool = True,
    return_clean_logits: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """
    Differentiable, efficient EAP-IG variant that matches the *original hook set* (input, a{l}.h0, m{l})
    but only computes attribution scores for edges that end at the output/logits node.

    "Best fix" here means:
      - We keep graph.forward_index(node) as int OR slice (no expansion)
      - We keep activation_difference with full forward dimension, and handle slice/int at reduction time
      - We avoid backward hooks and in-place mutation of global 'scores'
      - We compute gradients using autograd.grad on a captured logits.in_hook activation

    Returns:
      scores_to_logits_per_srcnode: Tensor [n_src_nodes_selected]
          Each entry corresponds to one src node in the ordered list:
            ['input', 'a0.h0','m0','a1.h0','m1', ...] (restricted by layers_src)
          For slice-index nodes, we sum contributions over that slice.
      clean_logits: Tensor [B, n_classes] (if return_clean_logits=True)

    Notes:
      - This is still steps x forward passes. Keep steps small (4-8).
      - If create_graph=True, this becomes 2nd-order and can be expensive.
      - Activation_difference is built under no_grad by default (baseline treated as constant).
    """

    device = clean_images.device
    B = clean_images.shape[0]
    n_pos = (model.cfg.image_size // model.cfg.patch_size) ** 2 + 1

    # --------------------- select src nodes (original set) ---------------------
    if layers_src is None:
        layers_src = list(range(graph.cfg["n_layers"]))

    src_nodes = [graph.nodes["input"]]
    for l in layers_src:
        src_nodes.append(graph.nodes[f"a{l}.h0"])
        src_nodes.append(graph.nodes[f"m{l}"])

    # logits bookkeeping: only want contributions from src forward indices strictly before logits in topological order
    logits_node = graph.nodes["logits"]
    prev_index_logits = graph.prev_index(logits_node)  # int

    def _overlaps_before_logits(idx: Union[int, slice], prev: int) -> bool:
        if isinstance(idx, int):
            return idx < prev
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            return start < prev  # any overlap at all
        return False

    src_specs: List[Tuple[object, Union[int, slice]]] = []
    for node in src_nodes:
        idx = graph.forward_index(node)  # int or slice
        if _overlaps_before_logits(idx, prev_index_logits):
            src_specs.append((node, idx))

    assert len(src_specs) > 0, "No src nodes selected before logits."

    # --------------------- activation_difference: full forward dim ---------------------
    # Full forward dim is simplest + faithful to original semantics. If you later want memory optimization,
    # we can do a slice-aware compact storage, but that requires knowing activation shapes.
    activation_difference = torch.zeros(
        (B, n_pos, graph.n_forward, model.cfg.d_model),
        device=device,
        dtype=model.cfg.dtype,
    )

    # forward hooks to build corrupted-clean activation difference
    def activation_hook(fwd_index: Union[int, slice], activations: Tensor, hook, add: bool = True):
        # We do this under no_grad when building Δa, so detach is optional; keep it for safety.
        acts = activations.detach()
        if add:
            activation_difference[:, :, fwd_index] += acts
        else:
            activation_difference[:, :, fwd_index] -= acts

    fwd_hooks_corrupted = []
    fwd_hooks_clean = []

    for node, fwd_index in src_specs:
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index, add=True)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))

    # --------------------- (1) build Δa and get clean_logits ---------------------
    # Baseline (corrupted/mean/optimal) is treated as constant; build under no_grad to save memory.
    with torch.no_grad():
        if intervention == "patching":
            assert corrupted_images is not None, "corrupted_images required for patching intervention"
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_images)
            input_activations_corrupted = activation_difference[:, :, input_fwd_index].clone()
        elif intervention in ("mean", "mean-positional"):
            per_position = (intervention == "mean-positional")

            # compute mean activations from *this* batch (or intervention batch)
            # using the SAME forward hooks you already have
            mean_accum = torch.zeros_like(activation_difference)  # [B,pos,fwd,d]
            # but we only need means broadcastable, so compute mean over batch (and maybe pos)

            with model.hooks(fwd_hooks=fwd_hooks_corrupted):  # re-use hook to write activations
                _ = model(clean_images)  # or corrupted_images, depending on your definition

            # now activation_difference currently contains activations (added) from this pass;
            # convert to mean baseline
            if per_position:
                # mean over batch -> [1,pos,fwd,d], broadcast to [B,pos,fwd,d]
                means = activation_difference.mean(dim=0, keepdim=True)
            else:
                # mean over batch and pos -> [1,1,fwd,d], broadcast
                means = activation_difference.mean(dim=(0, 1), keepdim=True)

            activation_difference.zero_()
            activation_difference += means.detach()  # STOPGRAD baseline
            input_activations_corrupted = means.detach()
        elif intervention == "optimal":
            assert optimal_ablations is not None, "optimal_ablations required for optimal intervention"
            activation_difference += optimal_ablations
        else:
            raise ValueError(f"Unsupported intervention={intervention}")

    # Clean forward: subtract clean acts into activation_difference, and compute clean_logits with grad
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (clean_images.is_cuda and amp_bf16)
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with amp_ctx:
        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_images).detach() # reference

    # We need input activations for IG interpolation (in activation_difference space)
    input_node = graph.nodes["input"]
    input_fwd_index = graph.forward_index(input_node)  # should be int in most graphs, but keep generic
    if isinstance(input_fwd_index, slice):
        raise RuntimeError(
            "Unexpected: graph.forward_index('input') returned a slice. "
            "If this happens in your graph, paste its slice and input activation shape; we’ll adapt."
        )

    input_activations_corrupted = input_activations_corrupted[:, :, input_fwd_index]
    input_activations_clean = input_activations_corrupted - activation_difference[:, :, input_fwd_index]

    # --------------------- (2) IG loop: interpolate input activation, grad wrt logits.in_hook ---------------------
    def input_interpolation_hook(alpha: float):
        def hook_fn(activations: Tensor, hook):
            new_input = input_activations_corrupted + alpha * (input_activations_clean - input_activations_corrupted)
            return new_input
        return hook_fn

    def capture_logits_in_hook(store: Dict[str, Tensor]):
        def hook_fn(activations: Tensor, hook):
            # Make it a leaf so autograd.grad can target it reliably
            a = activations.detach().requires_grad_(True)
            store["logits_in"] = a
            return a
        return hook_fn

    alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)

    # accumulate in float32 for stability
    contrib_full_accum = torch.zeros((graph.n_forward,), device=device, dtype=torch.float32)

    for alpha in alphas:
        store: Dict[str, Tensor] = {}

        fwd_hooks = [
            (input_node.out_hook, input_interpolation_hook(float(alpha))),
            (logits_node.in_hook, capture_logits_in_hook(store)),
        ]

        with amp_ctx:
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(clean_images)
                mval = metric(logits, clean_logits, label)  # scalar

        (g_logits_in,) = torch.autograd.grad(
            outputs=mval,
            inputs=store["logits_in"],
            create_graph=create_graph,
            allow_unused=False,
        )

        # Ensure [B, pos, d]
        if g_logits_in.ndim == 2:
            g_logits_in = g_logits_in[:, None, :]
        elif g_logits_in.ndim != 3:
            raise RuntimeError(f"Unexpected logits_in grad shape: {tuple(g_logits_in.shape)}")

        # contrib_full: [graph.n_forward]
        # activation_difference: [B, pos, forward, d]
        # g_logits_in:          [B, pos, d]
        contrib_full = einsum(
            activation_difference, g_logits_in,
            "b p f d, b p d -> f"
        ).to(torch.float32)

        contrib_full_accum += contrib_full

    contrib_full_accum /= float(steps)

    # --------------------- (3) reduce to per-src-node scores, slice-aware ---------------------
    per_src_scores: List[Tensor] = []
    metric_spec = src_specs[1:]
    for node, idx in metric_spec:
        if isinstance(idx, int):
            per_src_scores.append(contrib_full_accum[idx])
        else:
            # sum over all forward indices in the slice
            # Also cap to < prev_index_logits to match original "scores[:prev_index]" behavior.
            start = 0 if idx.start is None else idx.start
            stop = idx.stop
            step = 1 if idx.step is None else idx.step
            stop = min(stop, prev_index_logits)
            per_src_scores.append(contrib_full_accum[start:stop:step].sum())

    scores_to_logits_per_srcnode = torch.stack(per_src_scores, dim=0)  # [n_src_nodes_selected]

    return scores_to_logits_per_srcnode

def differentiable_eap_ig_to_logits(
    model,
    graph,
    clean_images: Tensor,
    corrupted_images: Optional[Tensor],
    label,  # keep same structure your metric expects
    metric: Callable[[Tensor, Tensor, object], Tensor],  # metric(logits, clean_logits, label)->scalar
    steps: int = 4,
    intervention: str = "mean-positional",  # "patching", "mean", "mean-positional", "optimal"
    intervention_means: Optional[Tensor] = None,  # precomputed means in *activation_difference* space
    optimal_ablations: Optional[Tensor] = None,    # preloaded optimal ablations tensor (already broadcastable)
    layers_src: Optional[List[int]] = None,        # restrict which layers contribute src nodes
    create_graph: bool = True,                     # True => differentiable wrt model params (2nd-order)
    amp_bf16: bool = True,
    return_clean_logits: bool = True,
    grad_detach_mode: str = "detach_g",
    precomputed_g_ref: Optional[Tensor] = None,
) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """
    Differentiable, efficient EAP-IG variant that matches the *original hook set* (input, a{l}.h0, m{l})
    but only computes attribution scores for edges that end at the output/logits node.

    "Best fix" here means:
      - We keep graph.forward_index(node) as int OR slice (no expansion)
      - We keep activation_difference with full forward dimension, and handle slice/int at reduction time
      - We avoid backward hooks and in-place mutation of global 'scores'
      - We compute gradients using autograd.grad on a captured logits.in_hook activation

    Returns:
      scores_to_logits_per_srcnode: Tensor [n_src_nodes_selected]
          Each entry corresponds to one src node in the ordered list:
            ['input', 'a0.h0','m0','a1.h0','m1', ...] (restricted by layers_src)
          For slice-index nodes, we sum contributions over that slice.
      clean_logits: Tensor [B, n_classes] (if return_clean_logits=True)

    Notes:
      - This is still steps x forward passes. Keep steps small (4-8).
      - If create_graph=True, this becomes 2nd-order and can be expensive.
      - Activation_difference is built under no_grad by default (baseline treated as constant).
    """

    device = clean_images.device
    B = clean_images.shape[0]
    n_pos = (model.cfg.image_size // model.cfg.patch_size) ** 2 + 1

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (clean_images.is_cuda and amp_bf16)
        else torch.autocast(device_type="cpu", enabled=False)
    )

    # --------------------- select src nodes (original set) ---------------------
    if layers_src is None:
        layers_src = list(range(graph.cfg["n_layers"]))

    src_nodes = [graph.nodes["input"]]
    for l in layers_src:
        src_nodes.append(graph.nodes[f"a{l}.h0"])
        src_nodes.append(graph.nodes[f"m{l}"])

    # logits bookkeeping: only want contributions from src forward indices strictly before logits in topological order
    logits_node = graph.nodes["logits"]
    prev_index_logits = graph.prev_index(logits_node)  # int

    def _overlaps_before_logits(idx: Union[int, slice], prev: int) -> bool:
        if isinstance(idx, int):
            return idx < prev
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            return start < prev  # any overlap at all
        return False

    src_specs: List[Tuple[object, Union[int, slice]]] = []
    for node in src_nodes:
        idx = graph.forward_index(node)  # int or slice
        if _overlaps_before_logits(idx, prev_index_logits):
            src_specs.append((node, idx))

    assert len(src_specs) > 0, "No src nodes selected before logits."

    input_node = graph.nodes["input"]
    input_fwd_index = graph.forward_index(input_node)
    if isinstance(input_fwd_index, slice):
        raise RuntimeError("graph.forward_index('input') returned a slice; expected int.")

    # -------------------------------------------------------------------------
    # 2. Build baseline activations under no_grad (constant reference point)
    # -------------------------------------------------------------------------
    # baseline_acts[node_name] = Tensor [B, n_pos, d_model], detached constant
    baseline_acts: Dict[str, Tensor] = {}

    def make_save_baseline_hook(name: str):
        def hook_fn(activations: Tensor, hook):
            baseline_acts[name] = activations.detach().clone()
            return activations  # don't modify forward pass

        return hook_fn

    baseline_hooks = [
        (node.out_hook, make_save_baseline_hook(node.name))
        for node, _ in src_specs
    ]

    with torch.no_grad():
        with amp_ctx:
            if intervention == "patching":
                assert corrupted_images is not None, "corrupted_images required for patching"
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(corrupted_images)

            elif intervention in ("mean", "mean-positional"):
                # Use clean batch mean as baseline
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(clean_images)
                per_position = (intervention == "mean-positional")
                if per_position:
                    # mean over batch dim -> [1, n_pos, d], expand to [B, n_pos, d]
                    for name, acts in baseline_acts.items():
                        if 'a' in name:
                            baseline_acts[name] = acts.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).clone()
                        else:
                            baseline_acts[name] = acts.mean(dim=0, keepdim=True).expand(B, -1, -1).clone()
                else:
                    # mean over batch and pos -> [1, 1, d], expand to [B, n_pos, d]
                    baseline_acts = {
                        name: acts.mean(dim=(0, 1), keepdim=True).expand(B, n_pos, -1).clone()
                        for name, acts in baseline_acts.items()
                    }

            elif intervention == "optimal":
                assert optimal_ablations is not None, "optimal_ablations required"
                # optimal_ablations: dict mapping node name -> [B, n_pos, d] or broadcastable
                # If it's a single tensor indexed by forward index, caller must pass as dict
                # Here we expect a dict: {node_name: tensor}
                for node, _ in src_specs:
                    if node.name in optimal_ablations:
                        baseline_acts[node.name] = optimal_ablations[node.name].detach().clone()
                    else:
                        raise ValueError(f"optimal_ablations missing key: {node.name}")

            else:
                raise ValueError(f"Unsupported intervention: {intervention}")

    # -------------------------------------------------------------------------
    # 3. Clean forward WITH gradients
    #    clean_acts[node_name] = Tensor [B, n_pos, d_model], IN computation graph
    # -------------------------------------------------------------------------
    clean_acts: Dict[str, Tensor] = {}

    def make_save_clean_hook(name: str):
        def hook_fn(activations: Tensor, hook):
            # Store without detaching — keeps activations in the computation graph
            clean_acts[name] = activations
            return activations

        return hook_fn

    clean_hooks = [
        (node.out_hook, make_save_clean_hook(node.name))
        for node, _ in src_specs
    ]

    with amp_ctx:
        with model.hooks(fwd_hooks=clean_hooks):
            clean_logits = model(clean_images).detach()

    # -------------------------------------------------------------------------
    # 4. Compute activation differences — kept in graph so backprop flows through
    #    act_diffs → clean_acts → model weights.
    #    g_accum will be detached (used as a constant reference gradient direction).
    # -------------------------------------------------------------------------
    act_diffs: Dict[str, Tensor] = {}
    for node, _ in src_specs:
        name = node.name
        baseline = baseline_acts[name]          # [B, n_pos, d], no grad
        clean = clean_acts[name]                # [B, n_pos, d], has grad
        diff = baseline - clean if 'a' in name else (baseline - clean).unsqueeze(-2)
        act_diffs[name] = diff                  # keep in graph; grad flows via act_diffs only

    # For IG: input interpolation between baseline and clean input activations
    input_name = input_node.name
    input_baseline = baseline_acts[input_name]  # [B, n_pos, d], constant
    input_clean = clean_acts[input_name]        # [B, n_pos, d], has grad

    # -------------------------------------------------------------------------
    # 5. IG loop: interpolate input, capture logits_in, compute grad
    #    g_logits_in has grad w.r.t. model weights via create_graph=True
    # -------------------------------------------------------------------------
    def make_input_interpolation_hook(alpha: float):
        def hook_fn(activations: Tensor, hook):
            # Interpolate: alpha=0 => baseline, alpha=1 => clean
            # We use the stored tensors directly (not activations arg) so that
            # the hook doesn't depend on the current forward pass input activations
            # (which would create a circular dependency).
            interp = input_baseline + alpha * (input_clean - input_baseline)
            return interp

        return hook_fn

    def make_capture_logits_in_hook(store: Dict[str, Tensor]):
        def hook_fn(activations: Tensor, hook):
            # Do NOT detach. Return as-is so the forward graph is preserved.
            # We store a reference; autograd.grad will target this tensor.
            store["logits_in"] = activations
            return activations

        return hook_fn

    alphas = torch.arange(1, steps, device=device) / steps

    # Accumulate per-node contributions as a list of tensors, then sum (out-of-place)
    # per_node_accum[i] corresponds to src_specs[i], shape [B, n_pos, d]
    # We accumulate g_logits_in * (1/steps) across IG steps, then einsum with act_diffs
    # This avoids building a huge n_forward-dim buffer while staying differentiable.
    g_accum: Optional[Tensor] = None  # [B, n_pos, d], accumulated gradient

    if precomputed_g_ref is None:
        for alpha_val in alphas:
            store: Dict[str, Tensor] = {}

            ig_hooks = [
                (input_node.out_hook, make_input_interpolation_hook(float(alpha_val))),
                (logits_node.in_hook, make_capture_logits_in_hook(store)),
            ]

            with amp_ctx:
                with model.hooks(fwd_hooks=ig_hooks):
                    logits = model(clean_images)
                    mval = metric(logits, clean_logits, label).float()  # scalar

            assert "logits_in" in store, "logits_in hook was not triggered"

            # Gradient of metric w.r.t. logits_in
            # create_graph=True => g_logits_in itself has a grad_fn back to model weights
            _create_graph = (grad_detach_mode == "detach_actdiff")
            (g_logits_in,) = torch.autograd.grad(
                outputs=mval,
                inputs=store["logits_in"],
                create_graph=_create_graph,
                allow_unused=False,
            )

            # g_logits_in: [B, pos, d] or [B, d] — normalize shape
            if g_logits_in.ndim == 2:
                g_logits_in = g_logits_in.unsqueeze(1)  # [B, 1, d]
            elif g_logits_in.ndim != 3:
                raise RuntimeError(f"Unexpected g_logits_in shape: {tuple(g_logits_in.shape)}")

            # Accumulate out-of-place
            if g_accum is None:
                g_accum = g_logits_in
            else:
                g_accum = g_accum + g_logits_in  # out-of-place add

        # Divide by steps out-of-place
        g_accum = g_accum / float(steps)  # [B, n_pos, d], differentiable

    # -------------------------------------------------------------------------
    # 6. Contract act_diffs with g_accum per src node (out-of-place, differentiable)
    #    score_i = sum_{b,p,d} act_diff_i[b,p,d] * g_accum[b,p,d]
    #    g_accum is detached — used as a fixed reference gradient direction.
    #    Backprop flows only through act_diffs → clean_acts → model weights.
    # -------------------------------------------------------------------------
    per_src_scores: List[Tensor] = []

    scored_specs = src_specs[1:]

    if precomputed_g_ref is not None:
        g_ref = precomputed_g_ref  # already detached constant
    elif grad_detach_mode == "detach_g":
        g_ref = torch.nan_to_num(g_accum.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    else:  # detach_actdiff: gradient flows through g_accum (2nd-order)
        g_ref = None  # not used, score computed differently

    for node, idx in scored_specs:
        name = node.name
        diff = act_diffs[name]  # [B, n_pos, d], in graph
        if g_ref is not None:
            score = torch.einsum("b p f d, b p d -> f", diff.float(), g_ref.float())
        else:  # detach_actdiff
            score = torch.einsum("b p f d, b p d -> f", diff.detach().float(), g_accum.float())
        per_src_scores.append(score)

    # Stack into [n_scored_nodes] tensor — differentiable
    scores_to_logits_per_srcnode = torch.concat(per_src_scores, dim=0)

    return scores_to_logits_per_srcnode


@torch.enable_grad()
def compute_eap_g_ref(
    model,
    graph,
    clean_images: Tensor,
    corrupted_images: Optional[Tensor],
    label,
    metric: Callable[[Tensor, Tensor, object], Tensor],
    steps: int = 4,
    intervention: str = "mean-positional",
    layers_src: Optional[List[int]] = None,
    amp_bf16: bool = False,
) -> Tensor:
    """
    Runs the IG loop and returns only g_ref (detached, NaN-guarded) for caching.
    Used by the 'accumulated' circuit variant to build a stable multi-batch g_ref buffer.
    Gradient does NOT flow to model weights (g_ref is detached before return).
    """
    device = clean_images.device
    B = clean_images.shape[0]
    n_pos = (model.cfg.image_size // model.cfg.patch_size) ** 2 + 1

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (clean_images.is_cuda and amp_bf16)
        else torch.autocast(device_type="cpu", enabled=False)
    )

    # ── 1. select src nodes ──────────────────────────────────────────────────
    if layers_src is None:
        layers_src = list(range(graph.cfg["n_layers"]))

    src_nodes = [graph.nodes["input"]]
    for l in layers_src:
        src_nodes.append(graph.nodes[f"a{l}.h0"])
        src_nodes.append(graph.nodes[f"m{l}"])

    logits_node = graph.nodes["logits"]
    prev_index_logits = graph.prev_index(logits_node)

    def _overlaps_before_logits(idx: Union[int, slice], prev: int) -> bool:
        if isinstance(idx, int):
            return idx < prev
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            return start < prev
        return False

    src_specs: List[Tuple[object, Union[int, slice]]] = []
    for node in src_nodes:
        idx = graph.forward_index(node)
        if _overlaps_before_logits(idx, prev_index_logits):
            src_specs.append((node, idx))

    assert len(src_specs) > 0, "No src nodes selected before logits."

    input_node = graph.nodes["input"]

    # ── 2. Build baseline activations under no_grad ──────────────────────────
    baseline_acts: Dict[str, Tensor] = {}

    def make_save_baseline_hook(name: str):
        def hook_fn(activations: Tensor, hook):
            baseline_acts[name] = activations.detach().clone()
            return activations
        return hook_fn

    baseline_hooks = [
        (node.out_hook, make_save_baseline_hook(node.name))
        for node, _ in src_specs
    ]

    with torch.no_grad():
        with amp_ctx:
            if intervention == "patching":
                assert corrupted_images is not None, "corrupted_images required for patching"
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(corrupted_images)
            elif intervention in ("mean", "mean-positional"):
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(clean_images)
                per_position = (intervention == "mean-positional")
                if per_position:
                    for name, acts in baseline_acts.items():
                        if 'a' in name:
                            baseline_acts[name] = acts.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).clone()
                        else:
                            baseline_acts[name] = acts.mean(dim=0, keepdim=True).expand(B, -1, -1).clone()
                else:
                    baseline_acts = {
                        name: acts.mean(dim=(0, 1), keepdim=True).expand(B, n_pos, -1).clone()
                        for name, acts in baseline_acts.items()
                    }
            else:
                raise ValueError(f"Unsupported intervention: {intervention}")

    # ── 3. Clean forward to capture input_clean and logits (for metric) ───────
    clean_acts: Dict[str, Tensor] = {}

    def make_save_clean_hook(name: str):
        def hook_fn(activations: Tensor, hook):
            clean_acts[name] = activations
            return activations
        return hook_fn

    clean_hooks = [
        (node.out_hook, make_save_clean_hook(node.name))
        for node, _ in src_specs
    ]

    with amp_ctx:
        with model.hooks(fwd_hooks=clean_hooks):
            clean_logits = model(clean_images).detach()

    input_name = input_node.name
    input_baseline = baseline_acts[input_name]
    input_clean = clean_acts[input_name]

    # ── 4. IG loop ─────────────────────────────────────────────────────────────
    def make_input_interpolation_hook(alpha: float):
        def hook_fn(activations: Tensor, hook):
            interp = input_baseline + alpha * (input_clean - input_baseline)
            return interp
        return hook_fn

    def make_capture_logits_in_hook(store: Dict[str, Tensor]):
        def hook_fn(activations: Tensor, hook):
            store["logits_in"] = activations
            return activations
        return hook_fn

    alphas = torch.arange(1, steps, device=device) / steps
    g_accum: Optional[Tensor] = None

    for alpha_val in alphas:
        store: Dict[str, Tensor] = {}

        ig_hooks = [
            (input_node.out_hook, make_input_interpolation_hook(float(alpha_val))),
            (logits_node.in_hook, make_capture_logits_in_hook(store)),
        ]

        with amp_ctx:
            with model.hooks(fwd_hooks=ig_hooks):
                logits = model(clean_images)
                mval = metric(logits, clean_logits, label).float()

        assert "logits_in" in store, "logits_in hook was not triggered"

        (g_logits_in,) = torch.autograd.grad(
            outputs=mval,
            inputs=store["logits_in"],
            create_graph=False,
            allow_unused=False,
        )

        if g_logits_in.ndim == 2:
            g_logits_in = g_logits_in.unsqueeze(1)
        elif g_logits_in.ndim != 3:
            raise RuntimeError(f"Unexpected g_logits_in shape: {tuple(g_logits_in.shape)}")

        if g_accum is None:
            g_accum = g_logits_in
        else:
            g_accum = g_accum + g_logits_in

    g_accum = g_accum / float(steps)

    return torch.nan_to_num(g_accum.detach(), nan=0.0, posinf=0.0, neginf=0.0)


def differentiable_eap_target_logit(
    model,
    graph,
    clean_images: Tensor,
    corrupted_images: Optional[Tensor],
    label,
    intervention: str = "mean-positional",
    layers_src: Optional[List[int]] = None,
    amp_bf16: bool = True,
) -> Tensor:
    """
    EAP (not EAP-IG) variant: single clean forward pass, mean target-class logit as the
    circuit-discovery metric (matching the REdit approach).

    Difference from differentiable_eap_ig_to_logits:
      - No interpolation loop: gradient is taken at the clean activation point only.
      - Metric: mean(logits[:, y]) instead of KL divergence.
      - Cheaper: 1 forward pass + 1 autograd.grad (vs. `steps` forward passes).

    Circuit granularity and output format are identical to differentiable_eap_ig_to_logits:
      scores_to_logits_per_srcnode [n_src_nodes], used the same way for shallow/deep split.

    Backprop path (pure 1st-order):
      circuit_loss -> scores -> act_diffs -> clean_acts -> model weights
      g_ref is detached, so no 2nd-order gradient is created.
    """
    device = clean_images.device
    B = clean_images.shape[0]
    n_pos = (model.cfg.image_size // model.cfg.patch_size) ** 2 + 1

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (clean_images.is_cuda and amp_bf16)
        else torch.autocast(device_type="cpu", enabled=False)
    )

    # ── 1. select src nodes (same convention as differentiable_eap_ig_to_logits) ─
    if layers_src is None:
        layers_src = list(range(graph.cfg["n_layers"]))

    src_nodes = []
    for l in layers_src:
        src_nodes.append(graph.nodes[f"a{l}.h0"])
        src_nodes.append(graph.nodes[f"m{l}"])

    logits_node = graph.nodes["logits"]
    prev_index_logits = graph.prev_index(logits_node)

    def _overlaps_before_logits(idx, prev):
        if isinstance(idx, int):
            return idx < prev
        if isinstance(idx, slice):
            return (0 if idx.start is None else idx.start) < prev
        return False

    src_specs: List[Tuple[object, object]] = [
        (node, graph.forward_index(node))
        for node in src_nodes
        if _overlaps_before_logits(graph.forward_index(node), prev_index_logits)
    ]
    assert len(src_specs) > 0, "No src nodes selected before logits."

    # ── 2. baseline activations (no grad) ─────────────────────────────────────
    baseline_acts: Dict[str, Tensor] = {}

    def _save_baseline(name: str):
        def fn(act, hook):
            baseline_acts[name] = act.detach().clone()
            return act
        return fn

    baseline_hooks = [(node.out_hook, _save_baseline(node.name)) for node, _ in src_specs]

    with torch.no_grad():
        with amp_ctx:
            if intervention == "patching":
                assert corrupted_images is not None, "corrupted_images required for patching"
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(corrupted_images)
            elif intervention in ("mean", "mean-positional"):
                with model.hooks(fwd_hooks=baseline_hooks):
                    _ = model(clean_images)
                per_pos = (intervention == "mean-positional")
                for name, acts in baseline_acts.items():
                    if per_pos:
                        baseline_acts[name] = acts.mean(0, keepdim=True).expand(B, *acts.shape[1:]).clone()
                    else:
                        mean_dims = tuple(range(acts.ndim - 1))
                        baseline_acts[name] = acts.mean(mean_dims, keepdim=True).expand_as(acts).clone()
            else:
                raise ValueError(f"Unsupported intervention: {intervention}")

    # ── 3. clean forward (in graph) — capture clean_acts and logits_in together ─
    clean_acts: Dict[str, Tensor] = {}
    store: Dict[str, Tensor] = {}

    def _save_clean(name: str):
        def fn(act, hook):
            clean_acts[name] = act   # kept in graph
            return act
        return fn

    def _capture_logits_in(act, hook):
        store["logits_in"] = act     # kept in graph
        return act

    clean_hooks = [(node.out_hook, _save_clean(node.name)) for node, _ in src_specs]

    with amp_ctx:
        with model.hooks(fwd_hooks=clean_hooks + [(logits_node.in_hook, _capture_logits_in)]):
            logits = model(clean_images)   # [B, n_classes], in graph

    assert "logits_in" in store, "logits_in hook was not triggered"

    # ── 4. single gradient: ∂ mean(logits[:, y]) / ∂ logits_in ──────────────
    label_t = label if isinstance(label, Tensor) else torch.tensor(label, device=device)
    if label_t.ndim > 1:
        label_t = label_t[:, 0]   # [B,2] labels from PACS loader
    label_t = label_t.long()

    target_logit = logits[torch.arange(B, device=device), label_t].mean()

    (g_logits_in,) = torch.autograd.grad(
        outputs=target_logit,
        inputs=store["logits_in"],
        create_graph=False,
        allow_unused=False,
    )
    if g_logits_in.ndim == 2:
        g_logits_in = g_logits_in.unsqueeze(1)

    # detach + guard NaN/Inf so they cannot poison the backward
    # g_ref = torch.nan_to_num(g_logits_in.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    # ── 5. act_diffs and per-node scores (same structure as IG variant) ────────
    act_diffs: Dict[str, Tensor] = {}
    for node, _ in src_specs:
        name = node.name
        base  = baseline_acts[name]
        clean = clean_acts[name]
        diff  = base - clean if 'a' in name else (base - clean).unsqueeze(-2)
        act_diffs[name] = diff   # in graph through clean_acts

    per_src_scores: List[Tensor] = []
    for node, _ in src_specs[1:]:   # exclude input node (no upstream to score)
        diff  = act_diffs[node.name]
        score = torch.einsum("b p f d, b p d -> f", diff.float(), g_logits_in.float())
        per_src_scores.append(score)

    return torch.concat(per_src_scores, dim=0)


def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    baseline = evaluate_baseline(model, dataloader, metric, device=device).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention,
                                                intervention_dataloader=intervention_dataloader,
                                                optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                skip_clean=True, device=device).mean().item()
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores

def get_scores_exact_optimized(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """
    num_samples = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean = torch.stack(clean).to(device)
        corrupted = torch.stack(corrupted).to(device)
        graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
        baseline = evaluate_baseline_per_sample(model, clean, corrupted, label, metric, device=device).mean().item()
        edges = graph.edges.values()
        for edge in tqdm(edges):
            edge.in_graph = False
            intervened_performance = evaluate_graph_per_sample(model, graph, clean, corrupted, label, metric, intervention=intervention,
                                                    intervention_dataloader=intervention_dataloader,
                                                    optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                    skip_clean=True, device=device).mean().item()
            edge.score += intervened_performance - baseline
            edge.in_graph = True
        num_samples += 1
    edges = graph.edges.values()
    for edge in edges:
        edge.score /= num_samples

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores

def get_scores_exact_optimized_parallel(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean = torch.stack(clean).to(device)
        corrupted = torch.stack(corrupted).to(device)
        graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
        baseline = evaluate_baseline_per_sample(model, clean, corrupted, label, metric, device=device).mean().item()
        edges = graph.edges.values()
        for edge in tqdm(edges):
            edge.in_graph = False
            intervened_performance = evaluate_graph_per_sample(model, graph, clean, corrupted, label, metric, intervention=intervention,
                                                    intervention_dataloader=intervention_dataloader,
                                                    optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                    skip_clean=True, device=device).mean().item()
            edge.score += intervened_performance - baseline
            edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                   intervention_dataloader: Optional[DataLoader] = None, optimal_ablation_path: Optional[str] = None,
                   quiet=False, device='cuda', task=None, model_name=None, model_id=None, profile_one_batch=False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position, device=device)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    batch_idx = 0
    profiler = None
    if profile_one_batch:
        profiler = cProfile.Profile()
    
    for clean, corrupted, label in dataloader:
        is_first_batch = (batch_idx == 0)
        batch_idx += 1
        
        if is_first_batch and profile_one_batch:
            print("\n" + "="*80)
            print("PROFILING FIRST BATCH (EAP) - Starting batch processing...")
            print("="*80)
            profiler.enable()
            batch_start_time = time.perf_counter()
        
        batch_size = len(clean)
        total_items += batch_size
        
        if is_first_batch and profile_one_batch:
            t0 = time.perf_counter()
        
        clean_images = torch.stack(clean).to(device)
        corrupted_images = torch.stack(corrupted).to(device)

        if is_first_batch and profile_one_batch:
            t1 = time.perf_counter()
            print(f"  Data preparation: {(t1-t0)*1000:.2f} ms")

        if is_first_batch and profile_one_batch:
            t2 = time.perf_counter()

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                                           scores)

        if is_first_batch and profile_one_batch:
            t3 = time.perf_counter()
            print(f"  make_hooks_and_matrices: {(t3-t2)*1000:.2f} ms")

        if is_first_batch and profile_one_batch:
            t4 = time.perf_counter()

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_images)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            elif intervention == 'optimal':
                activation_difference += optimal_ablations

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_images)

        if is_first_batch and profile_one_batch:
            t5 = time.perf_counter()
            print(f"  Forward passes (inference mode): {(t5-t4)*1000:.2f} ms")

        if is_first_batch and profile_one_batch:
            backward_start = time.perf_counter()

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_images)
            metric_value = metric(logits, clean_logits, label)
            metric_value.backward()

        if is_first_batch and profile_one_batch:
            backward_end = time.perf_counter()
            print(f"  Backward pass: {(backward_end-backward_start)*1000:.2f} ms")
            batch_end_time = time.perf_counter()
            print(f"\nTotal batch time: {(batch_end_time-batch_start_time)*1000:.2f} ms")
            profiler.disable()
            
            # Print profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            print("\n" + "="*80)
            print("DETAILED PROFILING (Top 30 functions by cumulative time)")
            print("="*80)
            print(s.getvalue())
            
            # Also print by total time
            s2 = io.StringIO()
            ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
            ps2.print_stats(30)
            print("\n" + "="*80)
            print("DETAILED PROFILING (Top 30 functions by total time)")
            print("="*80)
            print(s2.getvalue())
            
            # Save profiling stats to file if task and model_id are provided
            if task is not None and model_id is not None:
                profile_dir = 'profiling_results'
                os.makedirs(profile_dir, exist_ok=True)
                profile_filename = f"batch_profile_EAP_{task.replace('_', '-')}_model{model_id}.txt"
                profile_output_path = os.path.join(profile_dir, profile_filename)
                
                with open(profile_output_path, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("BATCH PROFILING STATISTICS (EAP)\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Task: {task}\n")
                    f.write(f"Model ID: {model_id}\n")
                    f.write(f"Batch size: {batch_size}\n")
                    f.write(f"Total batch time: {(batch_end_time-batch_start_time)*1000:.2f} ms\n\n")
                    f.write("TIMING BREAKDOWN:\n")
                    f.write(f"  Data preparation: {(t1-t0)*1000:.2f} ms\n")
                    f.write(f"  make_hooks_and_matrices: {(t3-t2)*1000:.2f} ms\n")
                    f.write(f"  Forward passes (inference mode): {(t5-t4)*1000:.2f} ms\n")
                    f.write(f"  Backward pass: {(backward_end-backward_start)*1000:.2f} ms\n")
                    f.write(f"  Other: {max(0, (batch_end_time-batch_start_time)*1000 - ((t1-t0)*1000 + (t3-t2)*1000 + (t5-t4)*1000 + (backward_end-backward_start)*1000)):.2f} ms\n\n")
                    f.write("="*80 + "\n")
                    f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
                    f.write("="*80 + "\n")
                    ps_cum = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                    ps_cum.print_stats(50)
                    f.write("\n" + "="*80 + "\n")
                    f.write("TOP 50 FUNCTIONS BY TOTAL TIME\n")
                    f.write("="*80 + "\n")
                    ps_tot = pstats.Stats(profiler, stream=f).sort_stats('tottime')
                    ps_tot.print_stats(50)
                
                print(f"\nProfiling stats saved to: {profile_output_path}")
            
            print("="*80 + "\n")

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                      metric: Callable[[Tensor], Tensor], intervention: Literal[
            'patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                      steps=30, intervention_dataloader: Optional[DataLoader] = None, return_logits=False,
                      optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, model_id=None, get_perexample_scores=False, profile_one_batch=False, return_actdiff_norms=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if intervention == 'mean' or intervention == 'mean-positional':
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        # if model_id is not None:
        #     model_name = f'sweep_{model_id}'
        # if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
        #     with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
        #         means = pickle.load(file)
        # else:
        #     per_position = 'positional' in intervention
        #     means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position, device=device)
        #     means = means.unsqueeze(0)
        #     if not per_position:
        #         means = means.unsqueeze(0)
        #     with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
        #         pickle.dump(means, file)
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position,
                                         device=device)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    elif 'labeled-mean' in intervention:
        if model_id is not None:
            model_name = f'sweep_{model_id}'
        if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
                means, class_counts = pickle.load(file)
        else:
            assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
            per_position = 'positional' in intervention
            means, class_counts = compute_mean_activations_per_class(model, graph, intervention_dataloader, num_classes=2, per_position=per_position, device=device)
            for cls in means:
                means[cls] = means[cls].unsqueeze(0)
            if not per_position:
                for cls in means:
                    means[cls] = means[cls].unsqueeze(0)
            means = dict(means)
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
                pickle.dump((means, class_counts), file)
        all_counts = sum([class_counts[class_id] for class_id in class_counts])

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    per_example_scores = torch.zeros((len(dataloader.dataset), graph.n_forward, graph.n_backward), device='cpu', dtype=model.cfg.dtype) if get_perexample_scores else None
    labels = []
    predictions = []

    # Pre-compute per-layer forward indices for actdiff norm extraction
    n_layers = graph.cfg['n_layers']
    if return_actdiff_norms:
        layer_attn_idx = {l: graph.forward_index(graph.nodes[f'a{l}.h0']) for l in range(n_layers)}
        layer_mlp_idx  = {l: graph.forward_index(graph.nodes[f'm{l}'])    for l in range(n_layers)}
        actdiff_norms  = {l: 0.0 for l in range(n_layers)}

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    batch_idx = 0
    profiler = None
    if profile_one_batch:
        profiler = cProfile.Profile()
    
    for clean, corrupted, label in dataloader:
        is_first_batch = (batch_idx == 0)
        batch_idx += 1
        
        if is_first_batch and profile_one_batch:
            print("\n" + "="*80)
            print("PROFILING FIRST BATCH - Starting batch processing...")
            print("="*80)
            profiler.enable()
            batch_start_time = time.perf_counter()
        
        batch_size = len(clean)
        total_items += batch_size
        labels.extend([this_label[0].item() if isinstance(this_label[0], torch.Tensor) else this_label[0] for this_label in label])

        if is_first_batch and profile_one_batch:
            t0 = time.perf_counter()
        
        clean_images = torch.stack(clean).to(device)
        if corrupted[0] is not None:
            corrupted_images = torch.stack(corrupted).to(device)

        if is_first_batch and profile_one_batch:
            t1 = time.perf_counter()
            print(f"  Data preparation: {(t1-t0)*1000:.2f} ms")

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        if is_first_batch and profile_one_batch:
            t2 = time.perf_counter()
        
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)

        if is_first_batch and profile_one_batch:
            t3 = time.perf_counter()
            print(f"  make_hooks_and_matrices: {(t3-t2)*1000:.2f} ms")

        if is_first_batch and profile_one_batch:
            t4 = time.perf_counter()
        
        with torch.inference_mode():
            if intervention == 'patching':
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_images)

            elif 'labeled-mean' in intervention:
                with torch.no_grad():
                    pred_logits = model(clean_images)
                    pred_classes = torch.argmax(pred_logits, dim=-1)
                batch_means = []
                all_classes = list(means.keys())
                for cls in pred_classes.cpu().tolist():
                    other_means = [means[other_cls] for other_cls in all_classes if other_cls != cls]
                    other_counts = all_counts - class_counts[cls]
                    other_mean = torch.stack(other_means, dim=0).sum(
                        dim=0) / other_counts  # shape: (graph.n_forward, d_model)
                    batch_means.append(other_mean)

                # shape: (batch_size, graph.n_forward, d_model)
                batch_means = torch.cat(batch_means, dim=0).to(activation_difference.device)

                activation_difference += batch_means

            elif 'mean' in intervention:
                activation_difference += means

            elif intervention == 'optimal':
                activation_difference += optimal_ablations

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_images)
            predictions.append(clean_logits.detach().cpu())

            input_activations_clean = input_activations_corrupted - activation_difference[:, :,
                                                                    graph.forward_index(graph.nodes['input'])]

            # Accumulate per-layer actdiff norms: activation_difference is now baseline - clean
            if return_actdiff_norms:
                for l in range(n_layers):
                    attn_diff = activation_difference[:, :, layer_attn_idx[l]].float()  # [B, pos, *, d]
                    mlp_diff  = activation_difference[:, :, layer_mlp_idx[l]].float()   # [B, pos, d]
                    # norm over the last dim, mean over remaining spatial/head dims, sum over batch
                    actdiff_norms[l] += (attn_diff.norm(dim=-1).mean(tuple(range(1, attn_diff.ndim - 1))).sum().item()
                                       + mlp_diff.norm(dim=-1).mean(tuple(range(1, mlp_diff.ndim - 1))).sum().item())

        if is_first_batch and profile_one_batch:
            t5 = time.perf_counter()
            print(f"  Forward passes (inference mode): {(t5-t4)*1000:.2f} ms")

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                            input_activations_clean - input_activations_corrupted)
                new_input.requires_grad = True
                return new_input

            return hook_fn

        if is_first_batch and profile_one_batch:
            ig_start_time = time.perf_counter()
        
        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            if is_first_batch and profile_one_batch:
                step_start = time.perf_counter()
            
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))],
                             bwd_hooks=bwd_hooks):
                logits = model(clean_images)
                metric_value = metric(logits, clean_logits, label)

                metric_value.backward()
            
            if is_first_batch and profile_one_batch:
                step_end = time.perf_counter()
                if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 5) == 0:
                    print(f"    IG step {step+1}/{steps}: {(step_end-step_start)*1000:.2f} ms")
        
        if is_first_batch and profile_one_batch:
            ig_end_time = time.perf_counter()
            print(f"  Total IG steps ({steps} steps): {(ig_end_time-ig_start_time)*1000:.2f} ms")
            batch_end_time = time.perf_counter()
            print(f"\nTotal batch time: {(batch_end_time-batch_start_time)*1000:.2f} ms")
            profiler.disable()
            
            # Print profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            print("\n" + "="*80)
            print("DETAILED PROFILING (Top 30 functions by cumulative time)")
            print("="*80)
            print(s.getvalue())
            
            # Also print by total time
            s2 = io.StringIO()
            ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
            ps2.print_stats(30)
            print("\n" + "="*80)
            print("DETAILED PROFILING (Top 30 functions by total time)")
            print("="*80)
            print(s2.getvalue())
            
            # Save profiling stats to file if task and model_id are provided
            if task is not None and model_id is not None:
                profile_dir = 'profiling_results'
                os.makedirs(profile_dir, exist_ok=True)
                profile_filename = f"batch_profile_{task.replace('_', '-')}_model{model_id}.txt"
                profile_output_path = os.path.join(profile_dir, profile_filename)
                
                with open(profile_output_path, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("BATCH PROFILING STATISTICS\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Task: {task}\n")
                    f.write(f"Model ID: {model_id}\n")
                    f.write(f"Batch size: {batch_size}\n")
                    f.write(f"IG steps: {steps}\n")
                    f.write(f"Total batch time: {(batch_end_time-batch_start_time)*1000:.2f} ms\n\n")
                    f.write("TIMING BREAKDOWN:\n")
                    f.write(f"  Data preparation: {(t1-t0)*1000:.2f} ms\n")
                    f.write(f"  make_hooks_and_matrices: {(t3-t2)*1000:.2f} ms\n")
                    f.write(f"  Forward passes (inference mode): {(t5-t4)*1000:.2f} ms\n")
                    f.write(f"  Total IG steps ({steps} steps): {(ig_end_time-ig_start_time)*1000:.2f} ms\n")
                    f.write(f"  Other: {max(0, (batch_end_time-batch_start_time)*1000 - ((t1-t0)*1000 + (t3-t2)*1000 + (t5-t4)*1000 + (ig_end_time-ig_start_time)*1000)):.2f} ms\n\n")
                    f.write("="*80 + "\n")
                    f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
                    f.write("="*80 + "\n")
                    ps_cum = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
                    ps_cum.print_stats(50)
                    f.write("\n" + "="*80 + "\n")
                    f.write("TOP 50 FUNCTIONS BY TOTAL TIME\n")
                    f.write("="*80 + "\n")
                    ps_tot = pstats.Stats(profiler, stream=f).sort_stats('tottime')
                    ps_tot.print_stats(50)
                
                print(f"\nProfiling stats saved to: {profile_output_path}")
            
            print("="*80 + "\n")

    scores /= total_items
    scores /= total_steps
    if get_perexample_scores:
        per_example_scores /= total_steps
        per_example_scores = {'scores': per_example_scores, 'labels': labels, 'logits': torch.cat(predictions)}
    else:
        per_example_scores = None

    if return_actdiff_norms:
        actdiff_norms_out = {l: v / total_items for l, v in actdiff_norms.items()}
        if return_logits:
            return scores, per_example_scores, clean_logits, actdiff_norms_out
        else:
            return scores, per_example_scores, actdiff_norms_out

    if return_logits:
        return scores, per_example_scores, clean_logits
    else:
        return scores, per_example_scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                              metric: Callable[[Tensor], Tensor], intervention: Literal[
            'patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                              steps=30, intervention_dataloader: Optional[DataLoader] = None,
                              optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, get_perexample_scores=False):
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
                means = pickle.load(file)
        else:
            per_position = 'positional' in intervention
            means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position, device=device)
            means = means.unsqueeze(0)
            if not per_position:
                means = means.unsqueeze(0)
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
                pickle.dump(means, file)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    per_example_scores = torch.zeros((len(dataloader.dataset), graph.n_forward, graph.n_backward), device='cpu', dtype=model.cfg.dtype) if get_perexample_scores else None
    labels = []
    predictions = []

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        labels.extend([this_label[0].item() if isinstance(this_label[0], torch.Tensor) else this_label[0] for this_label in label])

        clean_images = torch.stack(clean).to(device)
        if corrupted[0] is not None:
            corrupted_images = torch.stack(corrupted).to(device)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                     scores, per_example_scores=per_example_scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_images)

        elif 'mean' in intervention:
            activation_difference += means

        elif intervention == 'optimal':
            activation_difference += optimal_ablations

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_images)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()
            predictions.append(clean_logits.detach().cpu())

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output

            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1

                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_images)
                    metric_value = metric(logits, clean_logits, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps
    if get_perexample_scores:
        per_example_scores /= total_steps
        per_example_scores = {'scores': per_example_scores, 'labels': labels, 'logits': torch.cat(predictions)}
    else:
        per_example_scores = None

    return scores, per_example_scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                               metric: Callable[[Tensor], Tensor], quiet=False, device='cuda'):
    """Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        _type_: _description_
    """

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_images = torch.stack(clean).to(device)
        corrupted_images = torch.stack(corrupted).to(device)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                                           scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_images)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_images)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_images)
            metric_value = metric(logits, clean_logits, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_images)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                                       quiet=False, device='cuda') -> torch.Tensor:
    """Gets scores using Ferrando et al.'s (2024) information flow routes method.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        Tensor: scores based on information flow routes
    """
    # I could do some hacky overriding of make_hooks_and_matrices here but I will not
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                                         device=model.cfg.device, dtype=model.cfg.dtype)

        def output_hook(index, activations, hook):
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, output_activations[:, :, index].size(), output_activations.size())
                raise e

        # compute the score directly, without saving the input activations
        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts : batch pos backward hidden
                # output acts: batch pos forward hidden
                # add forward and backwards dimensions to acts and output acts respectively
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts : batch pos 1 backward hidden
                # output acts: batch pos forward 1 hidden
                proximity = torch.clamp(
                    - torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1,
                                               dim=-1) + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0)
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # aggregate over positions via sum/mean to get importance: forward backward
                # first mask out importances for padding positions
                max_len = input_lengths.max()
                mask = torch.arange(max_len, device=input_lengths.device,
                                    dtype=input_lengths.dtype).expand(len(input_lengths),
                                                                      max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                # print(importance.size(), mask.size())
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1, 1, 1)  # mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # squeezing backward dim in case it isn't real (i.e. it's an MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
                raise e

        hooks = []
        node = graph.nodes['input']
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))

        for layer in range(graph.cfg['n_layers']):
            node = graph.nodes[f'a{layer}.h0']
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate('qkv'):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

            node = graph.nodes[f'm{layer}']
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))

        node = graph.nodes['logits']
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_images = torch.stack(clean).to(device)
        input_lengths = torch.tensor([(model.cfg.image_size // model.cfg.patch_size)**2+1 for i in range(len(clean_images))]).to(device)

        hooks = make_hooks((model.cfg.image_size // model.cfg.patch_size)**2+1, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                _ = model(clean_images)

    scores /= total_items

    return scores


allowed_aggregations = {'sum', 'mean'}


def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
              method: Literal[
                  'EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'],
              intervention: Literal['patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
              aggregation='sum', get_perexample_scores=False,
              ig_steps: Optional[int] = None, intervention_dataloader: Optional[DataLoader] = None, return_logits=False,
              optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, model_id=None, profile_one_batch=False,
              return_actdiff_norms=False):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention,
                                intervention_dataloader=intervention_dataloader,
                                optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device, task=task, model_name=model_name, model_id=model_id, profile_one_batch=profile_one_batch)
    elif method == 'EAP-IG-inputs':
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention,
                                   intervention_dataloader=intervention_dataloader, get_perexample_scores=get_perexample_scores, return_logits=return_logits,
                                   optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device, task=task, model_name=model_name, model_id=model_id, profile_one_batch=profile_one_batch,
                                   return_actdiff_norms=return_actdiff_norms)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet, device=device)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention,
                                           intervention_dataloader=intervention_dataloader,
                                           optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device, task=task, model_name=model_name)
    elif method == 'information-flow-routes':
        scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet, device=device)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'exact-optimized':
        scores = get_scores_exact_optimized(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'exact-optimized-parallel':
        scores = get_scores_exact_optimized_parallel(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'random':
        scores = 2 * torch.rand_like(graph.scores) - 1
    else:
        raise ValueError(
            f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact', 'random'], but got {method}")

    if aggregation == 'mean':
        scores /= model.cfg.d_model
    perexample_scores = None
    actdiff_norms_out = None
    if isinstance(scores, tuple):
        if return_actdiff_norms and return_logits and len(scores) == 4:
            scores, perexample_scores, logits, actdiff_norms_out = scores
        elif return_actdiff_norms and len(scores) == 3:
            scores, perexample_scores, actdiff_norms_out = scores
            logits = None
        elif len(scores) > 2:
            scores, perexample_scores, logits = scores
        else:
            scores, perexample_scores = scores
            logits = None
    else:
        logits = None
    graph.scores[:] = scores.to(graph.scores.device)
    if perexample_scores:
        return perexample_scores, logits, actdiff_norms_out
    else:
        return None, logits, actdiff_norms_out

