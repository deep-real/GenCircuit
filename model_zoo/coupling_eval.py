import argparse
from itertools import islice

import timm
import torch
from torch import nn
from tqdm import tqdm

from utils.data_utils import get_loader_train
import torch.nn.functional as F
from coupling_metrics import metrics
from coupling_jacobian import jacobian, svd

import sharpness
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", default=384, type=int,
                    help="Resolution size")
parser.add_argument("--train_batch_size", default=512, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument('--linear_probe', action='store_true', help="Enable linear probe")
parser.add_argument('--use_adam', action='store_true', help="Enable adam optimizer")

parser.add_argument("-lr", "--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=1500, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=0,
                    help="random seed for initialization")
parser.add_argument('--batch_split', type=int, default=16,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

args = parser.parse_args()

def vit_hidden_states_timm(model: torch.nn.Module, x: torch.Tensor):
    """
    Mirror HF ViTModel(..., output_hidden_states=True).hidden_states:
      - hs[0]: embedding output (after patch_embed + pos_embed + pos_drop)
      - hs[i+1]: output after transformer block i
    Works for standard timm VisionTransformer (with/without dist_token).
    Returns a tuple of tensors on the same device as x, shapes (B, T, C).
    """
    model.eval()
    hs = []

    # 1) Patch embedding
    # timm VisionTransformer has .patch_embed
    x = model.patch_embed(x)   # (B, num_patches, C)

    # 2) Add class/dist tokens if present
    B, N, C = x.shape
    if hasattr(model, "cls_token") and model.cls_token is not None:
        cls_tok = model.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tok, x), dim=1)           # (B, 1+N, C)

    # distilled variants have a distillation token
    if hasattr(model, "dist_token") and model.dist_token is not None:
        dist_tok = model.dist_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((x, dist_tok), dim=1)            # (B, 2+N, C)

    # 3) Add positional embeddings + dropout (this matches HF "embeddings" state)
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        x = x + model.pos_embed
    if hasattr(model, "pos_drop") and model.pos_drop is not None:
        x = model.pos_drop(x)

    # ---- HF hidden_states[0]: embeddings ----
    hs.append(x)

    # 4) Pass through transformer blocks, capturing output after each block
    # (timm blocks are pre-norm inside; we match HF by recording post-block, pre-final-norm)
    for blk in model.blocks:
        x = blk(x)
        hs.append(x)

    # NOTE: HF's hidden_states list does NOT include the final encoder LayerNorm.
    # timm applies model.norm after blocks; we intentionally don't append it.

    return hs

def coupling_from_hooks(hooks, p=2, num_sing_vecs=(10,30,50), index=-1, index_in=None,
    activation=None, chunks=4, verbose=False, device="cuda"):
    """
    Computes the coupling of residual Jacobians across hooks.

    hooks:      dict of representations before and after skip connection
    - hooks[layer] = {0: x_in, 1: x_out}
    p:              order of p-norm for coupling measurement
    num_sing_vecs:  number of top singular vectors to use in computing coupling
    index:          output token index for Jacobian
    index_in:       input token index for Jacobian
    - by default uses `index`
    activation: specifies whether to apply activation to `x_out` before computing Jacobian
    chunks:     number of chunks in Jacobian computation
    """
    Jac = []

    for h in hooks:
        # timestamp("computing J of: ", h) if verbose else None

        x_in = hooks[h][0]
        x_out = hooks[h][1]
        dim = x_out.shape[-1]

        if activation is None:
            J = jacobian(x_out, x_in, index=-1, device=device, chunks=chunks).detach()
            Jac.append(J - torch.eye(dim).expand(x_out.size(0), dim, dim).to(device))
            # timestamp("Jacobian shape ", J.shape) if verbose else None
        else:
            J = jacobian(activation(x_out), x_in, index=index, index_in=index_in, device=device).detach()
            Jac.append(J - torch.eye(dim).to(device))

    # timestamp("Computing coupling metrics") if verbose else None

    Us, Ss, Vs = svd(Jac)
    coupling_ujv, coupling_vju = metrics(Jac, Us, Ss, Vs, p=p, num_sing_vecs=num_sing_vecs)

    return coupling_ujv, coupling_vju

model_dict = {
    'ViT-B_16-in21k':'vit_base_patch16_224_in21k',
    'ViT-S_16':'vit_small_patch16_224_in21k',
    'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
    'ViT-B_16-clip-openai': 'vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-laion2b': 'vit_base_patch16_clip_224.laion2b',
    'ViT-B_16-mae': 'vit_base_patch16_224.mae',
    'ViT-B_16-dinov2': 'vit_base_patch14_dinov2.lvd142m',
    'ViT-B_16-in1k': 'vit_base_patch16_224.orig_in21k_ft_in1k',
    'ViT-B_16-in21k-hook':'vit_base_patch16_224_in21k',
    'ViT-B_16-scratch': 'vit_base_patch16_224_in21k'
}

tests = [
    # 'PACS',
    'PACS-photo',
    # 'PACS-cartoon',
    # 'PACS-art_painting',
    # 'terra-incognita-38',
    'camelyon17'
]

device = 'cuda:5'

K = 77

loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')

for task_name in tests:
    csv_path = f"output/{task_name}_sweep_results_new.csv"
    df = pd.read_csv(csv_path)
    average_couplings = []
    for i, row in df.iterrows():
        model_type = row.model_type
        model_name = model_dict[model_type]
        ckpt_path = row.checkpoint

        if task_name == "waterbirds" or "metashift" in task_name or task_name == "camelyon17" or task_name == 'yearbook':
            num_classes = 2
        elif "PACS" in task_name:
            num_classes = 7
        elif "terra-incognita" in task_name or "cifar10" in task_name:
            num_classes = 10
        elif "FMoW" in task_name:
            num_classes = 62
        elif "iwildcam" in task_name:
            num_classes = 182

        # Load model
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.1,
            img_size=224
        )
        state_dict = torch.load(ckpt_path, map_location='cuda')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        args.model_type = model_type
        args.train_batch_size = 64  # or row.train_batch_size * batch_split
        args.eval_batch_size = 32
        args.img_size = 224
        args.seed = row.seed
        args.linear_probe = row.linear_probe
        args.dataset = f"{task_name}"  # or whatever you use
        args.batch_split = 1
        _, val_loader, test_loader = get_loader_train(args, model)

        average_coupling = {}
        for domain, loader in test_loader.items():
            print(f'computing coupling for domain {domain} for model {i}')
            couplings = []
            for batch in tqdm(islice(loader, 4)):
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, _ = batch

                x = x.to(device)
                y = y.to(device)
                num_tokens = 197
                chunks = 2 * (num_tokens // 20) + 5

                hidden_states = vit_hidden_states_timm(model, x)

                # format as hooks
                outputs_zip = {}
                for j in range(12):
                    outputs_zip[f"block_{j}"] = {0: hidden_states[j], 1: hidden_states[j + 1]}

                # compute coupling
                coupling_ujv, coupling_vju = coupling_from_hooks(
                    outputs_zip, activation=None, chunks=chunks, index=0, verbose=True, device=device, num_sing_vecs=(K,),
                )

                couplings.append(coupling_ujv[K]['trace'].mean())
            average_coupling[domain] = torch.cat(couplings).mean().item()
        average_couplings.append(average_coupling)

    average_couplings_dict = {k: [d[k] for d in average_couplings] for k in average_couplings[0]}
    for domain, couplings_list in average_couplings_dict.items():
        df[f'{domain}_coupling'] = couplings_list

    df.to_csv(csv_path)