import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import einops

from vit_prisma.models.weight_conversion import convert_clip_weights, convert_timm_weights, convert_dino_weights

IMAGE_URLS = [
    "https://picsum.photos/id/237/300/300",  # Dog
    "https://picsum.photos/id/1025/300/300",  # Bird
    "https://picsum.photos/id/1003/300/300",  # Mountain
    "https://picsum.photos/id/1005/300/300",  # Forest
    "https://picsum.photos/id/1011/300/300",  # City
    "https://picsum.photos/id/1015/300/300",  # Bridge
    "https://picsum.photos/id/1016/300/300",  # River
    "https://picsum.photos/id/1024/300/300",  # Cat
    "https://picsum.photos/id/1020/300/300",  # Landscape
    "https://picsum.photos/id/1021/300/300",  # Field
]

model_type_dict = {
    'ViT-B_16-clip-openai': 'clip',
    'ViT-B_16-clip-openai-0.03': 'clip',
    'ViT-B_16-clip-laion2b': 'clip',
    'ViT-B_16-clip-laion2b-0.03': 'clip',
    'ViT-B_16-clip-openai-lp': 'clip',
    'ViT-B_16-clip-laion2b-lp': 'clip',
    'ViT-B_16-in21k': 'timm',
    'ViT-B_16-mae': 'timm',
    'ViT-B_16-in1k': 'timm',
    'ViT-B_16-in21k-lp': 'timm',
    'ViT-B_16-mae-lp': 'timm',
    'ViT-B_16-in1k-lp': 'timm',
    'ViT-B_16-dinov2': 'timm',
    'ViT-B_16-scratch': 'timm',
}

def convert_timm_to_hookedvit(timm_state_dict, cfg):
    """
    Convert a timm ViT state_dict to HookedViT-compatible format.

    Args:
        timm_state_dict (dict): State dict from a timm ViT model.
        cfg (HookedViTConfig): Configuration of the HookedViT model.

    Returns:
        dict: HookedViT-compatible state dict.
    """
    new_state_dict = {}

    new_state_dict["cls_token"] = timm_state_dict.get("cls_token", torch.zeros((1, 1, cfg.d_model)))
    new_state_dict["pos_embed.W_pos"] = timm_state_dict["pos_embed"].squeeze(0)
    new_state_dict["embed.proj.weight"] = timm_state_dict["patch_embed.proj.weight"]
    new_state_dict["embed.proj.bias"] = timm_state_dict.get("patch_embed.proj.bias", torch.zeros(cfg.d_model))

    new_state_dict["ln_pre.w"] = timm_state_dict["norm_pre.weight"]
    new_state_dict["ln_pre.b"] = timm_state_dict["norm_pre.bias"]
    new_state_dict["ln_final.w"] = timm_state_dict["norm.weight"]
    new_state_dict["ln_final.b"] = timm_state_dict["norm.bias"]

    for layer in range(cfg.n_layers):
        prefix = f"blocks.{layer}"

        new_state_dict[f"blocks.{layer}.ln1.w"] = timm_state_dict[f"{prefix}.norm1.weight"]
        new_state_dict[f"blocks.{layer}.ln1.b"] = timm_state_dict[f"{prefix}.norm1.bias"]
        new_state_dict[f"blocks.{layer}.ln2.w"] = timm_state_dict[f"{prefix}.norm2.weight"]
        new_state_dict[f"blocks.{layer}.ln2.b"] = timm_state_dict[f"{prefix}.norm2.bias"]

        # qkv projection
        qkv_weight = timm_state_dict[f"{prefix}.attn.qkv.weight"]
        qkv_bias = timm_state_dict[f"{prefix}.attn.qkv.bias"]

        qkv_weight = qkv_weight.reshape(3, cfg.d_model, cfg.d_model)
        qkv_bias = qkv_bias.reshape(3, cfg.d_model)

        W_Q = einops.rearrange(qkv_weight[0], '(h dh) d -> h d dh', h=cfg.n_heads, dh=cfg.d_head)
        W_K = einops.rearrange(qkv_weight[1], '(h dh) d -> h d dh', h=cfg.n_heads, dh=cfg.d_head)
        W_V = einops.rearrange(qkv_weight[2], '(h dh) d -> h d dh', h=cfg.n_heads, dh=cfg.d_head)

        b_Q = einops.rearrange(qkv_bias[0], '(h dh) -> h dh', h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(qkv_bias[1], '(h dh) -> h dh', h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(qkv_bias[2], '(h dh) -> h dh', h=cfg.n_heads, dh=cfg.d_head)

        W_O = einops.rearrange(timm_state_dict[f"{prefix}.attn.proj.weight"], 'd (h dh) -> h dh d', h=cfg.n_heads, dh=cfg.d_head)
        b_O = timm_state_dict[f"{prefix}.attn.proj.bias"]

        new_state_dict[f"blocks.{layer}.attn.W_Q"] = W_Q
        new_state_dict[f"blocks.{layer}.attn.W_K"] = W_K
        new_state_dict[f"blocks.{layer}.attn.W_V"] = W_V
        new_state_dict[f"blocks.{layer}.attn.b_Q"] = b_Q
        new_state_dict[f"blocks.{layer}.attn.b_K"] = b_K
        new_state_dict[f"blocks.{layer}.attn.b_V"] = b_V
        new_state_dict[f"blocks.{layer}.attn.W_O"] = W_O
        new_state_dict[f"blocks.{layer}.attn.b_O"] = b_O

        # MLP
        W_in = einops.rearrange(timm_state_dict[f"{prefix}.mlp.fc1.weight"], 'm d -> d m')
        b_in = timm_state_dict[f"{prefix}.mlp.fc1.bias"]
        W_out = einops.rearrange(timm_state_dict[f"{prefix}.mlp.fc2.weight"], 'd m -> m d')
        b_out = timm_state_dict[f"{prefix}.mlp.fc2.bias"]

        new_state_dict[f"blocks.{layer}.mlp.W_in"] = W_in
        new_state_dict[f"blocks.{layer}.mlp.b_in"] = b_in
        new_state_dict[f"blocks.{layer}.mlp.W_out"] = W_out
        new_state_dict[f"blocks.{layer}.mlp.b_out"] = b_out

    # Head
    if 'head.weight' in timm_state_dict:
        new_state_dict["head.W_H"] = einops.rearrange(timm_state_dict["head.weight"], "c d -> d c")
        new_state_dict["head.b_H"] = timm_state_dict["head.bias"]

    return new_state_dict


def convert_weights(
    original_weights,
    model_name,
    config,
):
    """
    Convert weights for a specific model.

    Args:
        original_weights: Original weights
        model_name: Model name
        category: Model category
        config: Model configuration
        model_type: Model type

    Returns:
        Converted weights in Prisma format
    """
    # Special case for EVA02 models - use TIMM converter
    category = model_type_dict[model_name]

    # Special case for CLIP models that need unpacking
    # if category == 'clip':
    #     # vision_weights = original_weights.vision_model.state_dict()
    #     # projection_weights = original_weights.visual_projection.state_dict()
    #     # return convert_clip_weights(vision_weights, projection_weights, config)
    #     converter = convert_timm_to_hookedvit

    # Get appropriate converter based on category and type
    if category == 'timm':
        converter = convert_timm_weights
    elif category == 'clip':
        converter = convert_timm_to_hookedvit
    elif category == 'dino':
        converter = convert_dino_weights
    elif category == ModelCategory.OPEN_CLIP:
        converter = (
            convert_open_clip_text_weights
            if model_type == ModelType.TEXT
            else convert_open_clip_weights
        )
    elif category == ModelCategory.VIVIT:
        converter = convert_vivet_weights
    elif category == ModelCategory.VJEPA:
        converter = convert_vjepa_weights
    elif category == ModelCategory.KANDINSKY:
        converter = convert_kandinsky_clip_weights
    else:
        raise ValueError(f"No converter available for {category} with {model_type}")

    # Apply converter
    return converter(original_weights, config)

def compare_model_outputs(model_original, model_hooked, device="cpu", metric="abs"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model_original.eval()
    model_hooked.eval()

    divergences = []

    with torch.no_grad():
        for url in IMAGE_URLS:
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                out1 = model_original(input_tensor)
                out2 = model_hooked(input_tensor)

                if hasattr(out1, "logits"):
                    out1 = out1.logits
                    out2 = out2.logits

                if metric == "abs":
                    divergence = torch.mean(torch.abs(out1 - out2)).item()
                elif metric == "mse":
                    divergence = torch.nn.functional.mse_loss(out1, out2).item()
                else:
                    raise ValueError("Unsupported metric")

                divergences.append(divergence)
            except Exception as e:
                print(f"Failed to process {url}: {e}")

    mean_div = np.mean(divergences)
    print(f"Mean {metric} divergence over {len(divergences)} images: {mean_div:.6f}")
    return mean_div