import argparse

import timm
import torch
from torch import nn

from utils.data_utils import get_loader_train
import torch.nn.functional as F

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

class LogitNormalizationWrapper(nn.Module):
    def __init__(self, model, normalize_logits=False):
        super(LogitNormalizationWrapper, self).__init__()
        self.model = model
        self.normalize_logits = normalize_logits

    def forward(self, x):
        out = self.model(x)
        if self.normalize_logits:
            out = out - out.mean(dim=-1, keepdim=True)
            out_norms = out.norm(dim=-1, keepdim=True)
            out_norms = torch.max(out_norms, 10**-10 * torch.ones_like(out_norms))
            out = out / out_norms
        return out

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
    # 'PACS-photo',
    # 'PACS-cartoon',
    # 'PACS-art_painting',
    # 'terra-incognita-38',
    # 'terra-incognita-43',
    # 'terra-incognita-46',
    'terra-incognita-100',
    # 'camelyon17'
]
device = 'cuda:6'
loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')

for task_name in tests:
    csv_path = f"output/{task_name}_sweep_results_new.csv"
    df = pd.read_csv(csv_path)
    sharpness_vals = []
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
        model = LogitNormalizationWrapper(model, normalize_logits=True)
        args.model_type = model_type
        args.train_batch_size = 64  # or row.train_batch_size * batch_split
        args.eval_batch_size = 256
        args.img_size = 224
        args.seed = row.seed
        args.linear_probe = row.linear_probe
        args.dataset = f"{task_name}"  # or whatever you use
        args.batch_split = 1
        _, val_loader, _ = get_loader_train(args, model)

        sharpness_obj, sharpness_err, _, output = sharpness.eval_APGD_sharpness(
            model, val_loader, loss_f,
            rho=0.001, n_iters=20, n_restarts=1, step_size_mult=1.0,
            rand_init=False, no_grad_norm=False,
            verbose=True, return_output=True, adaptive=True, version='default', norm='linf', device=device)

        sharpness_vals.append(sharpness_obj)
    df['sharpness'] = sharpness_vals

    df.to_csv(csv_path,index=False)