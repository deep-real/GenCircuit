import argparse

import numpy as np
import timm
import torch
from torch import nn

from ATC_helpers.ATC_helper import get_entropy, find_ATC_threshold, get_ATC_acc
from ATC_helpers.model_helper import save_probs
from ATC_helpers.predict_acc_helper import inverse_softmax, softmax, get_doc, get_im_estimate
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

def inference(model, loader, device=torch.device('cuda')):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, data in enumerate(loader):
        if len(data) == 3:
            x, y, _ = data
        else:
            x, y = data
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = model(x, pre_logits=True)
            # import pdb; pdb.set_trace()
            # if type(h) is tuple:
                # h = h[-1]
            # if type(h) is dict:
            # h = h['feats']
            # h = model.projector(h)

        feature_vector.append(h.data.to(device))
        labels_vector.append(y.to(device))

    feature_vector = torch.cat(feature_vector)
    labels_vector = torch.cat(labels_vector)
    return feature_vector, labels_vector

def semantic_consistency(feats: torch.Tensor, labels: torch.Tensor, eps: float = 4.0):
    """
    feats:  (B, D) activations
    labels: (B,) integer class labels in [0, C-1]
    eps: activation threshold; use 0.0 for ReLU features, or 1e-5 for numerical safety

    Returns:
      per_dim: (D,) consistency per dimension (NaN for dims with 0 activated samples)
      mean_consistency: scalar mean over dims with >=1 activated sample
    """
    B, D = feats.shape
    labels = labels.to(torch.long)

    per_dim = torch.full((D,), float("nan"), device=feats.device)

    for j in range(D):
        mask = feats[:, j] > eps
        if mask.any():
            ys = labels[mask]
            # mode count
            counts = torch.bincount(ys)
            per_dim[j] = counts.max().float() / ys.numel()

    valid = ~torch.isnan(per_dim)
    mean_consistency = per_dim[valid].mean() if valid.any() else torch.tensor(float("nan"), device=feats.device)
    return mean_consistency

def eval_mono(test_loaders, model, device='cuda'):
    mono_vals = {}
    eps = 1e-2
    for domain_name, testloader in test_loaders.items():
        features, labels = inference(model, testloader, device=device)
        sp = semantic_consistency(features, labels)
        # pred_idx_new = np.argmax(probs_new, axis=-1)
        # pred_probs_new = np.max(probs_new, axis=-1)
        mono_vals[domain_name] = sp.item()
    return mono_vals



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
    'PACS',
    'PACS-photo',
    'PACS-cartoon',
    'PACS-art_painting',
    'terra-incognita-38',
    'terra-incognita-43',
    'terra-incognita-46',
    'terra-incognita-100',
    'camelyon17',
    'PACS-set2',
    'fmow-set2',
    'camelyon17-set2',
    'IN-set2',
]
from tqdm import tqdm
device = 'cuda:6'
for task_name in tests:
    all_atc_vals = []
    csv_path = f"output/{task_name}_sweep_results_new.csv"
    df = pd.read_csv(csv_path)
    sharpness_vals = []
    for i, row in tqdm(df.iterrows()):
        if 'PACS-set2' in task_name:
            if i > 0:
                continue
        model_type = row.model_type
        model_name = model_dict[model_type]
        try:
            ckpt_path = row.checkpoint
        except:
            ckpt_path = None

        if task_name == "waterbirds" or "metashift" in task_name or 'camelyon17' in task_name or task_name == 'yearbook':
            num_classes = 2
        elif "PACS" in task_name:
            num_classes = 7
        elif "terra-incognita" in task_name or "cifar10" in task_name:
            num_classes = 10
        elif "fmow" in task_name:
            num_classes = 62
        elif "iwildcam" in task_name:
            num_classes = 182
        elif 'IN' in task_name:
            num_classes = 1000

        # Load model
        if ckpt_path:
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
            model.load_state_dict(state_dict, strict=True)
        else:
            model = timm.create_model(
                model_name,
                pretrained=True,
            )
        model.to(device)
        args.model_type = model_type
        args.train_batch_size = 64  # or row.train_batch_size * batch_split
        args.eval_batch_size = 256
        args.img_size = 224
        args.dataset = f"{task_name}"  # or whatever you use
        args.batch_split = 1
        _, val_loader, test_loaders = get_loader_train(args, model)

        atc_vals = eval_mono(test_loaders, model, device=device)

        all_atc_vals.append(atc_vals)
    all_atc_vals_dict = {k: [d[k] for d in all_atc_vals] for k in all_atc_vals[0]}
    for domain, couplings_list in all_atc_vals_dict.items():
        col_name = f'test_mono_4_0_{domain}'

        if len(couplings_list) == 1:
            padded = [couplings_list[0]] + [np.nan] * (len(df) - 1)
            df[col_name] = padded
        else:
            df[col_name] = couplings_list

    df.to_csv(csv_path,index=False)