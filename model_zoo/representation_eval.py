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
from reptrix import alpha, rankme, lidar

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

def get_features(encoder_network, dataloader, transform=None, num_augmentations=10, device='cuda'):
    # Loop over the dataset and collect the representations
    all_features = []

    # Loop over the dataset and collect the representations
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, 0)):
            if len(data) == 3:
                inputs, _, _ = data
            else:
                inputs, _ = data
            inputs = inputs.to(device)
            if transform:
                inputs = torch.cat([transform(inputs) for _ in range(num_augmentations)], dim=0)
            with torch.no_grad():
                features = encoder_network(inputs)
            if transform:
                # put the augmentations in an additonal dimension
                features = features.reshape(-1, num_augmentations, features.shape[1])
            all_features.append(features[:,0,:].detach().cpu()) # get CLS representation


    # Concatenate all the features
    all_features = torch.cat(all_features, dim=0)
    return all_features

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
    # 'terra-incognita-100',
    # 'camelyon17',
    # 'PACS-set2',
    # 'fmow-set2',
    # 'camelyon17-set2',
    'IN-set2'
]
from tqdm import tqdm
device = 'cuda:5'
for task_name in tests:
    all_rankme_vals_id = []
    all_alpha_vals_id = []
    all_rankme_vals_target = []
    all_alpha_vals_target = []
    csv_path = f"output/{task_name}_sweep_results_new_r.csv"
    df = pd.read_csv(csv_path)
    for i, row in tqdm(df.iterrows()):
        model_type = row.model_type
        model_name = model_dict[model_type]
        try:
            ckpt_path = row.checkpoint
        except:
            ckpt_path = None

        if task_name == "waterbirds" or "metashift" in task_name or "camelyon17" in task_name or task_name == 'yearbook':
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

        encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
        encoder.eval()
        if not 'set2' in task_name:
            all_representations = get_features(encoder, val_loader, device=device)
            metric_rankme = rankme.get_rankme(all_representations).item()
            metric_alpha = alpha.get_alpha(all_representations)[0].item()

            all_rankme_vals_id.append(metric_rankme)
            all_alpha_vals_id.append(metric_alpha)

        rankme_vals_target = {}
        alpha_vals_target = {}
        for domain_name, loader in test_loaders.items():
            all_representations = get_features(encoder, loader, device=device)
            metric_rankme = rankme.get_rankme(all_representations).item()
            metric_alpha = alpha.get_alpha(all_representations)[0].item()
            rankme_vals_target[domain_name] = metric_rankme
            alpha_vals_target[domain_name] = metric_alpha
        all_rankme_vals_target.append(rankme_vals_target)
        all_alpha_vals_target.append(alpha_vals_target)

    if not 'set2' in task_name:
        df[f'test_rankme_id'] = all_rankme_vals_id
        df[f'test_alphaReQ_id'] = all_alpha_vals_id

    all_rankme_vals_dict = {k: [d[k] for d in all_rankme_vals_target] for k in all_rankme_vals_target[0]}
    all_alpha_vals_dict = {k: [d[k] for d in all_alpha_vals_target] for k in all_alpha_vals_target[0]}

    for domain, rankme_list in all_rankme_vals_dict.items():
        df[f'test_rankme_{domain}'] = rankme_list

    for domain, alpha_list in all_alpha_vals_dict.items():
        df[f'test_alphaReQ_{domain}'] = alpha_list

    df.to_csv(csv_path)