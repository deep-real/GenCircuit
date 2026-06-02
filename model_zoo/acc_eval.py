import argparse
import os.path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

from utils.data_utils import get_loader_train
from torch.nn import functional as F
import timm
import torch
from sklearn.metrics import balanced_accuracy_score

csv_path = "output/IN-set2_sweep_results.csv"
df = pd.read_csv(csv_path)

def compute_groupwise_accuracy(model, dataloader, device="cuda"):
    model.eval()
    correct_by_group = defaultdict(list)
    with torch.no_grad():
        for x, y, g in tqdm(dataloader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            for i in range(4):
                mask = (g == i)
                if mask.sum() > 0:
                    correct = (preds[mask] == y[mask]).float()
                    correct_by_group[i].extend(correct.cpu().tolist())

    accs = {i: sum(correct_by_group[i]) / len(correct_by_group[i]) if len(correct_by_group[i]) > 0 else float('nan')
            for i in range(4)}
    return accs

def compute_accuracy(model, dataloader, device="cuda", metric_name=''):
    model.eval()
    samples = 0
    correct = 0
    corr_where = []
    all_preds = torch.tensor([])
    all_y = torch.tensor([])
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y
            logits = model(x)
            preds = torch.argmax(logits, dim=-1).to('cpu')
            all_preds = torch.cat([all_preds, preds])
            all_y = torch.cat([all_y, y])
            correct += (preds == y).float().sum()
            corr_where.append(list(torch.where(preds == y)[0]))
            samples += len(y)

    all_pred_class = torch.unique(all_preds)
    metric = F1(prediction_fn=None, average='macro')
    f1 = metric.compute(all_preds, all_y)['F1-macro_all']
    accs = correct / samples
    return accs, f1

@torch.no_grad()
def compute_accuracy_and_extras(
    model,
    dataloader,
    device="cuda",
    metric_name='',
    f1_metric=None,          # pass a WILDS F1(metric) or sklearn-like metric if you want
    temperature: float = 1.0, # for energy
    atc_num_thresholds: int = 101
):
    model.eval()

    # Running sums/caches
    samples = 0
    correct_sum = 0.0
    corr_where = []

    # For post-hoc metrics
    all_logits = []
    all_preds  = []
    all_y      = []

    # For energy accumulation (to replicate your EMD snippet)
    energies_list = []

    for batch in tqdm(dataloader):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.to(device, non_blocking=True)
        # keep y on CPU for cheap stats/concat
        logits = model(x)                               # [B, C]
        preds  = torch.argmax(logits, dim=-1).cpu()     # [B]
        y_cpu  = y.cpu()

        # Basic accuracy pieces
        correct = (preds == y_cpu)
        correct_sum += correct.float().sum().item()
        samples += y_cpu.numel()
        # corr_where.append(list(torch.where(correct)[0].tolist()))

        # Collect for later metrics
        all_logits.append(logits.cpu())
        all_preds.append(preds)
        all_y.append(y_cpu)

        # Energy per-sample
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)  # [B]
        energies_list.append(energy.cpu())

    # Stack everything
    all_logits = torch.cat(all_logits, dim=0)  # [N, C]
    all_preds  = torch.cat(all_preds, dim=0).long()  # [N]
    all_y      = torch.cat(all_y, dim=0).long()      # [N]
    energies   = torch.cat(energies_list, dim=0)     # [N]

    metric = F1(prediction_fn=None, average='macro')
    f1 = metric.compute(all_preds, all_y)['F1-macro_all']
    bacc = balanced_accuracy_score(all_y, all_preds).item()
    acc = correct_sum / samples

    # Softmax probs for confidence/entropy
    probs = F.softmax(all_logits, dim=1)  # [N, C]
    max_conf, _ = probs.max(dim=1)        # [N]
    avg_confidence = max_conf.mean().item()

    # Negative entropy: sum_i p_i log p_i (i.e., -H)
    # clamp for stability
    log_probs = torch.log(probs.clamp_min(1e-12))
    neg_entropy = (probs * log_probs).sum(dim=1).mean().item()

    # Weight norm (global L2 over all params)
    with torch.no_grad():
        sq_sum = 0.0
        for p in model.parameters():
            if p is not None:
                sq_sum += float(torch.sum(p.detach()**2))
        weight_norm_l2 = (sq_sum ** 0.5)

    # Energy metrics
    avg_energy = energies.mean().item()

    # Your EMD-style score (replicates your snippet)
    # Note: this treats energies across the dataset as a "meta-distribution".
    # log_softmax over samples is unusual but we keep it to match your code.
    emd_score = torch.log_softmax(energies, dim=0).mean()
    emd_score = torch.log(-emd_score).item()  # as in your code

    return {
        "f1": f1,
        "acc": acc,
        "bacc": bacc,
        "avg_confidence": avg_confidence,
        "avg_negative_entropy": neg_entropy,
        "weight_norm_l2": weight_norm_l2,
        # "ATC_AUC": atc_auc,
        "avg_energy": avg_energy,
        "EMD_score": emd_score,
        # "corr_where": corr_where,
        "num_samples": samples,
    }

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
all_results = []
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--dataset", choices=["waterbirds", "cmnist", "celebA"], default="waterbirds",
                    help="Which downstream task.")
parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                    default="ViT",
                    help="Which variant to use.")
parser.add_argument("--model_type", default="ViT-B_16",
                    help="Which variant to use.")
parser.add_argument("--output_dir", default="output", type=str,
                    help="The output directory where checkpoints will be written.")
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
metric_name = 'acc'
device = 'cuda:7'

for i, row in df.iterrows():
    # if row.model_id < 6:
    #     continue
    # if row.model_id != 0:
    #     continue
    model_type = row.model_type
    model_name = model_dict[model_type]
    try:
        ckpt_path = row.checkpoint
    except:
        ckpt_path = None

    # Load model
    if ckpt_path:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=2,
            drop_rate=0.1,
            img_size=224
        )
        state_dict = torch.load(ckpt_path, map_location='cuda')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
        )
    model.to(device)

    # Setup args (must match your training setup)
    args.model_type = model_type
    args.train_batch_size = 32  # or row.train_batch_size * batch_split
    args.eval_batch_size = 32
    args.img_size = 224
    args.dataset = "IN-set2"  # or whatever you use
    args.batch_split = 1  # unless you used this in training
    domains = ['hospital1', 'hospital2']

    # Get data loaders
    train_loader, val_loader, test_loader = get_loader_train(args, model, test=True)

    # Compute accuracies
    # train_accs = compute_groupwise_accuracy(model, train_loader)
    # val_accs = compute_groupwise_accuracy(model, val_loader)
    # test_accs = compute_groupwise_accuracy(model, test_loader)
    # train_metrics = compute_accuracy_and_extras(model, train_loader, device=device)
    val_metrics = compute_accuracy_and_extras(model, val_loader, device=device)
    if isinstance(test_loader, list):
        test_metrics = []
        for this_test_loader in test_loader:
            test_metrics.append(compute_accuracy_and_extras(model, this_test_loader, metric_name=metric_name, device=device))
    elif isinstance(test_loader, dict):
        test_metrics = {}
        for name, this_test_loader in test_loader.items():
            test_metrics[name] = compute_accuracy_and_extras(model, this_test_loader, metric_name=metric_name, device=device)
    else:
        test_metrics = compute_accuracy_and_extras(model, test_loader, device=device)

    # Define tolerance for float comparisons
    # tol = 1e-4
    # mismatch = False
    #
    # if abs(val_metrics['acc'] - getattr(row, f"val_id_acc")) > tol:
    #     mismatch = True
    #     break
    #
    # if mismatch:
    #     print(f"[Mismatch] Updating model_id {row.model_id}, diff is {abs(val_metrics['acc'] - getattr(row, f'val_id_acc'))}")
    #     df.at[i, f"val_id_acc"] = val_metrics['acc']
    #     df.at[i, f"val_id_f1"] = val_metrics['f1']

    # print(
    #     f"Updating model_id {row.model_id}, diff is {abs(val_metrics['acc'] - getattr(row, f'val_id_acc'))}")
    df.at[i, f"id_acc"] = val_metrics['acc']
    df.at[i, f"id_bacc"] = val_metrics['bacc']
    df.at[i, f"id_f1"] = val_metrics['f1']
    df.at[i, f"id_AC"] = val_metrics['avg_confidence']
    df.at[i, f"id_ANE"] = val_metrics['avg_negative_entropy']
    df.at[i, f"id_l2"] = val_metrics['weight_norm_l2']
    df.at[i, f"id_EMD"] = val_metrics['EMD_score']

    # Add training accs to df
    # df.at[i, f"test_acc_id2"] = train_metrics['acc']
    # df.at[i, f"test_bacc_id2"] = train_metrics['bacc']
    # df.at[i, f"test_f1_id2"] = train_metrics['f1']
    # df.at[i, f"test_AC_id2"] = train_metrics['avg_confidence']
    # df.at[i, f"test_ANE_id2"] = train_metrics['avg_negative_entropy']
    # df.at[i, f"test_l2_id2"] = train_metrics['weight_norm_l2']
    # df.at[i, f"test_EMD_id2"] = train_metrics['EMD_score']


    if isinstance(test_metrics, list):
        for id, domain in enumerate(domains):
            df.at[i, f"test_acc_{domain}"] = test_metrics[id]['acc']
            df.at[i, f"test_bacc_{domain}"] = test_metrics[id]['bacc']
            df.at[i, f"test_f1_{domain}"] = test_metrics[id]['f1']
            df.at[i, f"test_AC_{domain}"] = test_metrics[id]['avg_confidence']
            df.at[i, f"test_ANE_{domain}"] = test_metrics[id]['avg_negative_entropy']
            df.at[i, f"test_l2_{domain}"] = test_metrics[id]['weight_norm_l2']
            df.at[i, f"test_EMD_{domain}"] = test_metrics[id]['EMD_score']
    elif isinstance(test_metrics, dict):
        for name, this_test_domain in test_metrics.items():
            df.at[i, f"test_acc_{name}"] = test_metrics[name]['acc']
            df.at[i, f"test_bacc_{name}"] = test_metrics[name]['bacc']
            df.at[i, f"test_f1_{name}"] = test_metrics[name]['f1']
            df.at[i, f"test_AC_{name}"] = test_metrics[name]['avg_confidence']
            df.at[i, f"test_ANE_{name}"] = test_metrics[name]['avg_negative_entropy']
            df.at[i, f"test_l2_{name}"] = test_metrics[name]['weight_norm_l2']
            df.at[i, f"test_EMD_{name}"] = test_metrics[name]['EMD_score']
    else:
        df.at[i, f"test_acc_ood"] = test_metrics['acc']
        df.at[i, f"test_bacc_ood"] = test_metrics['bacc']
        df.at[i, f"test_f1_ood"] = test_metrics['f1']
        df.at[i, f"test_AC_ood"] = test_metrics['avg_confidence']
        df.at[i, f"test_ANE_ood"] = test_metrics['avg_negative_entropy']
        df.at[i, f"test_l2_ood"] = test_metrics['weight_norm_l2']
        df.at[i, f"test_EMD_ood"] = test_metrics['EMD_score']

csv_path = "output/IN-set2_sweep_results_new_r.csv"
if os.path.exists(csv_path):
    full_df = pd.read_csv(csv_path)
    full_df.set_index('model_id', inplace=True)
    df.set_index('model_id', inplace=True)
    full_df.update(df)

    # For any new columns in df, we need to add them if they don't exist
    for col in df.columns:
        if col not in full_df.columns:
            full_df[col] = df[col]

    # Reset the index to turn the 'ID' index back into a column
    full_df.reset_index(inplace=True)
else:
    full_df = df
full_df.to_csv(csv_path, index=False)