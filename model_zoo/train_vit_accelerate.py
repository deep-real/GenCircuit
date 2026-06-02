
import os
import random
from collections import Counter

import numpy as np
from datetime import datetime
from datetime import timedelta
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

import pandas as pd
# from vit_prisma.models.base_vit import HookedViT
import torch
from tqdm import trange

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader_train
from utils.dist_util import get_world_size
import timm
# from apex.parallel import DistributedDataParallel as DDP
from utils.comm_utils import set_seed, AverageMeter, accuracy_func
import math
import logging
logger = logging.getLogger(__name__)

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
    'ViT-B_16-scratch': 'random'
}

def save_model(args, model, args_str, step='final', accelerator=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint_dir = os.path.join(args.output_dir, args_str)
    checkpoint_path = os.path.join(model_checkpoint_dir, f"{step}" + ".bin")
    if accelerator.is_main_process:
        if os.path.exists(checkpoint_path) != True:
             os.makedirs(model_checkpoint_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), checkpoint_path)
    return checkpoint_path


def setup(args):
    if args.dataset == "waterbirds" or "metashift" in args.dataset or "camelyon17" in args.dataset or args.dataset == 'yearbook':
        num_classes = 2
    elif "PACS" in args.dataset:
        num_classes = 7
    elif "terra-incognita" in args.dataset or "cifar10" in args.dataset:
        num_classes = 10
    elif "fmow" in args.dataset:
        num_classes = 62
    elif "iwildcam" in args.dataset:
        num_classes = 182
    model_name =model_dict[args.model_type]
    if 'hook' in args.model_type:
        model = HookedViT.from_pretrained(
                    "vit_base_patch16_224",
                    center_writing_weights=False,
                    center_unembed=False,
                    fold_ln=False,
                    refactor_factored_attn_matrices=False,
                )
    elif args.model_type == 'ViT-B_16-scratch':
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            drop_rate = 0.1,
            num_classes=num_classes,
            img_size=args.img_size
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate = 0.1,
            img_size = args.img_size
        )
        model.reset_classifier(num_classes)
    if args.linear_probe:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def valid_groups(args, model, test_loader, global_step, accelerator, num_classes=2):
    # Validation!
    # eval_losses = AverageMeter()

    model.eval()
    all_preds, all_labels, all_gs = [], [], []
    epoch_iterator = tqdm(test_loader) if accelerator.is_main_process else test_loader
    # loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y, g = batch;
        with torch.no_grad():
            logits = model(x)
            # eval_loss = loss_fct(logits, y)
            # eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        preds = accelerator.gather_for_metrics(preds)
        y = accelerator.gather_for_metrics(y)
        g = accelerator.gather_for_metrics(g)
        all_preds.append(preds)
        all_labels.append(y)
        all_gs.append(g)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_gs = torch.cat(all_gs)

    correct = (all_preds == all_labels)

    wb_group_train_sizes = np.array([3498, 184, 56, 1057], dtype=np.float32)
    wb_group_train_sizes /= wb_group_train_sizes.sum()

    accs = {}
    for i in range(4):
        mask = all_gs == i
        group_correct = correct[mask]
        accs[i] = group_correct.float().mean() if mask.sum() > 0 else float('nan')

    # Mean weighted accuracy using training distribution
    acc_array = np.array([accs[i].detach().cpu() for i in range(4)], dtype=np.float32)
    mean_weighted_acc = np.sum(acc_array * wb_group_train_sizes)

    return mean_weighted_acc, acc_array

def valid(args, model, test_loader, global_step, accelerator, num_classes, metric_name):
    # Validation!
    # eval_losses = AverageMeter()

    model.eval()
    all_preds, all_labels, all_gs = [], [], []
    epoch_iterator = tqdm(test_loader) if accelerator.is_main_process else test_loader
    # loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        g = None
        if len(batch) == 2:
            x, y = batch;
        elif len(batch[2].shape) > 1:
            x, y, _ = batch
        else:
            x, y, g = batch
        with torch.no_grad():
            logits = model(x)
            # eval_loss = loss_fct(logits, y)
            # eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        preds = accelerator.gather_for_metrics(preds)
        y = accelerator.gather_for_metrics(y)
        g = accelerator.gather_for_metrics(g) if g is not None else None
        all_preds.append(preds)
        all_labels.append(y)
        all_gs.append(g)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_gs = torch.cat(all_gs) if g is not None else None

    metric = F1(prediction_fn=None, average='macro')
    mean_F1 = metric.compute(all_preds.detach().cpu(), all_labels.detach().cpu())['F1-macro_all']
    correct = (all_preds == all_labels)
    mean_acc = correct.float().mean().item()

    # return mean_acc, acc_array
    del all_preds, all_labels, all_gs
    torch.cuda.empty_cache()
    return mean_acc, mean_F1

def compute_class_weight(loader, args):
    label_counts = Counter()

    for batch in loader:
        _, labels = batch  # assuming batch = (inputs, labels)
        label_counts.update(labels.tolist())

    # Step 2: Compute class weights (inverse frequency)
    num_classes = len(label_counts)
    total_samples = sum(label_counts.values())
    class_weights = torch.zeros(num_classes, dtype=torch.float)

    for cls, count in label_counts.items():
        class_weights[cls] = total_samples / (num_classes * count)

    # Optional: Normalize to sum to 1 (not required by PyTorch, but helps with interpretability)
    class_weights /= class_weights.sum()

    # Step 3: Move to device and create the loss function with class weights
    class_weights = class_weights.to(args.device)
    return class_weights

def train_model(args):
    import wandb
    from accelerate import Accelerator
    accelerator = Accelerator()
    # lrs = [0.003, 0.01, 0.03]
    # bzs = [256]
    # wds = [0.01]
    # use_class_weight = False
    # use_adams = [False]
    # linear_probe = [True, False]
    # warmups = [16,20]
    # seeds = [0]
    # pretrains = ['ViT-B_16-clip-openai', 'ViT-B_16-scratch', 'ViT-B_16-clip-laion2b', "ViT-B_16-mae", 'ViT-B_16-in21k', 'ViT-B_16-in1k']
    lrs = [0.01]
    bzs = [256]
    wds = [0]
    use_class_weight = False
    use_adams = [False]
    linear_probe = [False]
    warmups = [16]
    seeds = [0]
    pretrains = ['ViT-B_16-clip-laion2b']
    metric_name = 'acc'
    test_domain_names = ['sketch', 'photo', 'cartoon']
    csv_path = os.path.join(args.output_dir, f"{args.dataset}_sweep_results.csv")
    valid_func = valid_groups if args.dataset == "waterbirds" else valid
    if os.path.exists(csv_path):
        current_df = pd.read_csv(csv_path)
        model_id = current_df["model_id"].max() + 1
    else:
        model_id = 0
    for model_name in pretrains:
        for lp in linear_probe:
            if lp and model_name == "ViT-B_16-scratch":
                continue
            for warmup in warmups:
                for use_adam in use_adams:
                    for lr in lrs:
                        for bz in bzs:
                            for wd in wds:
                                for seed in seeds:
                                    args.linear_probe = lp
                                    args.model_type = model_name
                                    args.learning_rate = lr
                                    args.weight_decay = wd
                                    args.train_batch_size = bz
                                    args.use_adam = use_adam
                                    args.seed = seed
                                    args.warmup_steps = warmup
                                    args.use_class_weight = use_class_weight
                                    exclude_keys = {"local_rank", "name", "model_arch", "output_dir", "eval_batch_size", "eval_every", "device", "max_grad_norm", "save_every", "img_size", "batch_split"}  # example fields to exclude
                                    args_str = "".join([f"{k}_{v}-" for k, v in vars(args).items() if k not in exclude_keys])
                                    if os.path.exists(os.path.join(args.output_dir, args_str)):
                                        continue
                                    args, model = setup(args)
                                    # now = datetime.now()
                                    # time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

                                    if wandb.run is not None:
                                        wandb.finish()
                                    if accelerator.is_main_process:
                                        wandb.init(project="vit-sweep", name=args_str, config=vars(args), reinit=True, resume=False)
                                    save_model(args, model, args_str, 0, accelerator)
                                    args.train_batch_size = args.train_batch_size // args.batch_split
                                    train_loader, val_loader, test_loader = get_loader_train(args, model)
                                    if args.use_class_weight:
                                        class_weights = compute_class_weight(train_loader, args)
                                        cri = torch.nn.CrossEntropyLoss(weight=class_weights)
                                    else:
                                        cri = torch.nn.CrossEntropyLoss()
                                    # Prepare optimizer and scheduler
                                    if args.use_adam:
                                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
                                    else:
                                        optimizer = torch.optim.SGD(model.parameters(),
                                                                    lr=args.learning_rate,
                                                                    momentum=0.9,
                                                                    weight_decay=args.weight_decay)
                                    t_total = args.num_steps
                                    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
                                    # Distributed training
                                    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
                                        model, optimizer, train_loader, val_loader, test_loader
                                    )

                                    model.zero_grad()
                                    set_seed(args)
                                    losses = AverageMeter()
                                    global_step, best_acc, best_f1 = 0, 0, 0
                                    progress_bar = trange(global_step, t_total, disable=not accelerator.is_main_process)
                                    while True:
                                        model.train()
                                        epoch_iterator = train_loader
                                        for step, batch in enumerate(epoch_iterator):
                                            batch = tuple(t for t in batch)
                                            if len(batch) == 3:
                                                x, y, _ = batch;
                                            else:
                                                x, y = batch
                                            logits = model(x)
                                            loss = cri(logits, y.view(-1))
                                            if args.batch_split > 1:
                                                loss = loss / args.batch_split

                                            accelerator.backward(loss)

                                            if (step + 1) % args.batch_split == 0:
                                                losses.update(loss.item()*args.batch_split)
                                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                                                scheduler.step()
                                                optimizer.step()
                                                optimizer.zero_grad()
                                                global_step += 1
                                                progress_bar.update(1)
                                                if accelerator.is_main_process:
                                                    wandb.log({"train/loss": losses.val, "lr": scheduler.get_last_lr()[0]}, step=global_step)
                                                    # wandb.log({"train/loss": losses.val}, step=global_step)

                                                if global_step % args.eval_every == 0:
                                                # if global_step % args.eval_every == 0 or (global_step < 10 and global_step % 2 == 0):
                                                # if global_step % args.eval_every == 0 or (global_step < 100 and global_step % 20 == 0):
                                                    val_id_accuracy, val_id_F1 = valid_func(args, model, val_loader, global_step, accelerator, num_classes=model.module.num_classes if hasattr(model, "module") else model.num_classes, metric_name=metric_name)
                                                    # test_id_accuracy, test_accuracy_by_group = valid_func(args, model, test_loader, global_step, accelerator, num_classes=model.module.num_classes if hasattr(model, "module") else model.num_classes)
                                                    ckpt_path = save_model(args, model, args_str, global_step, accelerator)
                                                    if metric_name == 'F1':
                                                        if best_f1 < val_id_F1:
                                                            best_ckpt_path = ckpt_path
                                                            best_f1 = val_id_F1
                                                            best_acc = val_id_accuracy
                                                    else:
                                                        if best_acc < val_id_accuracy:
                                                            best_ckpt_path = ckpt_path
                                                            best_acc = val_id_accuracy
                                                            best_f1 = val_id_F1
                                                        # best_test_id_accuracies = test_id_accuracies
                                                    if accelerator.is_main_process:
                                                        log_dict = {
                                                            "accuracies/val_id_acc": val_id_accuracy,
                                                            "accuracies/val_id_f1": val_id_F1,
                                                        }
                                                        # for domain, acc in test_id_accuracies.items():
                                                        #     log_dict[f"accuracies/test_acc_{domain}"] = acc
                                                        wandb.log(log_dict, step=global_step)

                                                    model.train()

                                                if global_step % t_total == 0:
                                                    break
                                        losses.reset()
                                        if global_step % t_total == 0:
                                            break
                                    accelerator.wait_for_everyone()
                                    best_ckpt = torch.load(best_ckpt_path)
                                    accelerator.unwrap_model(model).load_state_dict(best_ckpt)
                                    best_test_id_accuracies = {}
                                    best_test_id_F1s = {}
                                    if isinstance(test_loader, list):
                                        for loader, domain_name in zip(test_loader, test_domain_names):
                                            test_acc, test_F1 = valid_func(
                                                args, model, loader, global_step, accelerator,
                                                num_classes=model.module.num_classes if hasattr(model,
                                                                                                "module") else model.num_classes,
                                                metric_name='acc'
                                            )
                                            best_test_id_accuracies[domain_name] = test_acc
                                            best_test_id_F1s[domain_name] = test_F1
                                    elif isinstance(test_loader, dict):
                                        for domain_name, loader in test_loader.items():
                                            test_acc, test_F1 = valid_func(
                                                args, model, loader, global_step, accelerator,
                                                num_classes=model.module.num_classes if hasattr(model,
                                                                                                "module") else model.num_classes,
                                                metric_name='acc'
                                            )
                                            best_test_id_accuracies[domain_name] = test_acc
                                            best_test_id_F1s[domain_name] = test_F1
                                    else:
                                        test_acc, test_F1 = valid_func(
                                                args, model, test_loader, global_step, accelerator,
                                                num_classes=model.module.num_classes if hasattr(model,
                                                                                                "module") else model.num_classes,
                                                metric_name='acc'
                                            )
                                        best_test_id_accuracies["id"] = test_acc
                                        best_test_id_F1s["id"] = test_F1
                                    write_header = not os.path.exists(csv_path)
                                    result = {
                                        "model_id": model_id,
                                        "model_type": args.model_type,
                                        "learning_rate": args.learning_rate,
                                        "train_batch_size": args.train_batch_size,
                                        "use_adam": use_adam,
                                        "weight_decay": args.weight_decay,
                                        "warmup_steps": args.warmup_steps,
                                        "num_steps": args.num_steps,
                                        "linear_probe": args.linear_probe,
                                        "seed": args.seed,
                                        "val_id_acc": best_acc,
                                        "val_id_f1": best_f1,
                                        "checkpoint": best_ckpt_path,
                                    }

                                    for domain, acc in best_test_id_accuracies.items():
                                        result[f"test_acc_{domain}"] = acc
                                    for domain, acc in best_test_id_F1s.items():
                                        result[f"test_f1_{domain}"] = acc
                                    # for i, acc in enumerate(best_test_accuracy_by_group):
                                    #     result[f"test_acc_{i}"] = acc
                                    # for i, acc in enumerate(best_val_accuracy_by_group):
                                    #     result[f"val_acc_{i}"] = acc
                                    model_id += 1
                                    if accelerator.is_main_process:
                                        if os.path.exists(csv_path):
                                            existing_df = pd.read_csv(csv_path, nrows=1)
                                            columns = existing_df.columns.tolist()
                                        else:
                                            columns = list(result.keys())

                                        df = pd.DataFrame([result])[columns]  # enforce column order
                                        df.to_csv(csv_path, mode='a', header=write_header, index=False)
                                        wandb.finish()
