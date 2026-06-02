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

def eval_ATC(val_loader, test_loaders, model, device='cuda'):
    probsv1, labelsv1 = save_probs(model, val_loader, device)
    pred_idxv1 = np.argmax(probsv1, axis=-1)
    pred_probsv1 = np.max(probsv1, axis=-1)
    v1acc = np.mean(pred_idxv1 == labelsv1) * 100.

    try:
        calibrator = TempScaling()
        calibrator.fit(inverse_softmax(probsv1), labelsv1)
    except:
        class Calibration:
            pass

        calibrator = Calibration()
        calibrator.calibrate = lambda x: x

    calib_probsv1 = softmax(inverse_softmax(probsv1))

    calib_pred_idxv1 = np.argmax(calib_probsv1, axis=-1)
    calib_pred_probsv1 = np.max(calib_probsv1, axis=-1)

    # entropy = get_entropy(probsv1)
    calib_entropy = get_entropy(calib_probsv1)

    # _, entropy_thres_balance = find_threshold_balance(entropy, pred_idxv1 == labelsv1 )
    _, calib_entropy_thres_balance = find_ATC_threshold(calib_entropy, calib_pred_idxv1 == labelsv1)
    # _, thres_balance = find_threshold_balance(pred_probsv1, pred_idxv1 == labelsv1 )
    _, calib_thres_balance = find_ATC_threshold(calib_pred_probsv1, calib_pred_idxv1 == labelsv1)

    atc_vals = {}

    for domain_name, testloader in test_loaders.items():
        probs_new, labels_new = save_probs(model, testloader, device)

        # pred_idx_new = np.argmax(probs_new, axis=-1)
        # pred_probs_new = np.max(probs_new, axis=-1)

        calib_probs_new = softmax(inverse_softmax(probs_new))
        # calib_pred_idx_new = np.argmax(calib_probs_new, axis=-1)
        calib_pred_probs_new = np.max(calib_probs_new, axis=-1)

        # import pdb; pdb.set_trace()
        # entropy_new = get_entropy(probs_new)
        # calib_entropy_new = get_entropy(calib_probs_new)

        # entropy_pred_balance = get_acc(entropy_thres_balance, entropy_new)
        # entropy_conf_balance = num_corr(pred_idx_new, entropy_new, entropy_thres_balance, labels_new)

        # calib_entropy_pred_balance = get_ATC_acc(calib_entropy_thres_balance, calib_entropy_new)
        # calib_entropy_conf_balance = num_corr(calib_pred_idx_new, calib_entropy_new, calib_entropy_thres_balance, labels_new)

        # pred_balance = get_acc(thres_balance, pred_probs_new)
        # conf_balance = num_corr(pred_idx_new, pred_probs_new, thres_balance, labels_new)

        calib_pred_balance = get_ATC_acc(calib_thres_balance, calib_pred_probs_new)
        # calib_conf_balance = num_corr(calib_pred_idx_new, calib_pred_probs_new, calib_thres_balance, labels_new)

        # test_acc = np.mean(pred_idx_new == labels_new) * 100.0
        # # calib_test_acc =  np.mean(calib_pred_idx_new == labels_new)*100.0
        #
        # # avg_conf = np.mean(pred_probs_new)*100.0
        # calib_avg_conf = np.mean(calib_pred_probs_new) * 100.0
        #
        # # doc_feat = v1acc + get_doc(pred_probsv1, pred_probs_new)*100.0
        # calib_doc_feat = v1acc + get_doc(calib_pred_probsv1, calib_pred_probs_new) * 100.0
        #
        # # im_estimate = get_im_estimate(pred_probsv1, pred_probs_new, (pred_idxv1 == labelsv1))
        # calib_im_estimtate = get_im_estimate(calib_pred_probsv1, calib_pred_probs_new, (calib_pred_idxv1 == labelsv1))

        # with open(logFile, "a") as f:
        # 	f.write(("{:.4f}, {:.4f}, {:.4f},{:.4f}, {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}," + \
        # 	"{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}," + \
        # 	"{:.4f}\n").format(test_acc, calib_test_acc, pred_balance, conf_balance,
        # 	entropy_pred_balance, entropy_conf_balance, calib_pred_balance, calib_conf_balance, \
        # 	calib_entropy_pred_balance, calib_entropy_conf_balance,\
        # 	avg_conf, calib_avg_conf, doc_feat, calib_doc_feat, im_estimate, calib_im_estimtate))

        atc_vals[domain_name] = calib_pred_balance.item()
    return atc_vals



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
    # 'camelyon17'
    # 'PACS-set2',
    # 'fmow-set2',
    # 'camelyon17-set2',
    'IN-set2',
]
from tqdm import tqdm
device = 'cuda:6'
for task_name in tests:
    all_atc_vals = []
    csv_path = f"output/{task_name}_sweep_results_new_r.csv"
    df = pd.read_csv(csv_path)
    sharpness_vals = []
    for i, row in tqdm(df.iterrows()):
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

        atc_vals = eval_ATC(val_loader, test_loaders, model, device=device)

        all_atc_vals.append(atc_vals)
    all_atc_vals_dict = {k: [d[k] for d in all_atc_vals] for k in all_atc_vals[0]}
    for domain, couplings_list in all_atc_vals_dict.items():
        df[f'test_ATC_{domain}'] = couplings_list

    df.to_csv(csv_path,index=False)