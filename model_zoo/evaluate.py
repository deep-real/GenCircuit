import glob
import logging
import argparse
import pickle
import re

import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch.nn.parallel
from evaluation_utils.evaluate_acc import calculate_acc
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="help identify checkpoint")
    parser.add_argument("--dataset", choices=["waterbirds","cmnist","celebA"], default="waterbirds",
                        help="Which downstream task.")
    parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                        default="ViT",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint_dir",
                        help="directory of saved model checkpoint")
    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The directory where checkpoints are stored.")
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    def list_all_files(root_dir):
        file_list = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    # model_type = 'ViT-B_16'
    checkpoint_dir = "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output"
    result_dir = "/home/yxpengcs/PycharmProjects/vit-spurious-robustness/result"
    folder_list = os.listdir(checkpoint_dir)
    for folder in ['dataset_waterbirds-model_type_ViT-B_16-mae-img_size_224-train_batch_size_64-linear_probe_True-use_adam_False-learning_rate_0.009-weight_decay_0.0-num_steps_1000-warmup_steps_500-seed_0-batch_split_1-n_gpu_8-']:
        # if os.path.exists(f'figures/{folder}.png'):
        #     continue
        # if folder != 'new':
        #     continue
        ckpt_files = list_all_files(os.path.join(checkpoint_dir, folder))
        result_folder_path = os.path.join(result_dir, folder)

        x_vals = []
        acc_curves = {
            0: [],
            1: [],
            2: [],
            3: []
        }
        model_type = '_'.join(os.path.basename(ckpt_files[0]).split('_')[:-1])
        args.model_type = 'ViT-B_16-mae'
        for ckpt in ckpt_files:
            # Extract numeric id using regex
            # if ckpt != '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_2025-05-30_08:05:38/waterbirds/ViT/ViT-B_16-in21k_400.bin':
            #     continue
            match = re.search(fr"(?:start|(\d+))\.bin", os.path.basename(ckpt))
            args.checkpoint_dir = ckpt
            if match:
                if match.group(1):
                    step = int(match.group(1))
                else:
                    step = 0
                x_vals.append(step)

                # Evaluate (or load) metrics
                acc_dict = calculate_acc(args)

                # with open(os.path.join(result_folder_path, f'ste{step}.p'), 'wb') as file:
                #     pickle.dump(acc_dict, file)

                # Append each accuracy to corresponding curve
                for key in acc_curves:
                    acc_curves[key].append(acc_dict[key])

        # Sort by x_vals
        # x_vals, *acc_lists = zip(*sorted(zip(x_vals, *acc_curves.values())))
        # for i, key in enumerate(acc_curves):
        #     acc_curves[key] = acc_lists[i]
        #
        # # Plotting
        # plt.figure(figsize=(10, 6))
        # for key, accs in acc_curves.items():
        #     plt.plot(x_vals, accs, marker='o', label=key)
        #
        # plt.xlabel("Checkpoint Step")
        # plt.ylabel("Accuracy")
        # plt.title("Accuracy Curves vs Checkpoint")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'figures/{folder}.png')

if __name__=="__main__":
    main()