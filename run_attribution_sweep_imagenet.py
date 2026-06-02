import argparse
import os
import pickle
from functools import partial

import numpy as np

from dataset import VisionEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node

from dataset_utils import setup_dataset
from metrics import get_metric

from model_utils import get_model, get_model_from_ckpt, model_dict, get_model_from_name

TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B"
}

if __name__ == "__main__":
    model_name = "ViT-B_16-in1k"
    import pandas as pd

    get_perexample_scores = False
    circuit_dir = 'circuits'
    fragment = None
    batch_size = 30
    metric_name = 'kl_divergence'
    method = 'exact'
    ablation = 'mean-positional'
    ig_steps = 5
    optimal_ablation_path = None
    device = 'cuda:6'
    level = 'edge'
    profile_one_batch = False  # Set to True to profile computation bottleneck for one batch
    # task = 'fmow-mean-id'
    tasks = ['IN-set2-mean-imagenet-r']
    # tasks = []
    # for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    #     tasks.append(f'metashift-control-{i}')
    # cifar_c = ["fog", "frost", "motion_blur"]
    # cifar_c = ["defocus_blur"]
    # severities = [1, 2, 3, 4, 5]
    # for data in cifar_c:
    #     if data == "id":
    #         tasks.append('cifar10-mean-id')
    #     else:
    #         for severity in severities:
    #             tasks.append(f'cifar10-mean-{data}-{severity}')
    # for i in range(5):
    #     tasks.append(f'fmow-mean-time1-region{i}')
    # wildcam_df = pd.read_csv('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/iwildcam_sweep_results_new.csv')
    # acc_cols = [c for c in wildcam_df.columns if c.startswith("test_acc")]
    # tasks = [f"iwildcam-mean-{col.replace('test_acc_', '').replace('_', '-')}" for col in acc_cols]
    # for domain in ['photo', 'cartoon', 'art_painting']:
    #     for split in ['train', 'val', 'test']:
    #         tasks.append(f"PACS-set2-mean-{domain}-{split}")
    # for i in range(0,2):
    #     for slide in [i * 10 + 8, i * 10 + 9]:
    #         tasks.append(f"camelyon17-set2-mean-hospital{i}_slide{slide}")
    # i = 1
    # for j in range(4,8):
    #     tasks.append(f"camelyon17-set2-mean-hospital{i}_slide{i*10+j}")
    # for i in range(2,5):
    #     for j in range(0,10):
    #         tasks.append(f"camelyon17-set2-mean-hospital{i}_slide{i*10+j}")
    # for corruption in ['motion_blur', 'zoom_blur', 'defocus_blur', 'snow', 'fog', 'frost', 'gaussian_noise', 'shot_noise']:
    # for corruption in ['snow']:
    #     for severity in [1,2,3,4,5]:
    #         tasks.append(f'camelyon17-set2-mean-{corruption}-{severity}-corrupt')
    # for corruption in ['motion_blur', 'defocus_blur', 'zoom_blur', 'glass_blur']:
    #     for severity in [1,2,3,4,5]:
    #         tasks.append(f'IN-set2-mean-imagenet-c-{corruption}-{severity}')
    # for corruption in ['gaussian_noise']:
    #     for severity in np.linspace(0.0, 1.0, 21):
    #         if severity == 0:
    #             continue
    #         tasks.append(f'camelyon17-set2-mean-{corruption}-{severity}-sensitive')
    # for transform in ['edge_stylize','cartoon_stylize','countour_stylize','emboss_stylize', 'edge_enhance_stylize','pallete_stylize','posterize_stylize', 'solarize_stylize']:
    #     tasks.append(f'camelyon17-set2-mean-{transform}-transform')
    ckpt_root_path = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness'
    split = 'train'

    # if model_id != 0:
    #     continue
    for task in tasks:
        method_name_saveable = f"{method}_{ablation}_{level}_{split}_{metric_name}"
        circuit_path = os.path.join(circuit_dir, method_name_saveable, f"{task.replace('_', '-')}_sweep_{model_name}")
        # if os.path.exists(circuit_path):
        #     continue
        # else:
        #     os.makedirs(circuit_path, exist_ok=True)

        model = get_model_from_name(model_name, device=device)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        # if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
        #     continue
        graph = Graph.from_model(model)
        # hf_task_name = f'vmib-bench/{TASKS_TO_HF_NAMES[task]}'
        # dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=split, task=task, model_name=model_name, num_examples=num_examples)
        dataset, intervention_dataset = setup_dataset(task, split=split, model_name=model_name, num_examples=1000000000, fragment=fragment, device=device)
        # if head is not None:
        #     head = head
        #     if len(dataset) < head:
        #         print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
        #         head = len(dataset)
        #     dataset.head(head)
        dataloader = dataset.to_dataloader(batch_size=batch_size)
        intervention_dataloader = intervention_dataset.to_dataloader(batch_size=batch_size)
        metric = get_metric(metric_name, task, model, model)
        attribution_metric = partial(metric, mean=True, loss=True)
        if level == 'edge':
            perexample_scores = attribute(model, graph, dataloader, attribution_metric, method, ablation, get_perexample_scores=get_perexample_scores, intervention_dataloader=intervention_dataloader,
                      ig_steps=ig_steps, optimal_ablation_path=optimal_ablation_path, device=device, task=task, model_id=0, profile_one_batch=profile_one_batch)
        else:
            attribute_node(model, graph, dataloader, attribution_metric, method,
                           ablation, neuron=level == 'neuron', ig_steps=ig_steps, intervention_dataloader=intervention_dataloader, device=device,
                           optimal_ablation_path=optimal_ablation_path, task=task, model_id=model_id, profile_one_batch=profile_one_batch)

        # Save the graph
        fragment_number = '_' + str(fragment) if fragment is not None else ''
        graph.to_pt(f'{circuit_path}/importances{fragment_number}.pt')
        if perexample_scores:
            with open(f'{circuit_path}/perexample_importances{fragment_number}.p', 'wb') as file:
                pickle.dump(perexample_scores, file)
