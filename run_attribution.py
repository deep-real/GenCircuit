import argparse
import os
import pickle
from functools import partial

from dataset import VisionEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node

from dataset_utils import setup_dataset
from metrics import get_metric

from model_utils import get_model
from print_results import COL_MAPPING

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal_ablation_path", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['node', 'neuron', 'edge'], default='edge')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--fragment", type=int, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--circuit-dir", type=str, default='circuits-dynamic')
    args = parser.parse_args()

    for model_name in args.models:
        for task in args.tasks:
            # model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
            model = get_model(model_name, task, device=args.device)
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True
            # if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
            #     continue
            graph = Graph.from_model(model)
            # hf_task_name = f'vmib-bench/{TASKS_TO_HF_NAMES[task]}'
            # dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name, num_examples=args.num_examples)
            dataset, intervention_dataset = setup_dataset(task, args, split=args.split, model_name=model_name, num_examples=args.num_examples, fragment=args.fragment)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            intervention_dataloader = intervention_dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric(args.metric, task, model, model)
            attribution_metric = partial(metric, mean=True, loss=True)
            if args.level == 'edge':
                perexample_scores = attribute(model, graph, dataloader, attribution_metric, args.method, args.ablation, intervention_dataloader=intervention_dataloader,
                          ig_steps=args.ig_steps, optimal_ablation_path=args.optimal_ablation_path, device=args.device, task=task, model_name=model_name)
            else:
                attribute_node(model, graph, dataloader, attribution_metric, args.method, 
                               args.ablation, neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                               optimal_ablation_path=args.optimal_ablation_path)

            # Save the graph
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}_{args.split}_{args.metric}"
            circuit_path = os.path.join(args.circuit_dir, method_name_saveable, f"{task.replace('_', '-')}_{model_name}")
            os.makedirs(circuit_path, exist_ok=True)

            fragment_number = '_' + str(args.fragment) if args.fragment is not None else ''
            graph.to_pt(f'{circuit_path}/importances{fragment_number}.pt')
            if perexample_scores:
                with open(f'{circuit_path}/perexample_importances{fragment_number}.p', 'wb') as file:
                    pickle.dump(perexample_scores, file)
