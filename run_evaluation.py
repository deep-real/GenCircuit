import math 
import os
import pickle

from typing import Literal, Optional
from functools import partial
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from metrics import get_metric
from run_attribution import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME
from print_results import COL_MAPPING
from model_utils import get_model
from dataset_utils import setup_dataset
from safetensors.torch import load_file
from ep.modeling.vit import ViTHeadModel

def compute_edge_entropy(graph):
    scores = [abs(edge.score) for edge in graph.edges.values()]

    if not scores:
        return None

    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]

    edge_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)

    return edge_entropy


def evaluate_area_under_curve(model: HookedTransformer, graph: Graph, dataloader, metrics, quiet:bool=False, 
                              level:Literal['edge', 'node','neuron']='edge', log_scale:bool=False, absolute:bool=True, task=None, model_name=None,
                              intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', intervention_dataloader:DataLoader=None,
                              optimal_ablation_path:Optional[str]=None, no_normalize:Optional[bool]=False, apply_greedy:bool=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention, intervention_dataloader=intervention_dataloader, task=task, model_name=model_name).mean().item()
    
    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)
    
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        
        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count + 1 if weighted_edge_count == 0 else weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader, task=task, model_name=model_name).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)
    
    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal 
        
        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, percentages, area_under, area_from_1, average, faithfulnesses


def evaluate_area_under_curve_from_ep_json(model: HookedTransformer, graph: Graph, dataloader, metrics, quiet: bool = False,
                              level: Literal['edge', 'node', 'neuron'] = 'edge', log_scale: bool = False,
                              absolute: bool = True,
                              intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                              intervention_dataloader: DataLoader = None,
                              optimal_ablation_path: Optional[str] = None, no_normalize: Optional[bool] = False,
                              apply_greedy: bool = False, json_paths = None):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention,
                                     intervention_dataloader=intervention_dataloader).mean().item()

    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)

    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for json_path in json_paths:
        this_graph = graph
        this_graph.load_circuit_from_json(json_path)

        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count + 1 if weighted_edge_count == 0 else weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)

    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i + 1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                      (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal

        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, percentages, area_under, area_from_1, average, faithfulnesses

def compare_graphs(reference: Graph, hypothesis: Graph, by_node: bool = False):
    # Track {true, false} {positives, negatives}
    TP, FP, TN, FN = 0, 0, 0, 0
    total = 0

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges

    for obj in ref_objs.values():
        total += 1
        if obj.name not in hyp_objs:
            if obj.in_graph:
                TP += 1
            else:
                FP += 1
            continue
            
        if obj.in_graph and hyp_objs[obj.name].in_graph:
            TP += 1
        elif obj.in_graph and not hyp_objs[obj.name].in_graph:
            FN += 1
        elif not obj.in_graph and hyp_objs[obj.name].in_graph:
            FP += 1
        elif not obj.in_graph and not hyp_objs[obj.name].in_graph:
            TN += 1
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # f1 = (2 * precision * recall) / (precision + recall)
    TP_rate = recall
    FP_rate = FP / (FP + TN)

    return {"precision": precision,
            "recall": recall,
            "TP_rate": TP_rate,
            "FP_rate": FP_rate}

def area_under_roc(reference: Graph, hypothesis: Graph, by_node: bool = False):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges
    
    num_objs = len(ref_objs.values())
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        this_num_objs = pct * num_objs
        if by_node:
            raise NotImplementedError("")
        else:
            hypothesis.apply_greedy(this_num_objs)
        scores = compare_graphs(reference, hypothesis)
        tpr_list.append(scores["TP_rate"])
        fpr_list.append(scores["FP_rate"])
        precision_list.append(scores["precision"])
        recall_list.append(scores["recall"])
    
    return {"TPR": tpr_list, "FPR": fpr_list,
            "precision": precision_list, "recall": recall_list}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal_ablation_path", type=str, default=None)
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--method", type=str, default=None, help="Method used to generate the circuit (only needed to infer circuit file name)")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument("--absolute", type=bool, default=True)
    parser.add_argument("--center", type=bool, default=True)
    parser.add_argument('--apply_greedy', action='store_true', help="Enable greedy application")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    parser.add_argument("--circuit-files", type=str, nargs='+', default=None)
    parser.add_argument("--output-dir", type=str, default='results')
    args = parser.parse_args()

    i = 0
    is_ep_direct_edge = False
    for model_name in args.models:
        for task in args.tasks:
            model = get_model(model_name, task)
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True
            # if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
            #     continue
            dataset, intervention_dataset = setup_dataset(task, args, split=args.split, model_name='ViT-B_16')
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(
                        f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            intervention_dataloader = intervention_dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model, model)
            attribution_metric = partial(metric, mean=False, loss=False)

            # check if the model is load correctly
            # for clean, corrupted, label in dataloader:
            #     clean_images = torch.stack(clean).to('cuda')
            #     corrupted_images = torch.stack(corrupted).to('cuda')
            #     with torch.inference_mode():
            #         logits = model(clean_images)
            #         good_bad = torch.gather(logits, -1, torch.tensor(label).to(logits.device))
            #         corrupted_logits = model(corrupted_images)
            #         print(f'mean diff: {(logits - control_logits).mean()}')
            # ckpt = torch.load('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp_group03/waterbirds/ViT/ViT-B_16final.bin')
            # new_state_dict = {}
            # for old_key, value in ckpt.items():
            #     if "vit.encoder.layer" in old_key:
            #         new_key = old_key.replace("vit.encoder.layer", "vit.encoder")
            #         new_state_dict[new_key] = value
            #     else:
            #         new_state_dict[old_key] = value
            # control_model = ViTHeadModel.from_pretrained(
            #     'google/vit-base-patch16-224-in21k',
            #     state_dict=new_state_dict,
            #     include_qkv=True,
            #     with_embedding_nodes=True,
            #     disable_linear_regularization_term=False,
            # ).eval().to('cuda')
            # for clean, corrupted, label in dataloader:
            #     clean_images = torch.stack(clean).to('cuda')
            #     corrupted_images = torch.stack(corrupted).to('cuda')
            #     with torch.inference_mode():
            #         logits = model(clean_images)
            #         control_logits = control_model(clean_images).logits
            #         print(f'mean diff: {(logits - control_logits).mean()}')
            if is_ep_direct_edge:
                json_paths = [
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es1.1-ns0.1-alpha3.0-t896/edges.json',
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es0.982-ns0.1-alpha3.0-t896/edges.json',
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es0.952-ns0.1-alpha3.0-t896/edges.json',
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es0.902-ns0.1-alpha3.0-t896/edges.json',
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es0.802-ns0.1-alpha3.0-t896/edges.json',
                    '/home/yxpengcs/PycharmProjects/Edge-Pruning/data/runs/bg_ablate-qkv_True-IN21k-ERM-Waterbirds-group03_final-class-all-kl_loss-waterbirds-bg-water-group_ViT-B_16-wo_node_loss-w_embedding-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw600-tbs8-es0.502-ns0.1-alpha3.0-t896/edges.json'
                    ]
                graph = Graph.from_model(model)
                eval_auc_outputs = evaluate_area_under_curve_from_ep_json(model, graph, dataloader, attribution_metric,
                                                             level=args.level,
                                                             absolute=args.absolute,
                                                             intervention=args.ablation,
                                                             optimal_ablation_path=args.optimal_ablation_path,
                                                             json_paths=json_paths)
                weighted_edge_counts, percentages, area_under, area_from_1, average, faithfulnesses = eval_auc_outputs
                d = {
                    "weighted_edge_counts": weighted_edge_counts,
                    "area_under": area_under,
                    "percentages": percentages,
                    "area_from_1": area_from_1,
                    "average": average,
                    "faithfulnesses": faithfulnesses
                }
                method_name_saveable = f"EP_direct_patching_edge"
            else:
                method_name_saveable = f"{args.method}_{args.ablation}_{args.level}_train_{args.metric}"
                if args.center:
                    center_str = '_center0'
                else:
                    center_str = ''
                if args.method == 'EP' or args.method == 'UGS':
                    file_name = f'graph{center_str}.json'
                else:
                    file_name = 'importances.pt'
                p = f"{args.circuit_dir}/{method_name_saveable}/{task.replace('_', '-')}_{model_name}/{file_name}"

                if args.circuit_files is not None:
                    p = args.circuit_files[i]
                    i += 1

                print(f"Loading circuit from {p}")
                if p.endswith('.json'):
                    graph = Graph.from_json(p)
                elif p.endswith('.pt'):
                    graph = Graph.from_pt(p)
                else:
                    raise ValueError(f"Invalid file extension: {p.suffix}")

                edge_entropy = compute_edge_entropy(graph)

                eval_auc_outputs = evaluate_area_under_curve(model, graph, dataloader, attribution_metric, level=args.level, task=task, model_name=model_name,
                                                             absolute=args.absolute, intervention=args.ablation, intervention_dataloader=intervention_dataloader,
                                                             optimal_ablation_path=args.optimal_ablation_path, apply_greedy=args.apply_greedy)
                weighted_edge_counts, percentages, area_under, area_from_1, average, faithfulnesses = eval_auc_outputs

                d = {
                    "weighted_edge_counts": weighted_edge_counts,
                    "area_under": area_under,
                    "percentages": percentages,
                    "area_from_1": area_from_1,
                    "average": average,
                    "faithfulnesses": faithfulnesses,
                    'edge_entropy': edge_entropy,
                }
                is_greedy = '_greedy' if args.apply_greedy else ''
                method_name_saveable = f"{args.method}_{args.ablation}_{args.level}{is_greedy}_{args.metric}"
            output_path = os.path.join(args.output_dir, method_name_saveable)
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/{task}_{model_name}_{args.split}_abs-{args.absolute}{center_str}.pkl", 'wb') as f:
                pickle.dump(d, f)

            log_weighted_edge_counts = np.log(weighted_edge_counts)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # First plot: edge count
            axs[0].plot(log_weighted_edge_counts, faithfulnesses)
            axs[0].set_xlabel("edge count")
            axs[0].set_ylabel("faithfulness")
            axs[0].set_title("faithfulness vs edge count")
            axs[0].grid(True)

            # Second plot: edge percentage
            axs[1].plot(percentages, faithfulnesses)
            axs[1].set_xlabel("edge percentage")
            axs[1].set_ylabel("faithfulness")
            axs[1].set_title("faithfulness vs edge percentage")
            axs[1].grid(True)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(f"{output_path}/{task}_{model_name}_{args.split}_abs-{args.absolute}{center_str}.png")