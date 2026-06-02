#CUDA_VISIBLE_DEVICES=3 python run_evaluation.py --models ViT-B_16_lamb-0.01-kl --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --models ViT-B_16_lamb-0.001-kl --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20

#CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --models small-ViT_lamb-0.001 --tasks colored-mnist --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --models google-ViT-B_lamb-0.01 --tasks IN-dog --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --models google-ViT-B_lamb-0.001 --tasks IN-car --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --models google-ViT-B_lamb-0.01 --tasks IN-car --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=7 python run_evaluation.py --models ViT-B_16_lamb-0.01 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=5 python run_evaluation.py --models ViT-B_16_lamb-0.001 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20

#CUDA_VISIBLE_DEVICES=2 python train_coloredobject_ViT.py

python fit_graph_metrics.py --metrics ID_acc --probit
#python fit_graph_metrics.py --metrics ID_acc circuit_stability
#python fit_graph_metrics.py --metrics ID_acc edge_entropy
python fit_graph_metrics.py --metrics ID_acc logit_contribution_scores --probit
python fit_graph_metrics.py --metrics ID_acc logit_contribution_scores_unnoramlized --probit
#python fit_graph_metrics.py --metrics ID_acc lower_edge_scores
#python fit_graph_metrics.py --metrics ID_acc total_nodes
python fit_graph_metrics.py --metrics ID_acc circuit_stability logit_contribution_scores --probit
#python fit_graph_metrics.py --metrics ID_acc circuit_stability edge_entropy logit_contribution_scores total_nodes lower_edge_scores
edge_start_ratio_deep_vs_shallow
edge_start_ratio_deep_vs_shallow_1
edge_start_ratio_deep_vs_shallow_2
edge_start_ratio_deep_vs_shallow_4
edge_start_ratio_deep_vs_shallow_5
python compute_metrics.py --metrics
middle_layer_entropy
weighted_shortcut_score
weighted_shortcut_score_normalized
early_to_deep_edge_importance
path_depth_entropy
deep_shallow_ratio
edge_start_ratio_deep_vs_shallow
shortcut_vs_deep_ratio
shortcut_vs_deep_ratio_1
shortcut_vs_deep_ratio_2
shortcut_vs_deep_ratio_4
shortcut_vs_deep_ratio_5
shortcut_vs_local_ratio
shortcut_vs_local_ratio_1
shortcut_vs_local_ratio_2
shortcut_vs_local_ratio_4
shortcut_vs_local_ratio_5
edge_start_ratio_deep_vs_shallow
edge_start_ratio_deep_vs_shallow_1
edge_start_ratio_deep_vs_shallow_2
edge_start_ratio_deep_vs_shallow_4
edge_start_ratio_deep_vs_shallow_5
logit_contribution_diff_deep_vs_shallow
logit_contribution_ratio_deep_vs_shallow
logit_contribution_ratio_deep_vs_shallow_v1
logit_contribution_ratio_deep_vs_shallow_1
logit_contribution_ratio_deep_vs_shallow_2
logit_contribution_ratio_deep_vs_shallow_4
logit_contribution_ratio_deep_vs_shallow_5
logit_contribution_ratio_deep_vs_shallow_signed
deep_logit_contribution
deep_logit_contribution_signed
deep_logit_contribution_normed
deep_logit_contribution_signed_normed
largest_edge_score
middle_layer_score_ratio
std_entropy_across_layers
logit_contribution_peak_layer
generalization_graph_metrics
norm_generalization_graph_metrics
attention_mlp_ratio
layerwise_score_variance
weighted_path_depth
layerwise_score_entropy
tail_shortcut_mass
sign_split_tail_mass
backbone_chain_mass
logits_inflow_by_src
signed_generalization_graph_metrics
normed_signed_generalization_graph_metrics
connectivity_by_layer_bands_signed
edge_norm
distance_from_id
robust_graph_similarity

#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method EAP --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method EAP-IG-activations --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method EAP-IG-inputs --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method information-flow-routes --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16_lamb-0.01 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16_lamb-0.001 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16_lamb-0.0001 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16_lamb-0.04 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20
#CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --models ViT-B_16_lamb-0.00001 --tasks waterbirds-bg-water-group --ablation patching --method UGS --batch-size 20


#CUDA_VISIBLE_DEVICES=0 python run_attribution.py --models ViT-B_16-erm --tasks colored-mnist --split train --ablation patching --method EAP --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=1 python run_attribution.py --models ViT-B_16-erm --tasks colored-mnist --split train --ablation patching --method EAP-IG-inputs --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=2 python run_attribution.py --models ViT-B_16-erm --tasks colored-mnist --split train --ablation patching --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=3 python run_attribution.py --models ViT-B_16-erm --tasks colored-mnist --split train --ablation patching --method information-flow-routes --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=4 python run_attribution.py --models ViT-B_16-erm --tasks colored-mnist --split train --ablation patching --method random --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=3 python run_evaluation.py --models ViT-B_16-erm_lamb-0.0001 --tasks colored-mnist --split test --ablation patching --method UGS --batch-size 20 --center "" --absolute "" --metric logit_diff
#CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --models ViT-B_16-erm --tasks colored-mnist --split test --ablation patching --method random --batch-size 20 --center "" --absolute "True" --metric logit_diff
#CUDA_VISIBLE_DEVICES=5 python run_evaluation.py --models ViT-B_16-erm --tasks colored-mnist --split test --ablation patching --method EAP-IG-activations --batch-size 20 --center "" --absolute "" --metric logit_diff
#CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --models ViT-B_16-erm --tasks colored-mnist --split test --ablation patching --method EAP-IG-activations --batch-size 20 --center "" --absolute "True" --metric logit_diff


#CUDA_VISIBLE_DEVICES=0 python run_attribution.py --models ViT-B_16-scratch --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=0 python run_attribution.py --models ViT-B_16-mae --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=1 python run_attribution.py --models ViT-B_16-mae-lp --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=2 python run_attribution.py --models ViT-B_16-in1k --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=3 python run_attribution.py --models ViT-B_16-in1k-lp --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=4 python run_attribution.py --models ViT-B_16-in21k-lp --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=5 python run_attribution.py --models ViT-B_16-clip-openai --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=6 python run_attribution.py --models ViT-B_16-clip-openai-lp --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=7 python run_attribution.py --models ViT-B_16-clip-laion2b --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#CUDA_VISIBLE_DEVICES=3 python run_attribution.py --models ViT-B_16-in21k --tasks waterbirds-mean-worst-group --split train --ablation mean-positional --method EAP-IG-activations --batch-size 10 --num-examples 1000000000 --metric logit_diff
#python run_evaluation.py --models ViT-B_16-clip-openai ViT-B_16-clip-openai-lp --tasks waterbirds-mean-worst-group --split test --ablation mean-positional --method EAP-IG-activations --batch-size 20 --center "" --absolute "" --metric logit_diff

#CUDA_VISIBLE_DEVICES=0 python optimalablation/edge_pruning_unif.py -l 1e-3 -e cf -n unif --minwindow 0.5 --maxwindow 2 --dataset waterbirds-bg-worst-group --batch_size 2 --model_name ViT-B_16-scratch
#CUDA_VISIBLE_DEVICES=1 python optimalablation/edge_pruning_unif.py -l 1e-3 -e cf -n unif --minwindow 0.5 --maxwindow 2 --dataset waterbirds-bg-worst-group --batch_size 2 --model_name ViT-B_16-in1k-lp
#CUDA_VISIBLE_DEVICES=2 python optimalablation/edge_runing_unif.py -l 1e-3 -e cf -n unif --minwindow 0.5 --maxwindow 2 --dataset waterbirds-bg-worst-group --batch_size 2 --model_name ViT-B_16-clip-openai-lp
#CUDA_VISIBLE_DEVICES=3 python optimalablation/edge_pruning_unif.py -l 1e-3 -e cf -n unif --minwindow 0.5 --maxwindow 2 --dataset waterbirds-bg-worst-group --batch_size 2 --model_name ViT-B_16-clip-laion2b-lp
#CUDA_VISIBLE_DEVICES=4 python optimalablation/edge_pruning_unif.py -l 1e-3 -e cf -n unif --minwindow 0.5 --maxwindow 2 --dataset waterbirds-bg-worst-group --batch_size 2 --model_name ViT-B_16-mae-lp
