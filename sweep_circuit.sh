#!/usr/bin/env bash
# Hyperparameter sweep for finetune_with_circuit_metric.py
# Sweeps: lr, weight_decay, lambda_circuit, circuit_loss_type,
#         use_auto_aug, use_ce_only, and (optionally) SAM circuit loss params.
# Results written to RESULTS_CSV with best OOD acc per run.

set -euo pipefail

# Fix CXXABI version mismatch: use conda env's libstdc++ instead of the system one
export LD_LIBRARY_PATH="/usa/yxpengcs/miniconda3/envs/vMIB/lib:${LD_LIBRARY_PATH:-}"

# ── fixed args ──────────────────────────────────────────────────────────────
TASK="PACS-mean-art_painting"
MODEL="ViT-B_16-in21k"
DEVICE="cuda:2"
EPOCHS=40
BATCH_SIZE=32
LAMBDA_CE=1.0
SAVE_DIR="checkpoints_circuit"
GRAD_CLIP=0.0
USE_SURROGATE=""          # set to "--use_surrogate" to enable
USE_ARCHIVED=""
USE_SAM_SURROGATE=""   # set to "" to disable SAM and resume original sweep
USE_ACTDIFF_REG="--use_actdiff_reg"                        # set to "--use_actdiff_reg" to enable; mutually exclusive with SAM
SHALLOW_THRESH=3

# ── actdiff_reg metric variants (only used when USE_ACTDIFF_REG is set) ──────
_actdiff_metrics=()
if [[ -n "$USE_ACTDIFF_REG" ]]; then
    _actdiff_metrics=("norm" "cosine")
else
    _actdiff_metrics=("norm")   # one no-op entry; value never reaches Python
fi
DEEP_THRESH=9

# ── circuit variant sweep (only used when neither SAM nor actdiff_reg active) ─
# accumulated: accumulates g_ref over circuit_accum_steps batches (dataset-level circuit estimate)
CIRCUIT_VARIANTS=("eap_ig_detach_g" "eap_ig_detach_actdiff" "target_logit" "accumulated")
CIRCUIT_ACCUM_STEPS_VALUES=(4 8)  # only used for 'accumulated'; other variants always use 1

# ── sweep grids (original) ───────────────────────────────────────────────────
LRS=(1e-4)
WEIGHT_DECAYS=(5e-7)
#LAMBDA_CIRCUITS=(0.0)
LAMBDA_CIRCUITS=(0.1 0.2 0.05)
#CIRCUIT_LOSS_TYPES=(ratio log)
CIRCUIT_LOSS_TYPES=("log")
USE_AUTO_AUG=("")
USE_CE_ONLY=("")

# ── SAM sweep grids — each mode sweeps only its own params ───────────────────
SAM_MODES=("activation")

# surrogate / dropout_ref / ema_ref: sweep rho only
SIMPLE_GRAD_MODES=("surrogate" "ema_ref" "task_loss" "kl")
SIMPLE_RHO_VALUES=(0.05)

# finite_diff: fixed rho, sweep fd_K × fd_alpha
FD_RHO=0.05
FD_K_VALUES=(5 10)
FD_ALPHA_VALUES=(0.01)

# random_init: fixed rho, sweep pgd_sigma
RI_RHO=0.05
PGD_SIGMA_VALUES=(0.01 0.05)

# Combo format: "sam_mode:grad_mode:rho:fd_k:fd_alpha:pgd_sigma"
# Unused fields carry placeholder "_".
_sam_combos=()
if [[ -n "$USE_SAM_SURROGATE" ]]; then
    for _sm in "${SAM_MODES[@]}"; do
        # ── simple modes: sweep rho ──────────────────────────────────────────
        for _gm in "${SIMPLE_GRAD_MODES[@]}"; do
        for _rho in "${SIMPLE_RHO_VALUES[@]}"; do
            _sam_combos+=("${_sm}:${_gm}:${_rho}:_:_:_")
        done; done

        # ── finite_diff: fixed rho, sweep fd_K × fd_alpha ────────────────────
        for _fdk in "${FD_K_VALUES[@]}"; do
        for _fda in "${FD_ALPHA_VALUES[@]}"; do
            _sam_combos+=("${_sm}:finite_diff:${FD_RHO}:${_fdk}:${_fda}:_")
        done; done

        # ── random_init: fixed rho, sweep pgd_sigma ───────────────────────────
        for _sigma in "${PGD_SIGMA_VALUES[@]}"; do
            _sam_combos+=("${_sm}:random_init:${RI_RHO}:_:_:${_sigma}")
        done
    done
else
    _sam_combos=("N/A:N/A:N/A:_:_:_")   # one no-op; values never reach Python
fi

# ── circuit variant combos (only active when SAM and actdiff_reg are both off) ─
_circuit_combos=()
if [[ -z "$USE_SAM_SURROGATE" && -z "$USE_ACTDIFF_REG" ]]; then
    for _cv in "${CIRCUIT_VARIANTS[@]}"; do
        if [[ "$_cv" == "accumulated" ]]; then
            for _cas in "${CIRCUIT_ACCUM_STEPS_VALUES[@]}"; do
                _circuit_combos+=("${_cv}:${_cas}")
            done
        else
            _circuit_combos+=("${_cv}:1")   # accum_steps=1 means recompute every batch
        fi
    done
else
    _circuit_combos=("N/A:N/A")   # no-op entry; values never reach Python
fi

# ── output CSV ──────────────────────────────────────────────────────────────
RESULTS_CSV="sweep_results_${TASK}.csv"
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "lr,weight_decay,lambda_circuit,circuit_loss_type,use_auto_aug,use_ce_only,use_archived,use_surrogate,sam_mode,sam_grad_mode,rho,best_ood_acc,run_id,run_name,fd_k,fd_alpha,pgd_sigma,actdiff_metric,circuit_variant,circuit_accum_steps" > "$RESULTS_CSV"
fi

# ── sweep ───────────────────────────────────────────────────────────────────
for lr in "${LRS[@]}"; do
for wd in "${WEIGHT_DECAYS[@]}"; do
for lcirc in "${LAMBDA_CIRCUITS[@]}"; do
for closs in "${CIRCUIT_LOSS_TYPES[@]}"; do
for use_auto_aug in "${USE_AUTO_AUG[@]}"; do
for use_ce_only in "${USE_CE_ONLY[@]}"; do
for actdiff_metric in "${_actdiff_metrics[@]}"; do
for _sam_combo in "${_sam_combos[@]}"; do
for _circuit_combo in "${_circuit_combos[@]}"; do
    IFS=':' read -r sam_mode sam_grad_mode rho fd_k fd_alpha pgd_sigma <<< "$_sam_combo"
    IFS=':' read -r circuit_variant circuit_accum_steps <<< "$_circuit_combo"

    auto_aug_label=$([ -n "$use_auto_aug" ]   && echo "1" || echo "0")
    ce_only_label=$([ -n "$use_ce_only" ]     && echo "1" || echo "0")
    archived_label=$([ -n "$USE_ARCHIVED" ]   && echo "1" || echo "0")
    surrogate_label=$([ -n "$USE_SURROGATE" ] && echo "1" || echo "0")

    echo "======================================================"
    echo "lr=$lr  wd=$wd  lcirc=$lcirc  closs=$closs  auto_aug=$auto_aug_label  ce_only=$ce_only_label"
    if [[ -n "$USE_SAM_SURROGATE" ]]; then
        echo "sam_mode=$sam_mode  rho=$rho  sam_grad=$sam_grad_mode  fd_k=$fd_k  fd_alpha=$fd_alpha  pgd_sigma=$pgd_sigma"
    fi
    if [[ "$circuit_variant" != "N/A" ]]; then
        echo "circuit_variant=$circuit_variant  circuit_accum_steps=$circuit_accum_steps"
    fi
    echo "======================================================"

    log_file=$(mktemp /tmp/sweep_XXXXXX.log)

    # Build circuit-loss args conditionally (SAM and actdiff_reg are mutually exclusive)
    sam_args=""
    if [[ -n "$USE_SAM_SURROGATE" ]]; then
        sam_args="--use_sam_surrogate --sam_mode $sam_mode --rho_s $rho --rho_d $rho --sam_grad_mode $sam_grad_mode"
        if [[ "$sam_grad_mode" == "finite_diff" ]]; then
            sam_args="$sam_args --fd_K $fd_k --fd_alpha $fd_alpha"
        elif [[ "$sam_grad_mode" == "random_init" ]]; then
            sam_args="$sam_args --pgd_sigma $pgd_sigma"
        fi
    elif [[ -n "$USE_ACTDIFF_REG" ]]; then
        sam_args="--use_actdiff_reg --actdiff_metric $actdiff_metric"
    fi

    circuit_args=""
    if [[ "$circuit_variant" != "N/A" ]]; then
        circuit_args="--circuit_variant $circuit_variant --circuit_accum_steps $circuit_accum_steps"
    fi

    python finetune_with_circuit_metric.py \
        --task         "$TASK"          \
        --model_name   "$MODEL"         \
        --device       "$DEVICE"        \
        --epochs       "$EPOCHS"        \
        --batch_size   "$BATCH_SIZE"    \
        --lr           "$lr"            \
        --weight_decay "$wd"            \
        --lambda_ce    "$LAMBDA_CE"     \
        --lambda_circuit "$lcirc"       \
        --circuit_loss_type "$closs"    \
        --shallow_thresh "$SHALLOW_THRESH" \
        --deep_thresh    "$DEEP_THRESH"    \
        --save_dir     "$SAVE_DIR"      \
        --grad_clip    "$GRAD_CLIP"     \
        $USE_SURROGATE \
        $USE_ARCHIVED \
        $use_auto_aug \
        $use_ce_only \
        $sam_args \
        $circuit_args \
        2>&1 | tee "$log_file" || true  # allow NaN abort (exit 1) without stopping sweep

    # Parse run name, run id, and best OOD acc — all printed before sys.exit(1) on NaN abort too.
    best_ood=$(grep "BEST_OOD_ACC:"    "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    run_name=$(grep "WANDB_RUN_NAME:"  "$log_file" | tail -1 | sed 's/.*WANDB_RUN_NAME: //'  || echo "N/A")
    run_id=$(  grep "WANDB_RUN_ID:"    "$log_file" | tail -1 | sed 's/.*WANDB_RUN_ID: //'    || echo "N/A")

    echo "$lr,$wd,$lcirc,$closs,$auto_aug_label,$ce_only_label,$archived_label,$surrogate_label,$sam_mode,$sam_grad_mode,$rho,$best_ood,$run_id,$run_name,$fd_k,$fd_alpha,$pgd_sigma,$actdiff_metric,$circuit_variant,$circuit_accum_steps" >> "$RESULTS_CSV"
    rm -f "$log_file"

done
done
done
done
done
done
done
done
done

echo ""
echo "Sweep complete. Results saved to $RESULTS_CSV"
