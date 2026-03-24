# Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings

> Yunxiang Peng, Mengmeng Ma, Ziyu Yao, Xi Peng
> University of Delaware · George Mason University
> CVPR 2026

---

## Overview

Standard generalization proxies (model confidence, accuracy-on-the-line) assess only model **outputs**, leading to overconfidence and underspecification. We instead use a model's **internal circuit** — the causal interactions between its transformer components — as a label-free proxy for generalization.

We address two practical scenarios:

| Scenario | Metric | Key Idea |
|---|---|---|
| **Before deployment** — model selection | Dependency Depth Bias (DDB) | Models with stronger deep-layer pathways generalize better |
| **After deployment** — performance monitoring | Circuit Shift Score (CSS) | Circuit rewiring relative to ID baseline tracks performance drop |

Across PACS, Camelyon17, Terra Incognita, FMoW, and ImageNet, DDB and CSS improve correlation with OOD performance by **+13.4%** and **+34.1%** over existing proxies, and enable silent failure detection with **~45% gain** in alarm F1.

---

## Method

**Circuit Definition.** We define a circuit as a continuous edge-weight mapping over the ViT computation graph:

$$c^{\mathcal{M}}_{\mathcal{D}}(e) := \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathrm{KL}\!\left(\mathcal{M}_{\setminus\{e\}}(x),\, \mathcal{M}(x)\right) \right]$$

Edge weights are computed via **EAP-IG** with mean-positional ablation.

**Dependency Depth Bias (DDB).** Measures a model's relative dependency on deep vs. shallow source layers via the inter-layer dependency matrix Λ:

$$\mathrm{DDB}(\Lambda, \tau) = \log \frac{\sum_{i \in \mathcal{L}_\text{high}, j \in J} \Lambda_{ij}}{\sum_{i \in \mathcal{L}_\text{low}, j \in J} \Lambda_{ij}}$$

Three variants: DDB_global (all target layers), DDB_deep (deep targets), DDB_out (output node only). DDB_out achieves the best overall correlation (0.766 average R²/SRCC/KRCC).

**Circuit Shift Score (CSS).** Measures deviation of the OOD circuit from the ID baseline. Best variant CSS_(v,SRCC) uses Spearman rank correlation between edge weight vectors, achieving 0.811 average correlation — exceeding the strongest baseline by 0.341.

---

## Setup

```bash
# Install EAP-IG (circuit discovery backbone)
git clone -b MIB https://github.com/hannamw/EAP-IG/
cd EAP-IG && pip install . && cd ..

pip install torch timm transformers accelerate wandb
```

---

## Usage

**Circuit Discovery (DDB / CSS computation):**
```bash
python run_attribution.py \
    --models ViT-B_16-in21k \
    --tasks PACS-mean-art_painting \
    --method EAP-IG-inputs \
    --level edge \
    --ablation mean-positional
```

**Compute DDB / CSS metrics:**
```bash
python compute_metrics.py
```

**Circuit-regularized fine-tuning** (future work direction from the paper — directly optimizing DDB during training):
```bash
python finetune_with_circuit_metric.py \
    --task PACS-mean-art_painting \
    --model_name ViT-B_16-in21k \
    --device cuda:0 \
    --epochs 40 \
    --lr 1e-4 \
    --lambda_circuit 0.1 \
    --circuit_variant eap_ig_detach_g
```

Multi-GPU sweep for circuit regularization:
```bash
# Edit GPU_IDS and NUM_GPUS in sweep_circuit_large.sh
bash sweep_circuit_large.sh
```

---

## Results

**Pre-deployment model selection** (averaged R², SRCC, KRCC across PACS, Camelyon17, Terra Incognita):

| Method | Average |
|---|---|
| ID Accuracy | 0.632 |
| Average Confidence | 0.585 |
| ATC | 0.520 |
| **DDB_out (Ours)** | **0.766** |

**Post-deployment monitoring** (averaged across PACS, FMoW, Camelyon17, ImageNet):

| Method | Average |
|---|---|
| ATC | 0.470 |
| Average Confidence | 0.391 |
| **CSS_(v,SRCC) (Ours)** | **0.811** |

---

## Citation

```bibtex
@article{peng2026insideout,
  title={Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings},
  author={Peng, Yunxiang and Ma, Mengmeng and Yao, Ziyu and Peng, Xi},
  journal={CVPR},
  year={2026}
}
```

---

## Acknowledgement

Supported by NSF CAREER 2340074, SLES 2416937, III CORE 2412675, and NIH R21CA301093.
