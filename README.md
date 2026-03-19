# Predicting Deep Zero-Shot CNNs from Text — Unofficial Reproduction

> **NOTE:** This repository is implemented from scratch as a course project for CIML (Computational Intelligence and Machine Learning) since there is no official or public implementation of the paper available.

## Paper Details

- **Title:** Predicting deep zero-shot convolutional neural networks using textual descriptions [[PDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Ba_Predicting_Deep_Zero-Shot_ICCV_2015_paper.pdf)]
- **Authors:** Jimmy Lei Ba, Kevin Swersky, Sanja Fidler, Ruslan Salakhutdinov (University of Toronto)
- **Venue:** ICCV 2015
- **Summary:** Uses textual descriptions (e.g. Wikipedia articles) to predict the weights of both convolutional and fully connected layers in a deep CNN for zero-shot visual classification, without hand-defined semantic attributes. The key idea is: given a text description $t_c$ for class $c$, a neural network $f_t(t_c)$ predicts classifier weights $w_c$, and the classification score is $\hat{y}_c = w_c^\top g_v(x)$ where $g_v$ maps images to a joint embedding space (Sec 3.2). This extends to predicting convolutional filters (Sec 3.3) and a joint fc+conv model (Sec 3.4).

<table width="100%"><tr><td align="left"><a href="https://huggingface.co/LiXiuyin/zero-shot-cnn-comp7404-group17/tree/main"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface" alt="Hugging Face"></a></td><td align="center"><a href="https://github.com/LiXiuyin/zero-shot-cnn-comp7404-group17"><img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="GitHub"></a></td><td align="right"><a href="https://drive.google.com/file/d/1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz/view"><img src="https://img.shields.io/badge/Google%20Drive-Images-red?logo=google-drive" alt="Google Drive"></a></td></tr></table>

## Setup

### Option 1: Using uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync && source .venv/bin/activate                # install deps + activate
```

### Option 2: pip + venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Conda

```bash
conda create -n ciml python=3.14 && conda activate ciml
pip install -r requirements.txt
```

> **Note:** Requires Python >= 3.10 (see `pyproject.toml`). Keep the version consistent with `.python-version`.

## Quick Start

The entire workflow is driven by **three scripts**. Each script automatically sets up the environment and downloads datasets if needed.

### 1. Train all paper models — `train.sh`

```bash
bash train.sh          # 12 h on one RTX 5090
```

This will:
1. Sync the `uv` environment and activate it
2. Download the datasets if not already present
3. Train all models needed to reproduce the paper results:
   - **Table 1:** 5-fold cross-validation for fc, conv, fc+conv models with BCE loss on both datasets; additionally trains hinge and euclidean variants for conv and fc+conv (single run) for the "best among loss functions" comparison
   - **Table 2:** FC model with Hinge and Euclidean loss on CUB-200-2011 (BCE reuses Table 1 checkpoints)
   - **Table 3:** FC+Conv models with conv4_3 and pool5 layers on CUB-200-2011 (conv5_3 reuses Table 1)
   - **Table 4:** FC and FC+Conv models with BCE loss on the full dataset (no unseen classes) with a 50/50 split for both CUB-200-2011 and Oxford Flowers-102
   - **Figure 2:** Word sensitivities of unseen classes — uses the FC model trained above
4. Upload checkpoints to HuggingFace for easy access

> **Notes:**
> - All checkpoints are saved in `checkpoints/fold{i}/` for cross-validation runs and `checkpoints/` for single runs. Logs are saved in `logs/`.
> - For Tables 2, 3, and Figure 2, we report results on CUB-200-2011 (the paper uses CUB-200-2010, which is not publicly available).
> - For Tables 1 and 4, we report results on CUB-200-2011 and Oxford Flowers-102.

### 2. Reproduce results — `reproduce.sh`

```bash
bash reproduce.sh      # downloads checkpoints from HuggingFace if not trained locally
```

This will:
1. Install LaTeX if not already present
2. Download pre-trained checkpoints from HuggingFace (if you haven't trained locally)
3. Sync the `uv` environment and download datasets if needed
4. Run evaluation scripts (`table1.py` – `table4.py`, `figure2.py`) with the correct checkpoints and datasets
5. Compile all tables to LaTeX and save to `results/`

| Output | Location |
|--------|----------|
| CSV tables | `results/tables/Table*.csv` |
| LaTeX tables | `results/tex/Table*.tex` |
| Compiled PDF | `results/AllTables.pdf` |
| Figures | `results/figures/Figure*.png` |

### 3. Run innovation experiments — `innovate.sh`

```bash
bash innovate.sh       # extensions beyond the paper
```

This will:
1. Sync the `uv` environment and download datasets if needed
2. Train 8 innovation experiments on the fc+conv model (CUB-200-2011):

| Section | Variable | Experiments |
|---------|----------|-------------|
| **A. Loss** | Auxiliary loss on top of BCE | CLIP contrastive, Center alignment, Embedding MSE |
| **B. Text encoder** | Replace TF-IDF | SBERT (384-d), SBERT-multi (384-d), CLIP text (512-d) |
| **C. Image backbone** | Replace VGG-19 | DenseNet-121, ResNet-50 |

3. Evaluate each checkpoint and generate the innovation summary table
4. Upload innovation checkpoints to HuggingFace

All innovation checkpoints are saved under `checkpoints/innov/`.

> For training arguments, checkpoint naming, manual per-table commands, and other details, see **[`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)**.

---

## Reproduced Results

> **Note:** Our results differ from the paper because: (1) Tables 2, 3, Fig 2 use CUB-200-2011 instead of CUB-200-2010 (no longer available; ~2× more images); (2) Tables 2–4 use single runs instead of 5-fold CV; (3) conv/fc+conv use lr=5e-4 (paper uses 1e-4). See [Known Deviations](docs/REPRODUCTION_GUIDE.md#known-deviations-from-paper) for details.

### Table 1 — ROC-AUC and PR-AUC comparison (Paper Sec. 5.4)

> Paper: "For both ROC-AUC and PR-AUC, we report the best numbers obtained among the models trained on different objective functions." 

| Dataset | Model | ROC-AUC unseen (Paper / Ours) | ROC-AUC seen | ROC-AUC mean | PR-AUC unseen | PR-AUC seen | PR-AUC mean |
|---|---|---|---|---|---|---|---|
| CUB-200-2011 | fc | 0.82 / 0.797 | 0.974 / 0.978 | 0.943 / 0.942 | 0.11 / 0.240 | 0.33 / 0.467 | 0.286 / 0.422 |
| CUB-200-2011 | conv | 0.80 / 0.710 | 0.96 / 0.928 | 0.925 / 0.885 | 0.085 / 0.085 | 0.15 / 0.153 | 0.14 / 0.139 |
| CUB-200-2011 | fc+conv | 0.85 / 0.819 | 0.98 / 0.964 | 0.953 / 0.935 | 0.13 / 0.228 | 0.37 / 0.335 | 0.31 / 0.314 |
| Oxford Flower | fc | 0.70 / 0.540 | 0.987 / 0.928 | 0.90 / 0.852 | 0.07 / 0.082 | 0.65 / 0.433 | 0.52 / 0.364 |
| Oxford Flower | conv | 0.65 / 0.550 | 0.97 / 0.852 | 0.85 / 0.792 | 0.054 / 0.085 | 0.61 / 0.173 | 0.46 / 0.156 |
| Oxford Flower | fc+conv | 0.71 / 0.535 | 0.989 / 0.869 | 0.93 / 0.803 | 0.067 / 0.085 | 0.69 / 0.419 | 0.56 / 0.353 |

### Table 2 — Loss function comparison, fc model (Paper Sec. 5.5)

> Paper reports on CUB-200-2010; ours on CUB-200-2011.

| Metric | BCE (Paper / Ours) | Hinge (Paper / Ours) | Euclidean (Paper / Ours) |
|---|---|---|---|
| unseen ROC-AUC | 0.82 / 0.697 | 0.795 / 0.641 | 0.70 / 0.797 |
| seen ROC-AUC | 0.973 / 0.972 | 0.97 / 0.899 | 0.95 / 0.978 |
| mean ROC-AUC | 0.937 / 0.917 | 0.934 / 0.847 | 0.90 / 0.942 |
| unseen PR-AUC | 0.103 / 0.074 | 0.10 / 0.061 | 0.076 / 0.240 |
| seen PR-AUC | 0.33 / 0.411 | 0.41 / 0.269 | 0.37 / 0.467 |
| mean PR-AUC | 0.287 / 0.344 | 0.35 / 0.228 | 0.31 / 0.422 |
| unseen class acc. | 0.01 / 0.141 | 0.006 / 0.082 | 0.12 / 0.271 |
| seen class acc. | 0.35 / 0.437 | 0.43 / 0.484 | 0.263 / 0.446 |
| mean class acc. | 0.17 / 0.378 | 0.205 / 0.403 | 0.19 / 0.411 |
| unseen top-5 acc. | 0.176 / 0.399 | 0.182 / 0.294 | 0.428 / 0.556 |
| seen top-5 acc. | 0.58 / 0.767 | 0.668 / 0.827 | 0.45 / 0.802 |
| mean top-5 acc. | 0.38 / 0.693 | 0.41 / 0.721 | 0.44 / 0.753 |

### Table 3 — Conv feature layer ablation, fc+conv (Paper Sec. 5.6)

> Paper reports on CUB-200-2010; ours on CUB-200-2011.

| Metric | Conv5_3 (Paper / Ours) | Conv4_3 (Paper / Ours) | Pool5 (Paper / Ours) |
|---|---|---|---|
| mean ROC-AUC | 0.91 / 0.919 | 0.6 / 0.918 | 0.82 / 0.927 |
| mean PR-AUC | 0.28 / 0.404 | 0.09 / 0.433 | 0.173 / 0.457 |
| mean top-5 acc. | 0.25 / 0.741 | 0.153 / 0.766 | 0.02 / 0.770 |

### Table 4 — Full-dataset supervised baseline, 50/50 split, top-1 acc. (Paper Sec. 5.7)

> Paper also includes CUB-2010 (fc: 0.60, fc+conv: 0.62), which we cannot reproduce.

| Model | CUB-2011 (Paper / Ours) | Oxford Flowers (Paper / Ours) |
|---|---|---|
| fc | 0.64 / 0.483 | 0.73 / 0.722 |
| fc+conv | 0.66 / 0.503 | 0.77 / 0.771 |

---

## Our Improvements (beyond the paper)

### A. Loss Improvements

Auxiliary loss terms on top of BCE: $L = L_{cls} + \lambda L_{aux}$

1. **CLIP Contrastive Loss** — encourages matched image-text pairs to be close and mismatched pairs to be far apart:

$$L_{CLIP} = -\log \frac{\exp(\text{sim}(z_i, t_i) / \tau)}{\sum_j \exp(\text{sim}(z_i, t_j) / \tau)}$$

2. **Center Alignment Loss** — aligns the global centers of visual and textual embeddings:

$$L_{center} = \| \text{mean}(f) - \text{mean}(g) \|^2_2$$

3. **Embedding MSE Loss** — directly minimizes MSE between image and text embeddings: $L_{emb} = \text{MSE}(g, f)$

**Ablation of loss functions on CUB-200-2011**

| Setting | Seen PR-AUC | Seen ROC-AUC | Unseen PR-AUC | Unseen ROC-AUC |
|---|---:|---:|---:|---:|
| BCE | 0.561 | 0.983 | 0.070 | 0.689 |
| CLIP-loss | **0.578** | **0.985** | **0.077** | **0.702** |
| CenterAln | 0.554 | 0.980 | 0.066 | 0.692 |
| EmbMSE | 0.168 | 0.931 | 0.074 | 0.695 |

### B. Stronger Text Encoding

Replace TF-IDF with modern text encoders:
- **SBERT** (all-MiniLM-L6-v2, 384-d)
- **SBERT-multi** (sentence-level pooling, 384-d)
- **CLIP Text Encoder** (ViT-B/32, 512-d)

**Ablation of text encoders (CUB-200-2011)**

| Text encoder | Seen PR-AUC | Seen ROC-AUC | Unseen PR-AUC | Unseen ROC-AUC |
|---|---:|---:|---:|---:|
| TF-IDF(Baseline) | 0.335 | 0.964 | 0.228 | 0.819 |
| SBERT | **0.510** | 0.980 | 0.144 | 0.841 |
| SBERT-multi | 0.487 | 0.979 | 0.143 | 0.804 |
| CLIP-text | 0.500 | **0.982** | **0.284** | **0.864** |

### C. Stronger Image Encoding

Replace VGG-19 with modern CNN backbones:
- **DenseNet-121** (conv branch: denseblock3, 1024×14×14)
- **ResNet-50** (conv branch: layer3, 1024×14×14)

**Ablation of image backbones (CUB-200-2011)**

| Backbone | Seen PR-AUC | Seen ROC-AUC | Unseen PR-AUC | Unseen ROC-AUC |
|---|---:|---:|---:|---:|
| VGG-19(Baseline) | 0.335 | 0.964 | **0.228** | **0.819** |
| DNet121 | **0.535** | **0.983** | 0.090 | 0.740 |
| ResNet50 | 0.416 | 0.970 | 0.066 | 0.682 |

---

## Directory Structure

```
├── train.sh             # Train all paper models (≈12 h on one RTX 5090)
├── reproduce.sh         # Generate all tables and figures from checkpoints
├── innovate.sh          # Run innovation experiments (loss, text, backbone ablation)
├── main.py              # Entry point: train / eval / sanity check
├── pyproject.toml       # Project metadata and dependencies (for uv/pip)
├── requirements.txt     # requirements-style export (generated via `uv export`)
├── data/
│   ├── dataset.py       # ZeroShotDataset with train/test_seen/test_unseen splits
│   ├── download_dataset.py  # Download images from Google Drive
│   ├── preparation.py   # Data split helpers: prepare_birds/flowers_zero_shot, prepare_*_50_50
│   ├── image_preprocessor.py # Image transforms (center crop 224×224)
│   ├── sampler.py       # ClassAwareSampler for balanced batches
│   ├── text_processor.py  # TF-IDF text features (9763-d)
│   ├── text_sbert.py    # SBERT text encoder (384-d)
│   ├── text_sbert_multi.py  # SBERT multi-granularity encoder (384-d)
│   ├── text_clip.py     # CLIP text encoder (512-d)
│   └── wikipedia/       # Wikipedia JSONL texts (birds.jsonl, flowers.jsonl)
├── models/
│   ├── zero_shot_model.py   # ZeroShotModel: fc / conv / fc+conv (Sec 3.2–3.4)
│   ├── text_encoder.py      # Text encoder ft(·): p → 300 → k
│   ├── image_encoder.py     # Image encoder gv(·): VGG19/DenseNet/ResNet → k
│   └── weight_predictor.py  # ConvWeightPredictor: 300 → K'×3×3 filters
├── scripts/
│   ├── train.py         # Main training loop with early stopping + CV
│   ├── evaluate.py      # ROC-AUC, PR-AUC, Top-1/Top-5 evaluation
│   └── reproduce/       # Table/figure generation scripts
│       ├── table1.py – table4.py   # Paper table reproduction
│       ├── table_innov.py          # Innovation summary table
│       ├── figure2.py              # Word sensitivity + NN retrieval
│       ├── compile_all_tables.py   # LaTeX compilation
│       ├── common.py               # Shared helpers (checkpoint auto-detection, output dirs)
│       ├── eval_utils.py           # Evaluation utilities (inference, metrics, CV fold loading)
│       └── README.md               # Checkpoint naming, table mapping, quick-start guide
├── utils/
│   ├── config.py        # Default hyperparameters (paper + extensions)
│   ├── losses.py        # BCE, Hinge, Euclidean, CLIP, alignment losses
│   ├── seed_utils.py    # Seed management for reproducibility
│   └── filename_utils.py  # Auto-generate checkpoint/log filenames
├── results/             # Generated CSV tables, LaTeX files, figures
├── checkpoints/         # Model checkpoints (auto-generated, gitignored)
├── logs/                # Training CSV logs (auto-generated)
```

## Implementation Details (paper faithful)

This implementation strictly follows Ba et al. ICCV 2015:

| Component | Paper specification (Section) | Implementation |
|-----------|-------------------------------|----------------|
| **VGG-19** | ImageNet pretrained, frozen (Sec 5.1) | torchvision VGG19, no fine-tuning |
| **Image preprocessing** | Shortest side → 224px, center crop 224×224 (Sec 5.1) | torchvision transforms |
| **gv branch** | 4096 → 300 → k, k=50 (Sec 3.2, 5.1) | Linear(4096, 300) → ReLU → Linear(300, 50) |
| **ft branch** | p → 300 → k, p=9763 (Sec 3.2, 5.1) | Linear(9763, 300) → ReLU → Linear(300, 50) |
| **Conv branch** | K'=5 filters, 3×3, from conv5_3 (Sec 3.3, 5.1) | Conv2d(512→5, 3×3) + predicted K'×3×3 filters |
| **Joint model** | score = fc_score + conv_score (Eq. 5, Sec 3.4) | `_forward_fc() + _forward_conv()` |
| **Loss: BCE** | Eq. 6: sum over I_ij log σ(ŷ) (Sec 4.1) | `F.binary_cross_entropy_with_logits(reduction="sum")` |
| **Loss: Hinge** | Eq. 7: sum max(0, ε − I_ij ŷ) (Sec 4.2) | `F.relu(margin - targets * scores).sum()` |
| **Loss: Euclidean** | MSE in (g, f) space (Sec 4.2.1) | `F.mse_loss(scores, targets, reduction="sum")` |
| **Minibatch** | Sum only over classes in batch, O(B×U) (Sec 4.1) | `torch.unique` for dynamic class selection |
| **Optimizer** | Adam, lr=1e-4 (Sec 5.1) | Adam(lr=1e-4) fc; Adam(lr=5e-4) conv/fc+conv* |
| **TF-IDF** | 9763-d, log normalization (Sec 5.2) | `sublinear_tf=True`, `max_features=9763` |
| **Data split** | 40/160 (CUB), 20/82 (Flowers), 80/20 within seen (Sec 5.2, 5.3) | Configurable via `--n_unseen`, `--train_ratio` |
| **Evaluation** | 5-fold cross-validation (Sec 5.2) | `--n_folds 5` (default) |

*lr=5e-4 for conv/fc+conv is an empirical adjustment for better convergence; the paper uses lr=1e-4 for all models.

> For the complete list of training arguments, checkpoint naming convention, manual per-table commands, and known deviations from the paper, see **[`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)**.

## Citation

```bibtex
@inproceedings{ba2015predicting,
  title={Predicting deep zero-shot convolutional neural networks using textual descriptions},
  author={Ba, Jimmy Lei and Swersky, Kevin and Fidler, Sanja and Salakhutdinov, Ruslan},
  booktitle={ICCV},
  year={2015}
}
```

## Disclaimer

This is an unofficial, educational reproduction and is not affiliated with the paper authors or their institutions.
