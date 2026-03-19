# Reproduction Guide — Ba et al. ICCV 2015

This guide provides detailed instructions for reproducing all paper experiments and generating comparison tables. For a high-level overview and quick start, see the [README](../README.md).

## Paper Experiments Overview

| Table/Figure | Paper Section | Paper Dataset | Our Dataset | Model | Key Variable |
|---|---|---|---|---|---|
| Table 1 | Sec 5.4 | CUB-2010, CUB-2011, Flowers | CUB-2011, Flowers | fc, conv, fc+conv | model type |
| Table 2 | Sec 5.5 | **CUB-2010** | CUB-2011 | fc | loss function (BCE / Hinge / Euclidean) |
| Table 3 | Sec 5.6 | **CUB-2010** | CUB-2011 | fc+conv | conv layer (conv4\_3 / conv5\_3 / pool5) |
| Table 4 | Sec 5.7 | CUB-2010, CUB-2011, Flowers | CUB-2011, Flowers | fc, fc+conv | full-dataset 50/50 split |
| Figure 2 | Sec 5.8 | **CUB-2010** | CUB-2011 | fc | word sensitivity + NN retrieval |

> **Bold** = our dataset differs from the paper's. CUB-200-2010 (6,033 images) is no longer publicly available; we use CUB-200-2011 (11,788 images, same 200 bird classes).

**Data splits (Paper Sec 5.2, 5.3):**
- CUB: 40 unseen / 160 seen classes; seen classes use 80% train / 20% test; 5-fold cross-validation
- Flowers: 20 unseen / 82 seen classes; seen classes use 80% train / 20% test

**Paper Table 1 note:** "For both ROC-AUC and PR-AUC, we report the best numbers obtained among the models trained on different objective functions." We reproduce only the "Ours (fc/conv/fc+conv)" rows, not the DA/GPR baseline methods from Elhoseiny et al. [5] and Kulis et al. [15].

---

## Recommended: One-command Full Pipeline

```bash
# Step 1: Train all paper models (~12 h on one RTX 5090)
bash train.sh

# Step 2: Generate all tables and figures
bash reproduce.sh
```

- `train.sh` trains every required checkpoint. **Table 1 models use 5-fold CV** (saved in `checkpoints/fold{i}/`); all other models are single runs (saved in `checkpoints/` root).
- `reproduce.sh` automatically detects fold directories and averages CV folds when computing Table 1 metrics. Results are written to `results/`.

For innovation experiments (beyond the paper):

```bash
bash innovate.sh
```

Checkpoints saved under `checkpoints/innov/`.

---

## Environment Setup

```bash
# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download datasets
cd data && python download_dataset.py && cd ..
```

---

## Step 1: Training

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `fc` | `fc`, `conv`, or `fc+conv` |
| `--dataset` | `cub` | `cub` or `flowers` |
| `--epochs` | `50` | Training epochs (paper uses 200) |
| `--loss` | `bce` | `bce`, `hinge`, or `euclidean` |
| `--n_unseen` | 40 (CUB) / 20 (Flowers) | Unseen classes (0 = full dataset) |
| `--train_ratio` | `0.8` | Train split for seen classes |
| `--conv_feature_layer` | `conv5_3` | `conv4_3`, `conv5_3`, or `pool5` |
| `--batch_size` | `200` | Batch size (paper default) |
| `--lr` | `1e-4` | Learning rate for fc (conv/fc+conv always use 5e-4) |
| `--save` | auto | Custom checkpoint path |
| `--log_file` | auto | Custom log path |
| `--no_early_stopping` | — | Disable early stopping (enabled by default) |
| `--patience` | `20` | Early stopping patience |
| `--min_epochs` | `50` | Minimum epochs before early stopping |
| `--seed` | `42` | Random seed |
| `--deterministic` | — | Deterministic cuDNN (exact GPU reproducibility) |
| `--n_folds` | `5` | CV folds (5 = 5-fold into `fold{i}/`; 1 = single run at root) |
| `--text_encoder` | `tfidf` | `tfidf`, `sbert`, `sbert_multi`, or `clip` |
| `--image_backbone` | `vgg19` | `vgg19`, `densenet121`, or `resnet50` |
| `--standard_sampler` | — | Use RandomSampler (paper method); default is ClassAwareSampler |
| `--use_clip_loss` | — | Add auxiliary CLIP contrastive loss (fc/fc+conv only) |
| `--clip_weight` | `0.1` | CLIP loss weight λ |
| `--clip_temperature` | `0.07` | CLIP softmax temperature |
| `--use_center_align` | — | Add center alignment loss (fc/fc+conv only) |
| `--center_align_weight` | `0.1` | Center alignment loss weight |
| `--use_embedding_loss` | — | Add embedding MSE loss (fc/fc+conv only) |
| `--embedding_weight` | `1.0` | Embedding MSE loss weight |

### Early Stopping (enabled by default)

- **Zero-shot mode** (`n_unseen > 0`): monitors **Unseen Top-1 accuracy**
- **Full-dataset mode** (`n_unseen = 0`): monitors **Test Top-1 accuracy**
- Stops after `--patience` epochs without improvement (default: 20)
- Requires at least `--min_epochs` before stopping (default: 50)
- Saves the **best checkpoint** (not the final epoch)

### Training Output Format

Zero-shot mode (Tables 1–3):
```
Epoch   1/200 | Loss: 46.8649 | Seen:  35.2%/ 62.8% | Unseen:  12.3%/ 28.5% | ETA: 15:32
```

Full-dataset mode (Table 4):
```
Epoch   1/200 | Loss: 45.123 | Test:  12.3%/ 45.6% | ETA: 15:32
```

### Checkpoint Naming Convention

```
{model_type}_{loss}_{dataset}_{layer}_{n_unseen}[_tr{train_ratio}].pt
```

- `fc+conv` is mapped to `fc_conv` (avoids `+` in filenames)
- `_tr{ratio}` only appears when `train_ratio != 0.8`

Examples:
- `fc_bce_cub_fc_40.pt` — FC, BCE, CUB, 40 unseen
- `conv_hinge_cub_conv5_3_40.pt` — Conv, Hinge, CUB, conv5_3, 40 unseen
- `fc_conv_bce_flowers_conv5_3_20.pt` — FC+Conv, BCE, Flowers, conv5_3, 20 unseen
- `fc_bce_cub_fc_0_tr0.5.pt` — FC, BCE, CUB, full dataset, 50/50 split

### Table 1 Models — 5-fold cross-validation (Paper Sec 5.2: "5-fold cross-validation is used")

```bash
# CUB
python scripts/train.py --model_type fc      --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/fc_bce_cub_fc_40.pt

python scripts/train.py --model_type conv    --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/conv_bce_cub_conv5_3_40.pt

python scripts/train.py --model_type fc+conv --dataset cub     --epochs 200
# -> checkpoints/fold{0-4}/fc_conv_bce_cub_conv5_3_40.pt

# Flowers
python scripts/train.py --model_type fc      --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/fc_bce_flowers_fc_20.pt

python scripts/train.py --model_type conv    --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/conv_bce_flowers_conv5_3_20.pt

python scripts/train.py --model_type fc+conv --dataset flowers --epochs 200
# -> checkpoints/fold{0-4}/fc_conv_bce_flowers_conv5_3_20.pt
```

> **Note:** `train.sh` also trains hinge and euclidean variants for conv and fc+conv on both datasets (single runs) for Table 1's "best among loss functions" comparison.

### Table 2 Models — Loss function ablation (Paper Sec 5.5, single run)

BCE reuses Table 1 fold checkpoints (auto-detected). Only hinge and euclidean need explicit training:

```bash
python scripts/train.py --model_type fc --dataset cub --loss hinge     --epochs 200 --n_folds 1
# -> checkpoints/fc_hinge_cub_fc_40.pt

python scripts/train.py --model_type fc --dataset cub --loss euclidean --epochs 200 --n_folds 1
# -> checkpoints/fc_euclidean_cub_fc_40.pt
```

### Table 3 Models — Conv layer ablation (Paper Sec 5.6, single run)

conv5_3 reuses Table 1 fold checkpoints. Only conv4_3 and pool5 need explicit training:

```bash
python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer conv4_3 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_conv4_3_40.pt

python scripts/train.py --model_type fc+conv --dataset cub --conv_feature_layer pool5   --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_pool5_40.pt
```

### Table 4 Models — Full dataset supervised (Paper Sec 5.7, single run, 50/50 split)

```bash
python scripts/train.py --model_type fc      --dataset cub     --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_bce_cub_fc_0_tr0.5.pt

python scripts/train.py --model_type fc+conv --dataset cub     --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt

python scripts/train.py --model_type fc      --dataset flowers --n_unseen 0 --train_ratio 0.5 --epochs 400 --n_folds 1
# -> checkpoints/fc_bce_flowers_fc_0_tr0.5.pt

python scripts/train.py --model_type fc+conv --dataset flowers --n_unseen 0 --train_ratio 0.5 --epochs 200 --n_folds 1
# -> checkpoints/fc_conv_bce_flowers_conv5_3_0_tr0.5.pt
```

> **Note:** Flowers FC model uses 400 epochs instead of 200 for full-dataset convergence.

---

## Step 2: Generating Results

### Table 1 — Model Type Comparison (Paper Sec 5.4)

**CUB** (evaluation script auto-detects and averages fold0–4):
```bash
python scripts/reproduce/table1.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --checkpoint_fc checkpoints/fold0/fc_bce_cub_fc_40.pt \
    --checkpoint_conv checkpoints/fold0/conv_bce_cub_conv5_3_40.pt \
    --checkpoint_fc_conv checkpoints/fold0/fc_conv_bce_cub_conv5_3_40.pt \
    --out_dir results
```

**Flowers** (use `--flowers_checkpoint_*` — different from CUB args):
```bash
python scripts/reproduce/table1.py \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --flowers_checkpoint_fc checkpoints/fold0/fc_bce_flowers_fc_20.pt \
    --flowers_checkpoint_conv checkpoints/fold0/conv_bce_flowers_conv5_3_20.pt \
    --flowers_checkpoint_fc_conv checkpoints/fold0/fc_conv_bce_flowers_conv5_3_20.pt \
    --out_dir results
```

### Table 2 — Loss Function Comparison (Paper Sec 5.5)

```bash
python scripts/reproduce/table2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --checkpoint_bce checkpoints/fold0/fc_bce_cub_fc_40.pt \
    --checkpoint_hinge checkpoints/fc_hinge_cub_fc_40.pt \
    --checkpoint_euclidean checkpoints/fc_euclidean_cub_fc_40.pt \
    --out_dir results
```

### Table 3 — Conv Layer Ablation (Paper Sec 5.6)

```bash
python scripts/reproduce/table3.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --checkpoint_conv5 checkpoints/fold0/fc_conv_bce_cub_conv5_3_40.pt \
    --checkpoint_conv4 checkpoints/fc_conv_bce_cub_conv4_3_40.pt \
    --checkpoint_pool5 checkpoints/fc_conv_bce_cub_pool5_40.pt \
    --out_dir results
```

### Table 4 — Supervised 50/50 Baseline (Paper Sec 5.7)

```bash
# CUB
python scripts/reproduce/table4.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --checkpoint_fc checkpoints/fc_bce_cub_fc_0_tr0.5.pt \
    --checkpoint_fc_conv checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt \
    --out_dir results

# Flowers
python scripts/reproduce/table4.py \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --checkpoint_fc checkpoints/fc_bce_flowers_fc_0_tr0.5.pt \
    --checkpoint_fc_conv checkpoints/fc_conv_bce_flowers_conv5_3_0_tr0.5.pt \
    --out_dir results
```

### Figure 2 — Word Sensitivity + Nearest-Neighbour Retrieval (Paper Sec 5.8)

```bash
python scripts/reproduce/figure2.py \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --checkpoint_fc checkpoints/fold0/fc_bce_cub_fc_40.pt \
    --out_dir results
```

### Evaluation Script Arguments

| Argument | Description |
|----------|-------------|
| `--cub_root` | CUB image root directory |
| `--flowers_root` | Oxford Flowers-102 image root directory |
| `--wikipedia_birds` | CUB Wikipedia text path |
| `--wikipedia_flowers` | Flowers Wikipedia text path |
| `--checkpoint_*` | Explicit checkpoint paths (recommended) |
| `--checkpoint_dir` | Checkpoint directory for auto-detection |
| `--out_dir` | Output directory (default: `results/`) |
| `--device` | `cuda` or `cpu` |
| `--batch_size` | Evaluation batch size (default: 64) |

### Output Files

| Type | Location | Use |
|------|----------|-----|
| CSV | `results/tables/Table*.csv` | Spreadsheet viewing |
| LaTeX | `results/tex/Table*.tex` | Paper tables |
| PDF | `results/AllTables.pdf` | Compiled output |
| Figures | `results/figures/Figure*.png` | Paper figures |

---

## Checkpoint Auto-detection

When no explicit `--checkpoint_*` path is provided, scripts search `checkpoint_dir` using named patterns.

**Resolution priority:**
1. Explicit path (if provided and exists)
2. Scan `checkpoint_dir` and `fold{i}/` subdirectories; pick by longest name-prefix match between the lookup key and the file stem (deterministic, not time-based)
3. Legacy names for backward compatibility: `fc.pt`, `conv.pt`, etc.

**CV fold aggregation:**
When a key matches files in multiple `fold{i}/` directories, `resolve_cv_checkpoints` collects all folds and the evaluation script averages results across folds.

**Dataset-specific keys** prevent cross-dataset contamination:
- `fc_bce_cub` → `fold{i}/fc_bce_cub_fc_*.pt` (CUB only)
- `fc_bce_flowers` → `fold{i}/fc_bce_flowers_fc_*.pt` (Flowers only)
- `fc_conv_cub_conv5_3` → `fold{i}/fc_conv_*_cub_conv5_3_*.pt` (Table 3 CUB conv5_3 only)

---

## Implementation Details

### Paper Alignment

| Component | Paper Spec (Section) | Implementation |
|-----------|---------------------|----------------|
| VGG-19 | ImageNet pretrained, frozen (Sec 5.1) | torchvision VGG19, no fine-tuning |
| Image preprocessing | Shortest side → 224px, center crop 224×224 (Sec 5.1) | torchvision transforms |
| ft(·) text encoder | p → 300 → k, k=50 (Sec 3.2, 5.1) | Linear(9763, 300) → ReLU → Linear(300, 50) |
| gv(·) image encoder | 4096 → 300 → k (Sec 3.2, 5.1) | Linear(4096, 300) → ReLU → Linear(300, 50) |
| Conv branch g'v(·) | K'=5 filters 3×3, conv5_3 512×14×14 (Sec 3.3, 5.1) | Conv2d(512→5, 3×3) + predicted K'×3×3 |
| Joint model | ŷ = w^T gv(x) + o(conv(w', g'v(a))) (Eq. 5, Sec 3.4) | `_forward_fc() + _forward_conv()` |
| Initialization | Small init (Sec 5.1) | std=0.01 for all weight-prediction layers |
| Loss: BCE | Eq. 6, sum reduction (Sec 4.1) | `F.binary_cross_entropy_with_logits(reduction="sum")` |
| Loss: Hinge | Eq. 7, margin=1 (Sec 4.2) | `F.relu(margin - targets * scores).sum()` |
| Loss: Euclidean | MSE in embedding space (Sec 4.2.1) | `F.mse_loss(scores, targets, reduction="sum")` |
| Minibatch | Sum only over classes in batch, O(B×U) (Sec 4.1) | `torch.unique` dynamic class selection |
| Optimizer | Adam (Sec 5.1) | Adam(lr=1e-4) fc; Adam(lr=5e-4) conv/fc+conv* |
| Batch size | 200 (Sec 5.1) | Default: 200 |
| TF-IDF | 9763-d, log normalization (Sec 5.2) | `sublinear_tf=True`, `max_features=9763` |
| Pooling | Global average pooling for conv (Sec 3.3) | `out.flatten(2).mean(2)` |
| CV | 5-fold cross-validation (Sec 5.2) | `--n_folds 5` (default) |

*lr=5e-4 for conv/fc+conv is empirical; the paper uses lr=1e-4 for all models.

### Minibatch Loss (Paper Sec. 4, Eq. 6–7)

The model scores all classes, but loss is computed only over classes present in the batch (O(B×U), U ≤ B):

```python
all_scores = model(images, text_features)          # [B, C_all]
unique_classes, inverse = torch.unique(labels, return_inverse=True)
batch_scores = all_scores[:, unique_classes]        # [B, U]
```

### Known Deviations from Paper

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|-------------------|--------|
| Dataset (Tables 2, 3, Fig 2) | CUB-200-2010 (6,033 images) | CUB-200-2011 (11,788 images) | CUB-2010 no longer available |
| Learning rate (conv/fc+conv) | lr=1e-4 for all models | lr=5e-4 for conv/fc+conv | Empirical: improves convergence |
| Cross-validation (Tables 2–4) | 5-fold CV for all CUB experiments | Single run (`--n_folds 1`) | Training cost; Table 1 uses 5-fold |
| Framework | Torch (Lua) | PyTorch | Original Torch is deprecated |
| Sampler | Not specified (random) | ClassAwareSampler (default) | Better class diversity per batch |
| Early stopping | Not mentioned | Enabled by default (patience=20) | Prevents overfitting |
| Fine-tuning | Table 4 models are fine-tuned on full dataset | No fine-tuning | May affect Table 4 performance |
| Wikipedia text | Texts collected ~10 years ago | Texts collected by us | Original repository not publicly available |

---

## Data Preparation

The project expects data under `data/`:
- `images/birds/`: CUB-200-2011 images (200 classes, 11,788 images)
- `images/flowers/`: Oxford Flowers-102 images (102 classes, 8,189 images)
- `wikipedia/birds.jsonl`: Wikipedia texts for bird classes (included in repository)
- `wikipedia/flowers.jsonl`: Wikipedia texts for flower classes (included in repository)

Images are downloaded via `data/download_dataset.py` (or automatically by `train.sh`). Alternatively, download manually from the [Google Drive link](https://drive.google.com/file/d/1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz/view) and unzip to `data/images/`.

Wikipedia texts were collected by us (the original paper's CUB-200-2010 texts are not publicly available).

VGG-19 features are extracted on-the-fly (frozen weights, no fine-tuning) following the paper protocol.

---

## Reproducibility

The code is **reproducible**: with the same command you get the same results across runs.

- **Seed**: Training and evaluation use a fixed random seed (default `42`). Data splits, model initialization, and batch order are deterministic. Use `--seed` to change it.
- **GPU**: By default cuDNN uses non-deterministic algorithms for speed. For bit-exact GPU reproducibility, run with `--deterministic` (may be slower).
- **Evaluation**: `scripts/evaluate.py` sets the same seed (42) so evaluation is deterministic.

### Evaluate a single checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/fc_bce_cub_fc_40.pt --dataset cub
```

Use the same `--dataset`, `--n_unseen`, `--train_ratio`, `--text_encoder`, and `--image_backbone` as training so the data split and model architecture match.

### Performance Optimizations (do not affect correctness)

- Multi-worker data loading (`NUM_WORKERS=8`)
- `pin_memory` and `persistent_workers` for faster GPU transfer
- `cudnn.benchmark=True` for optimized conv algorithms
- Vectorized label mapping and cached transform objects
- ClassAwareSampler (pre-computed batches for better class diversity)

---

## Reference

- **Paper**: Ba et al., "Predicting deep zero-shot convolutional neural networks using textual descriptions", ICCV 2015
- **Data**: `data/` directory (auto-loaded)
- **VGG-19 weights**: downloaded automatically on first run
