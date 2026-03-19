#!/usr/bin/env bash
# =============================================================================
# reproduce.sh -- Generate paper tables and figures (Ba et al. ICCV 2015)
#
# Requires bash 4+ (associative arrays). On macOS, use: brew install bash
#
# Checkpoint mode (set USE_CONFIG below):
#   CV     -- train.sh default: 5-fold CV, checkpoints in checkpoints/fold{i}/
#   SINGLE -- single run (--n_folds 1), checkpoints in checkpoints/
# =============================================================================
set -euo pipefail

# -- LaTeX installation (cross-platform) --------------------------------------
echo "=== Checking LaTeX installation ==="
if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    if ! command -v xelatex &> /dev/null; then
        echo "Installing texlive (this may take a while)..."
        sudo apt update
        sudo apt install -y texlive-full
    fi
elif command -v brew &> /dev/null; then
    echo "Detected macOS system"
    if ! command -v xelatex &> /dev/null; then
        echo "Installing MacTeX (this may take a while)..."
        brew install mactex
    fi
else
    echo "WARNING: No package manager found. Please install LaTeX manually."
    echo "  - Linux: sudo apt install texlive-full"
    echo "  - macOS: brew install mactex"
fi

# -- Download checkpoints from HuggingFace ------------------------------------
echo ""
echo "=== Downloading checkpoints from HuggingFace ==="
mkdir -p checkpoints
cd checkpoints
if [ ! -f "fc_bce_cub_fc_40.pt" ]; then
    git clone https://huggingface.co/LiXiuyin/zero-shot-cnn-comp7404-group17 temp_repo
    mv temp_repo/* .
    rm -rf temp_repo
    echo "Checkpoints downloaded successfully"
else
    echo "Checkpoints already exist, skipping download"
fi
cd ..

# -- Environment --------------------------------------------------------------
echo ""
echo "=== Setting up environment ==="
uv sync && source .venv/bin/activate

echo ""
echo "=== Downloading datasets (if needed) ==="
cd data && python download_dataset.py && cd ..

# -- Checkpoint path configuration --------------------------------------------
# declare -A is required for associative arrays (bash 4+).
# Keys must exactly match the get_checkpoint() call sites below.

# Config 1: single-fold training (--n_folds 1), checkpoints directly in checkpoints/
declare -A CHECKPOINTS_SINGLE
CHECKPOINTS_SINGLE=(
    [fc]="checkpoints/fc_bce_cub_fc_40.pt"
    [conv]="checkpoints/conv_bce_cub_conv5_3_40.pt"
    [fc_conv]="checkpoints/fc_conv_bce_cub_conv5_3_40.pt"
    [fc_bce]="checkpoints/fc_bce_cub_fc_40.pt"
    [fc_hinge]="checkpoints/fc_hinge_cub_fc_40.pt"
    [fc_euclidean]="checkpoints/fc_euclidean_cub_fc_40.pt"
    [fc_conv4]="checkpoints/fc_conv_bce_cub_conv4_3_40.pt"
    [fc_conv5]="checkpoints/fc_conv_bce_cub_conv5_3_40.pt"
    [pool5]="checkpoints/fc_conv_bce_cub_pool5_40.pt"
    [fc_5050]="checkpoints/fc_bce_cub_fc_0_tr0.5.pt"
    [fc_conv_5050]="checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt"

    [fc_flowers]="checkpoints/fc_bce_flowers_fc_20.pt"
    [conv_flowers]="checkpoints/conv_bce_flowers_conv5_3_20.pt"
    [fc_conv_flowers]="checkpoints/fc_conv_bce_flowers_conv5_3_20.pt"
    [fc_flowers_5050]="checkpoints/fc_bce_flowers_fc_0_tr0.5.pt"
    [fc_conv_flowers_5050]="checkpoints/fc_conv_bce_flowers_conv5_3_0_tr0.5.pt"
)

# Config 2: 5-fold CV training (default --n_folds 5), checkpoints in fold{i}/
# Use fold0 as the representative checkpoint for single-model reproduce scripts;
# table scripts with --n_folds > 0 will automatically aggregate all folds.
declare -A CHECKPOINTS_CV
CHECKPOINTS_CV=(
    [fc]="checkpoints/fold0/fc_bce_cub_fc_40.pt"
    [conv]="checkpoints/fold0/conv_bce_cub_conv5_3_40.pt"
    [fc_conv]="checkpoints/fold0/fc_conv_bce_cub_conv5_3_40.pt"
    [fc_bce]="checkpoints/fold0/fc_bce_cub_fc_40.pt"
    # hinge/euclidean only exist at root (no CV folds trained for these losses)
    [fc_hinge]="checkpoints/fc_hinge_cub_fc_40.pt"
    [fc_euclidean]="checkpoints/fc_euclidean_cub_fc_40.pt"
    # conv4_3 and pool5 only exist at root (no CV folds trained for these layers)
    [fc_conv4]="checkpoints/fc_conv_bce_cub_conv4_3_40.pt"
    [fc_conv5]="checkpoints/fold0/fc_conv_bce_cub_conv5_3_40.pt"
    [pool5]="checkpoints/fc_conv_bce_cub_pool5_40.pt"
    # 50/50 split checkpoints only exist at root (Table 4)
    [fc_5050]="checkpoints/fc_bce_cub_fc_0_tr0.5.pt"
    [fc_conv_5050]="checkpoints/fc_conv_bce_cub_conv5_3_0_tr0.5.pt"

    [fc_flowers]="checkpoints/fold0/fc_bce_flowers_fc_20.pt"
    [conv_flowers]="checkpoints/fold0/conv_bce_flowers_conv5_3_20.pt"
    [fc_conv_flowers]="checkpoints/fold0/fc_conv_bce_flowers_conv5_3_20.pt"
    # 50/50 split Flowers checkpoints only exist at root
    [fc_flowers_5050]="checkpoints/fc_bce_flowers_fc_0_tr0.5.pt"
    [fc_conv_flowers_5050]="checkpoints/fc_conv_bce_flowers_conv5_3_0_tr0.5.pt"
)

# Select config: train.sh default is 5-fold CV -> use CV here.
# Switch to SINGLE if you trained with --n_folds 1.
USE_CONFIG="CV"       # 5-fold CV (matches train.sh default)
# USE_CONFIG="SINGLE" # single-fold (switch when trained with --n_folds 1)

if [ "$USE_CONFIG" = "CV" ]; then
    echo "=== Using CV training checkpoints (checkpoints/fold0/) ==="
else
    echo "=== Using single-fold training checkpoints (checkpoints/) ==="
fi

# Lookup function: direct associative array access (no key-losing copy)
get_checkpoint() {
    local key=$1
    if [ "$USE_CONFIG" = "CV" ]; then
        echo "${CHECKPOINTS_CV[$key]:-}"
    else
        echo "${CHECKPOINTS_SINGLE[$key]:-}"
    fi
}

OUT_DIR="results"

# -- Table 1: Model type comparison (CUB + Flowers) ---------------------------
echo ""
echo "=== Generating Table 1: Model type comparison ==="

python scripts/reproduce/table1.py \
    --checkpoint_fc "$(get_checkpoint 'fc')" \
    --checkpoint_conv "$(get_checkpoint 'conv')" \
    --checkpoint_fc_conv "$(get_checkpoint 'fc_conv')" \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

python scripts/reproduce/table1.py \
    --flowers_checkpoint_fc "$(get_checkpoint 'fc_flowers')" \
    --flowers_checkpoint_conv "$(get_checkpoint 'conv_flowers')" \
    --flowers_checkpoint_fc_conv "$(get_checkpoint 'fc_conv_flowers')" \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 2: Loss function comparison (CUB) ----------------------------------
echo ""
echo "=== Generating Table 2: Loss function comparison ==="

python scripts/reproduce/table2.py \
    --checkpoint_bce "$(get_checkpoint 'fc_bce')" \
    --checkpoint_hinge "$(get_checkpoint 'fc_hinge')" \
    --checkpoint_euclidean "$(get_checkpoint 'fc_euclidean')" \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 3: Conv feature layer ablation (CUB) -------------------------------
echo ""
echo "=== Generating Table 3: Conv feature layer ablation ==="

python scripts/reproduce/table3.py \
    --checkpoint_conv4 "$(get_checkpoint 'fc_conv4')" \
    --checkpoint_conv5 "$(get_checkpoint 'fc_conv5')" \
    --checkpoint_pool5 "$(get_checkpoint 'pool5')" \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

# -- Table 4: Supervised baseline 50/50 split ---------------------------------
echo ""
echo "=== Generating Table 4: Supervised baseline (50/50 split) ==="

python scripts/reproduce/table4.py \
    --checkpoint_fc "$(get_checkpoint 'fc_5050')" \
    --checkpoint_fc_conv "$(get_checkpoint 'fc_conv_5050')" \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

python scripts/reproduce/table4.py \
    --checkpoint_fc "$(get_checkpoint 'fc_flowers_5050')" \
    --checkpoint_fc_conv "$(get_checkpoint 'fc_conv_flowers_5050')" \
    --flowers_root data/images/flowers \
    --wikipedia_flowers data/wikipedia/flowers.jsonl \
    --out_dir "$OUT_DIR"

# -- Compile all tables to LaTeX ----------------------------------------------
echo ""
echo "=== Compiling all tables to LaTeX ==="

python scripts/reproduce/compile_all_tables.py
if [ -f "results/tex/AllTables.tex" ]; then
    xelatex -output-directory=results results/tex/AllTables.tex
    echo "LaTeX compilation complete"
else
    echo "WARNING: AllTables.tex not found, skipping LaTeX compilation"
fi

# -- Figure 2: Word sensitivity + Nearest neighbor retrieval ------------------
echo ""
echo "=== Generating Figure 2: Word sensitivity + Nearest neighbor ==="

python scripts/reproduce/figure2.py \
    --checkpoint_fc "$(get_checkpoint 'fc')" \
    --cub_root data/images/birds \
    --wikipedia_birds data/wikipedia/birds.jsonl \
    --out_dir "$OUT_DIR"

echo ""
echo "=== reproduce.sh complete ==="
echo ""
echo "Results saved to: $OUT_DIR/"
echo "  - tables/ : CSV data files"
echo "  - tex/    : LaTeX table files"
echo "  - figures/: Figure images"
