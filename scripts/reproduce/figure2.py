"""
Figure 2. [LEFT]: Word sensitivities of unseen classes using the fc model on CUB-200-2011.
        [RIGHT]: The ability of the text features to describe visual features (nearest neighbors).
Results are computed from the loaded fc model and CUB data (word ablation + nearest-neighbor retrieval).
Reproduces Figure 2 from Ba et al. ICCV 2015 (original used CUB200-2010).
Output: results/figures/Figure2.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import ImageClassDataset, prepare_birds_zero_shot
from data.text_processor import texts_to_tfidf
from scripts.reproduce.common import FIG_DPI, FIG_TWO_COL_INCH, get_figures_dir, resolve_checkpoint
from scripts.reproduce.eval_utils import load_model
from utils.config import (
    CONV_CHANNELS,
    CONV_FEATURE_LAYER,
    FT_HIDDEN,
    GV_HIDDEN,
    K,
    TEXT_DIM,
)


def _pr_auc_class(scores: np.ndarray, labels: np.ndarray, class_idx: int) -> float:
    y_true = (labels == class_idx).astype(np.float64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    try:
        return float(average_precision_score(y_true, scores[:, class_idx]))
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 from model outputs.")
    parser.add_argument("--cub_root", type=str, default="", help="Path to CUB_200_2011")
    parser.add_argument("--wikipedia_birds", type=str, default="data/wikipedia/birds.jsonl", help="Path to birds Wikipedia text")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Default dir for checkpoints (used when --checkpoint_fc not set)")
    parser.add_argument("--checkpoint_fc", type=str, default="", help="fc model checkpoint")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_unseen_show", type=int, default=3, help="Number of unseen classes to plot (left panel)")
    parser.add_argument("--max_words_ablate", type=int, default=0, help="Max TF-IDF dims to try per class (0 = all non-zero, as in paper)")
    args = parser.parse_args()

    figures_dir = get_figures_dir()
    if args.out_dir:
        figures_dir = Path(args.out_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

    code_root = Path(__file__).resolve().parents[2]
    device = torch.device(args.device)
    model_kw = dict(
        text_dim=TEXT_DIM,
        k=K,
        ft_hidden=FT_HIDDEN,
        gv_hidden=GV_HIDDEN,
        conv_channels=CONV_CHANNELS,
        conv_feature_layer=CONV_FEATURE_LAYER,
    )

    jsonl_birds = code_root / args.wikipedia_birds
    checkpoint_fc = resolve_checkpoint(args.checkpoint_fc, args.checkpoint_dir, "fc")
    if not args.cub_root or not Path(args.cub_root).exists() or not jsonl_birds.exists() or not checkpoint_fc:
        print("Provide --cub_root, --checkpoint_fc or --checkpoint_dir (and ensure wikipedia jsonl exists) to compute Figure 2.")
        return

    out = prepare_birds_zero_shot(args.cub_root, jsonl_birds)
    train_p, train_l, test_p, test_l, class_names, text_feat, seen_idx, unseen_idx = out
    text_t = torch.from_numpy(text_feat).float().to(device)
    loader = DataLoader(
        ImageClassDataset(test_p, test_l),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = load_model("fc", checkpoint_fc, device, **model_kw)

    # Build vocabulary using the SAME text loading pipeline as prepare_birds_zero_shot
    # to ensure dimension-to-word mapping is consistent with text_feat
    from data.dataset import load_from_json
    _, _, class_texts_map, _ = load_from_json(jsonl_birds, Path(args.cub_root) / "images", verbose=False)
    ordered_texts = []
    for cname in class_names:
        parts = cname.split(".", 1)
        class_id = int(parts[0])
        ordered_texts.append(class_texts_map.get(class_id, ""))
    _, vectorizer = texts_to_tfidf(ordered_texts, max_features=9763)
    try:
        vocab = vectorizer.get_feature_names_out()
    except AttributeError:
        vocab = vectorizer.get_feature_names()
    vocab = list(vocab)

    # Full scores and labels for test set
    from tqdm import tqdm

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Computing scores", leave=True):
            imgs = imgs.to(device)
            s = model(imgs, text_t)
            all_scores.append(s.cpu().numpy())
            all_labels.append(lbls.numpy())
    scores_full = np.concatenate(all_scores, axis=0)
    labels_full = np.concatenate(all_labels, axis=0)
    print()  # Add newline for cleaner output

    # Image features g(x) for all test images (for right panel)
    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting features", leave=True):
            imgs = imgs.to(device)
            g = model.image_encoder(imgs)
            feats.append(g.cpu().numpy())
    g_all = np.concatenate(feats, axis=0)
    f_all = model.text_encoder(text_t).detach().cpu().numpy()
    print()  # Add newline for cleaner output

    # Left panel: word sensitivity for first n_unseen_show unseen classes
    unseen_show = unseen_idx[: args.n_unseen_show]
    classes_left = [class_names[c] for c in unseen_show]
    words_per_class = []
    drops_per_class = []
    pr_aucs_per_class = []

    for c in tqdm(unseen_show, desc="Analyzing words", unit="class", leave=True):
        orig_vec = text_feat[c].copy()
        orig_norm = np.linalg.norm(orig_vec)
        pr_baseline = _pr_auc_class(scores_full, labels_full, c)
        pr_aucs_per_class.append(pr_baseline)
        non_zero = np.where(orig_vec != 0)[0]
        if args.max_words_ablate > 0 and len(non_zero) > args.max_words_ablate:
            non_zero = non_zero[np.argsort(-np.abs(orig_vec[non_zero]))[: args.max_words_ablate]]
        drops = []
        for dim in non_zero:
            new_vec = orig_vec.copy()
            new_vec[dim] = 0.0
            new_norm = np.linalg.norm(new_vec)
            if new_norm > 1e-9:
                new_vec = new_vec * (orig_norm / new_norm)
            # Optimization: Use pre-computed image features instead of re-running forward passes
            text_mod = text_t.cpu().numpy().copy()
            text_mod[c] = new_vec.astype(np.float32)
            text_mod_t = torch.from_numpy(text_mod).float().to(device)
            with torch.no_grad():
                f_mod = model.text_encoder(text_mod_t)  # [C, k]
            s_mod = g_all @ f_mod.detach().cpu().numpy().T  # [N, C] - matrix multiplication only
            pr_new = _pr_auc_class(s_mod, labels_full, c)
            drop = pr_baseline - pr_new
            drops.append((dim, drop, vocab[dim] if dim < len(vocab) else f"dim{dim}"))
        drops.sort(key=lambda x: -x[1])
        top5_words = [d[2] for d in drops[:5]]
        top5_drops = [d[1] for d in drops[:5]]
        words_per_class.append(top5_words)
        drops_per_class.append(top5_drops)

    # Right panel: within-class and overall nearest neighbors
    nn_per_class = []
    for c in unseen_show:
        w_c = f_all[c]
        dots = g_all @ w_c
        within_mask = labels_full == c
        if within_mask.sum() > 0:
            within_idx = np.where(within_mask)[0][np.argmax(dots[within_mask])]
        else:
            within_idx = None
        top_overall = np.argsort(-dots)[:3]
        nn_per_class.append((c, within_idx, top_overall))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_show = len(unseen_show)
    n_overall = 3

    # Layout: left panel (vertical bar charts) + right panel (image grid)
    # Left panel: n_show subplots vertically arranged, each with vertical bars
    # Right panel: 1 within-class NN + 3 overall NN = 4 image columns per row
    fig = plt.figure(figsize=(14, max(4.0, n_show * 2.2)))
    plt.rc("font", size=8)

    # Use gridspec: left half for bar charts, right half for image grid
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # --- LEFT panel: word sensitivity bar charts (vertical subplots) ---
    def _clean_name(name: str) -> str:
        """Remove leading number prefix (e.g. '003.') and replace underscores."""
        if "." in name:
            name = name.split(".", 1)[-1]
        return name.replace("_", " ")

    # Create vertical subplots for each class
    gs_left = gs[0].subgridspec(n_show, 1, hspace=0.4)

    for i, (cls, words, drops_vals) in enumerate(zip(classes_left, words_per_class, drops_per_class)):
        ax = fig.add_subplot(gs_left[i])
        n_words = len(words)
        x_pos = np.arange(n_words)

        # Vertical bar chart
        ax.bar(x_pos, drops_vals, width=0.6, color="steelblue", alpha=0.85,
               edgecolor="white", linewidth=0.5)

        # Add word labels below bars
        ax.set_xticks(x_pos)
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=6)

        # Single dashed line for this class's PR-AUC
        pr_auc = pr_aucs_per_class[i]
        ax.axhline(pr_auc, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        # Class name as title
        ax.set_title(_clean_name(cls), fontsize=8, fontweight="bold", pad=3)

        # y-axis label
        ax.set_ylabel("PR-AUC", fontsize=6)

        # Set reasonable limits
        max_drop = max(drops_vals) if drops_vals else 0.1
        ax.set_ylim(0, max(pr_auc, max_drop) * 1.15)

        # Clean styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- RIGHT panel: nearest neighbor images ---
    # Add a header row for column titles
    gs_right = gs[1].subgridspec(
        n_show + 1, 1 + n_overall,
        height_ratios=[0.12] + [1.0] * n_show,
        hspace=0.35, wspace=0.08,
    )

    # Column headers
    ax_hdr0 = fig.add_subplot(gs_right[0, 0])
    ax_hdr0.text(0.5, 0.2, "Within-class\nnearest neighbor", ha="center", va="center",
                 fontsize=7, fontweight="bold")
    ax_hdr0.axis("off")

    ax_hdr1 = fig.add_subplot(gs_right[0, 1:])
    ax_hdr1.text(0.5, 0.2, "Overall nearest neighbors", ha="center", va="center",
                 fontsize=7, fontweight="bold")
    ax_hdr1.axis("off")

    for row_i, (c, within_idx, top_overall) in enumerate(nn_per_class):
        data_row = row_i + 1  # offset by header row

        # Within-class nearest neighbor
        ax = fig.add_subplot(gs_right[data_row, 0])
        if within_idx is not None:
            img = Image.open(test_p[within_idx]).convert("RGB")
            ax.imshow(img)
            ax.set_xlabel(_clean_name(class_names[labels_full[within_idx]]),
                          fontsize=5, labelpad=2)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#CCCCCC")
        ax.set_ylabel(_clean_name(class_names[c]), fontsize=7, fontweight="bold",
                       rotation=0, labelpad=60, va="center")

        # Overall nearest neighbors
        for col_j, img_idx in enumerate(top_overall):
            ax = fig.add_subplot(gs_right[data_row, 1 + col_j])
            img = Image.open(test_p[img_idx]).convert("RGB")
            ax.imshow(img)
            ax.set_xlabel(_clean_name(class_names[labels_full[img_idx]]),
                          fontsize=5, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#CCCCCC")

    fig.suptitle(
        "[LEFT] Word sensitivities of unseen classes (fc model, CUB-200-2011)\n"
        "[RIGHT] Text features describing visual features (nearest neighbors)",
        ha="center", va="bottom", fontsize=9, y=0.01
    )
    out_path = figures_dir / "Figure2.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
