"""
TableInnov: Innovation ablation comparison table (Ba et al. ICCV 2015 extensions).

Columns: one per innovation variant (A0–A7 loss ablation, B1–B3 text encoder,
         C1–C2 image backbone with fc+conv).
Rows:    PR-AUC unseen / ROC-AUC unseen / PR-AUC seen / ROC-AUC seen.

Handles both single checkpoints and CV fold checkpoints automatically.

Output: results/tables/TableInnov.csv
        results/tex/TableInnov.tex

Usage:
    python scripts/reproduce/table_innov.py \\
        --cub_root data/images/birds \\
        [--innov_dir checkpoints/innov] \\
        [--n_folds 0]   # 0 = auto-detect from fold* dirs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import prepare_birds_zero_shot, ImageClassDataset
from scripts.reproduce.common import get_tables_dir, get_tex_dir, write_table_csv
from scripts.reproduce.eval_utils import (
    compute_zero_shot_metrics,
    evaluate_cv_folds,
    load_model,
    run_inference,
)
from utils.config import K, FT_HIDDEN, GV_HIDDEN, CONV_CHANNELS, CONV_FEATURE_LAYER, _TEXT_ENCODER_DIMS


# ---------------------------------------------------------------------------
# Innovation registry
# Each entry: (id, col_name, model_type, ckpt_name, text_encoder, image_backbone)
# col_name: short label used as column header in the table
# ckpt_name: base filename (without .pt) under innov_dir
# ---------------------------------------------------------------------------
_INNOVATIONS = [
    # Section A – loss function ablation (fc+conv, VGG-19, TF-IDF)
    ("A0", "BCE",       "fc+conv", "fc_conv_bce",          "tfidf",       "vgg19"),
    ("A1", "CLIP-loss", "fc+conv", "fc_conv_clip",         "tfidf",       "vgg19"),
    ("A2", "CenterAln", "fc+conv", "fc_conv_center_align", "tfidf",       "vgg19"),
    ("A3", "EmbMSE",    "fc+conv", "fc_conv_embedding_mse","tfidf",       "vgg19"),
    # Section B – text encoder ablation (fc+conv, VGG-19)
    ("B1", "SBERT",     "fc+conv", "fc_conv_sbert",        "sbert",       "vgg19"),
    ("B2", "SBERT-m",   "fc+conv", "fc_conv_sbert_multi",  "sbert_multi", "vgg19"),
    ("B3", "CLIP-text", "fc+conv", "fc_conv_clip_text",    "clip",        "vgg19"),
    # Section C – image backbone ablation (fc+conv, TF-IDF)
    ("C1", "DenseNet121","fc+conv", "fc_conv_densenet121", "tfidf",       "densenet121"),
    ("C2", "ResNet50",  "fc+conv", "fc_conv_resnet50",     "tfidf",       "resnet50"),
]

_METRIC_KEYS = [
    ("pr_auc_unseen",  "PR-AUC unseen"),
    ("roc_auc_unseen", "ROC-AUC unseen"),
    ("pr_auc_seen",    "PR-AUC seen"),
    ("roc_auc_seen",   "ROC-AUC seen"),
]

_TEX_METRICS_ORDER = [
    ("pr_auc_seen", "PR-AUC"),
    ("roc_auc_seen", "ROC-AUC"),
    ("pr_auc_unseen", "PR-AUC"),
    ("roc_auc_unseen", "ROC-AUC"),
]


def _find_checkpoints(innov_dir: Path, ckpt_name: str) -> tuple[list[str], str]:
    """Return (paths, mode) where mode is 'single', 'cv', or 'missing'."""
    single = innov_dir / f"{ckpt_name}.pt"
    if single.exists():
        return [str(single)], "single"

    fold_ckpts = sorted(
        innov_dir.glob(f"fold*/{ckpt_name}.pt"),
        key=lambda p: p.parent.name,
    )
    if fold_ckpts:
        return [str(p) for p in fold_ckpts], "cv"

    return [], "missing"


def _fmt(val, std=None) -> str:
    if val is None:
        return "—"
    return f"{val:.3f}"


def _parse_float(cell: str) -> float | None:
    """Parse numeric value from a table cell (supports '0.123' or '0.123±0.004')."""
    if not cell or cell == "—":
        return None
    # CV format uses "mean±std"
    head = cell.split("±", 1)[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def _bold_best_by_column(rows: list[list[str]], value_col_indices: list[int]) -> list[list[str]]:
    """
    Bold best (max) value per metric column within a section.
    rows: list of rows like [name, v1, v2, v3, v4] (all strings).
    value_col_indices: indices of metric columns to consider (e.g. [1,2,3,4]).
    """
    best: dict[int, float] = {}
    for j in value_col_indices:
        vals = [_parse_float(r[j]) for r in rows]
        vals2 = [v for v in vals if v is not None]
        if vals2:
            best[j] = max(vals2)

    out = [r[:] for r in rows]
    for i, r in enumerate(out):
        for j in value_col_indices:
            v = _parse_float(r[j])
            if v is None:
                continue
            # Exact equality is fine here: values are formatted to 3 decimals in _fmt()
            if j in best and f"{v:.3f}" == f"{best[j]:.3f}":
                out[i][j] = r"\bfseries " + r[j]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate innovation ablation table (TableInnov).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cub_root", default="", help="CUB-200-2011 images root")
    parser.add_argument("--wikipedia_birds", default="data/wikipedia/birds.jsonl")
    parser.add_argument("--innov_dir", default="checkpoints/innov",
                        help="Directory containing innovation checkpoints")
    parser.add_argument("--n_folds", type=int, default=0,
                        help="Expected CV folds (0 = auto-detect from fold* dirs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_unseen", type=int, default=40)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory (default: results/)")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    innov_dir = Path(args.innov_dir)
    if not innov_dir.is_absolute():
        innov_dir = code_root / innov_dir

    tables_dir = get_tables_dir()
    tex_dir = get_tex_dir()
    if args.out_dir:
        tables_dir = Path(args.out_dir) / "tables"
        tex_dir = Path(args.out_dir) / "tex"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tex_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Prepare CUB dataset (single split for single-checkpoint eval)
    jsonl_birds = code_root / args.wikipedia_birds
    cub_ok = args.cub_root and Path(args.cub_root).exists() and jsonl_birds.exists()
    if not cub_ok:
        print(f"[WARN] CUB data not found at '{args.cub_root}'. "
              "Pass --cub_root to enable evaluation.")

    # Pre-load dataset for single-checkpoint evaluation
    single_data = None
    if cub_ok:
        out = prepare_birds_zero_shot(
            args.cub_root, str(jsonl_birds),
            n_unseen=args.n_unseen,
            unseen_seed=args.seed,
            split_seed=args.seed,
            train_ratio_seen=args.train_ratio,
        )
        _, _, test_p, test_l, _, text_feat, seen_idx, unseen_idx = out
        single_data = dict(
            test_p=test_p, test_l=test_l,
            seen_idx=seen_idx, unseen_idx=unseen_idx,
            num_classes=len(seen_idx) + len(unseen_idx),
        )
        # text_feat is per-text_encoder; we'll re-use for tfidf runs and reload for others

    # -----------------------------------------------------------------
    # Evaluate each innovation and collect metrics
    # -----------------------------------------------------------------
    results: dict[str, dict] = {}  # ckpt_name → metrics dict

    for innov_id, col_name, model_type, ckpt_name, text_encoder, image_backbone in _INNOVATIONS:
        print(f"\n[{innov_id}] {col_name}  ckpt={ckpt_name}")
        ckpt_paths, mode = _find_checkpoints(innov_dir, ckpt_name)

        if mode == "missing":
            print(f"  → checkpoint not found, skipping")
            results[ckpt_name] = {}
            continue

        if not cub_ok:
            results[ckpt_name] = {}
            continue

        text_dim = _TEXT_ENCODER_DIMS[text_encoder]
        model_kw = dict(
            text_dim=text_dim,
            k=K,
            ft_hidden=FT_HIDDEN,
            gv_hidden=GV_HIDDEN,
            conv_channels=CONV_CHANNELS,
            conv_feature_layer=CONV_FEATURE_LAYER,
            image_backbone=image_backbone,
        )

        try:
            if mode == "cv" and len(ckpt_paths) >= 2:
                print(f"  → CV evaluation ({len(ckpt_paths)} folds)")
                m = evaluate_cv_folds(
                    ckpt_paths,
                    model_type=model_type,
                    dataset="cub",
                    images_root=args.cub_root,
                    wikipedia_jsonl=str(jsonl_birds),
                    device=device,
                    batch_size=args.batch_size,
                    base_seed=args.seed,
                    n_unseen=args.n_unseen,
                    train_ratio=args.train_ratio,
                    text_encoder=text_encoder,
                    **model_kw,
                )
            else:
                # Single checkpoint evaluation
                print(f"  → single checkpoint: {ckpt_paths[0]}")
                # For non-tfidf encoders, text_feat needs to be re-prepared
                if text_encoder != "tfidf":
                    out = prepare_birds_zero_shot(
                        args.cub_root, str(jsonl_birds),
                        n_unseen=args.n_unseen,
                        unseen_seed=args.seed,
                        split_seed=args.seed,
                        train_ratio_seen=args.train_ratio,
                        text_encoder=text_encoder,
                    )
                    _, _, tp, tl, _, tf, si, ui = out
                    text_t = torch.from_numpy(tf).float()
                    nc = len(si) + len(ui)
                    loader = DataLoader(
                        ImageClassDataset(tp, tl),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                    )
                    seen_idx_eval, unseen_idx_eval = si, ui
                else:
                    text_t = torch.from_numpy(text_feat).float()
                    nc = single_data["num_classes"]
                    loader = DataLoader(
                        ImageClassDataset(single_data["test_p"], single_data["test_l"]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                    )
                    seen_idx_eval = single_data["seen_idx"]
                    unseen_idx_eval = single_data["unseen_idx"]

                model = load_model(model_type, ckpt_paths[0], device, **model_kw)
                scores, labels = run_inference(
                    model, loader, text_t, device, nc, desc=f"{innov_id} inference"
                )
                m = compute_zero_shot_metrics(scores, labels, seen_idx_eval, unseen_idx_eval)

            results[ckpt_name] = m
            pr_u = _fmt(m.get("pr_auc_unseen"), m.get("pr_auc_unseen_std"))
            roc_u = _fmt(m.get("roc_auc_unseen"), m.get("roc_auc_unseen_std"))
            print(f"  PR-AUC unseen={pr_u}  ROC-AUC unseen={roc_u}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            results[ckpt_name] = {}

    # -----------------------------------------------------------------
    # Build table: rows = metrics, columns = innovations
    # -----------------------------------------------------------------
    col_headers = [f"{iid} {cname}" for iid, cname, *_ in _INNOVATIONS]
    headers = ["Metric"] + col_headers

    rows = []
    for metric_key, metric_label in _METRIC_KEYS:
        row = [metric_label]
        for _, _, _, ckpt_name, *_ in _INNOVATIONS:
            m = results.get(ckpt_name, {})
            val = m.get(metric_key)
            std = m.get(metric_key + "_std")
            row.append(_fmt(val, std))
        rows.append(row)

    # CSV
    import csv
    csv_path = tables_dir / "TableInnov.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    # LaTeX
    def _section_rows(prefix: str) -> list[list[str]]:
        section = [(iid, cname, ckpt) for (iid, cname, _, ckpt, *_rest) in _INNOVATIONS if iid.startswith(prefix)]
        out_rows: list[list[str]] = []
        for iid, cname, ckpt_name in section:
            m = results.get(ckpt_name, {})
            vals = []
            for k, _lbl in _TEX_METRICS_ORDER:
                vals.append(_fmt(m.get(k), m.get(k + "_std")))
            out_rows.append([cname] + vals)
        return _bold_best_by_column(out_rows, [1, 2, 3, 4])

    loss_rows = _section_rows("A")      # [Setting, PR/ROC seen, PR/ROC unseen]
    text_rows = _section_rows("B")      # [Text encoder, ...]
    bb_rows = _section_rows("C")        # [Backbone, ...]

    latex_lines = [
        r"\documentclass{article}",
        r"",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{siunitx}",
        r"\usepackage{threeparttable}",
        r"",
        r"\sisetup{",
        r"  detect-weight=true,",
        r"  detect-inline-weight=math,",
        r"  group-digits=false,",
        r"  input-symbols = {—},",
        r"  table-format=1.3,",
        r"}",
        r"",
        r"\begin{document}",
        r"",
        r"% ===================== Table 1: Loss =====================",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{threeparttable}",
        r"\caption{Ablation of loss functions on CUB-200-2011}",
        r"\label{tab:innov_loss}",
        r"\begin{tabular}{l S S S S}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Setting & {PR-AUC} & {ROC-AUC} & {PR-AUC} & {ROC-AUC} \\",
        r"\midrule",
    ]
    for r in loss_rows:
        latex_lines.append(" & ".join(r) + r" \\")
    latex_lines += [
        r"TF-IDF & 0.37 & 0.98 & 0.13 & 0.85 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"",
        r"% ===================== Table 2: Text Encoder =====================",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Ablation of text encoders (CUB-200-2011)}",
        r"\label{tab:innov_text}",
        r"\begin{tabular}{l S S S S}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Text encoder & {PR-AUC} & {ROC-AUC} & {PR-AUC} & {ROC-AUC} \\",
        r"\midrule",
    ]
    for r in text_rows:
        latex_lines.append(" & ".join(r) + r" \\")
    latex_lines += [
        r"TF-IDF & 0.37 & 0.98 & 0.13 & 0.85 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"% ===================== Table 3: Backbone =====================",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Ablation of image backbones (CUB-200-2011)}",
        r"\label{tab:innov_backbone}",
        r"\begin{tabular}{l S S S S}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Backbone & {PR-AUC} & {ROC-AUC} & {PR-AUC} & {ROC-AUC} \\",
        r"\midrule",
    ]
    for r in bb_rows:
        latex_lines.append(" & ".join(r) + r" \\")
    latex_lines += [
        r"VGG-19 & 0.37 & 0.98 & 0.13 & 0.85 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\end{document}",
        r"",
    ]

    tex_path = tex_dir / "TableInnov.tex"
    tex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
