"""Shared paths and helpers for reproduction scripts (Ba et al. ICCV 2015)."""
from __future__ import annotations

from pathlib import Path

# Code root = parent of scripts/
CODE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = CODE_ROOT / "results"

# Default data paths (corrected paths based on actual directory structure)
DEFAULT_WIKIPEDIA_BIRDS = "data/wikipedia/birds.jsonl"
DEFAULT_WIKIPEDIA_FLOWERS = "data/wikipedia/flowers.jsonl"

# Default checkpoint directory (relative to CODE_ROOT). Scripts use --checkpoint_dir
# and fallback to these filenames when --checkpoint_* is not set.
DEFAULT_CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAMES = {
    "fc": "fc.pt",
    "conv": "conv.pt",
    "fc_conv": "fc_conv.pt",
    "fc_bce": "fc_bce.pt",
    "fc_hinge": "fc_hinge.pt",
    "fc_euclidean": "fc_euclidean.pt",
    "bce": "bce.pt",
    "hinge": "hinge.pt",
    "euclidean": "euclidean.pt",
    "conv4_3": "conv4_3.pt",
    "conv5_3": "conv5_3.pt",
    "pool5": "pool5.pt",
    # Per-model-type per-loss keys (used by table1 best-across-losses)
    "conv_bce": "conv_bce.pt",
    "conv_hinge": "conv_hinge.pt",
    "conv_euclidean": "conv_euclidean.pt",
    "fc_conv_bce": "fc_conv_bce.pt",
    "fc_conv_hinge": "fc_conv_hinge.pt",
    "fc_conv_euclidean": "fc_conv_euclidean.pt",
}


def resolve_checkpoint(explicit: str, checkpoint_dir: str, key: str) -> str:
    """
    Resolve checkpoint path with support for new detailed naming format (no timestamp).

    Resolution priority:
    1. Explicit path (if provided and exists)
    2. Scan checkpoint_dir for files matching pattern: {model_type}_{loss}_{dataset}_{layer}_{n_unseen}[_tr{ratio}].pt
    3. Fallback to old CHECKPOINT_NAMES for backward compatibility

    Returns empty string if not found.
    """
    # 1. Try explicit path first
    if explicit:
        p = Path(explicit)
        if p.is_absolute() and p.exists():
            return str(p)
        if p.exists():
            return str(p)

    # 2. Try to auto-detect from checkpoint_dir using pattern matching
    dir_to_use = checkpoint_dir if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    if dir_to_use:
        base = Path(dir_to_use)
        if not base.is_absolute():
            base = CODE_ROOT / base

        if base.exists() and base.is_dir():
            # Define patterns for different model types (no timestamp in filename)
            # Filename format: {model_type}_{loss}_{dataset}_{layer}_{n_unseen}[_tr{ratio}].pt
            patterns = {
                # fc models: must NOT be fc_conv, pattern includes "_fc_" (layer field)
                "fc": ["fc_*_cub_fc_*.pt", "fc_*_flowers_fc_*.pt"],
                # Flowers-only fc (prevents CUB checkpoint from being used for Flowers eval)
                "fc_flowers": ["fc_*_flowers_fc_*.pt"],
                # conv models: must NOT be fc_conv, pattern includes "_conv" (layer field)
                "conv": ["conv_*_cub_conv4_3_*.pt", "conv_*_cub_conv5_3_*.pt", "conv_*_cub_pool5_*.pt",
                        "conv_*_flowers_conv4_3_*.pt", "conv_*_flowers_conv5_3_*.pt", "conv_*_flowers_pool5_*.pt"],
                # Flowers-only conv
                "conv_flowers": ["conv_*_flowers_conv4_3_*.pt", "conv_*_flowers_conv5_3_*.pt", "conv_*_flowers_pool5_*.pt"],
                # fc+conv models: must start with "fc_conv_", include specific conv layers
                "fc_conv": ["fc_conv_*_cub_conv4_3_*.pt", "fc_conv_*_cub_conv5_3_*.pt", "fc_conv_*_cub_pool5_*.pt",
                           "fc_conv_*_flowers_conv4_3_*.pt", "fc_conv_*_flowers_conv5_3_*.pt", "fc_conv_*_flowers_pool5_*.pt"],
                # Flowers-only fc+conv
                "fc_conv_flowers": ["fc_conv_*_flowers_conv4_3_*.pt", "fc_conv_*_flowers_conv5_3_*.pt", "fc_conv_*_flowers_pool5_*.pt"],
                # loss functions: match fc models with specific loss (exclude fc_conv)
                "bce": ["fc_bce_cub_*.pt", "fc_bce_flowers_*.pt"],
                "hinge": ["fc_hinge_cub_*.pt", "fc_hinge_flowers_*.pt"],
                "euclidean": ["fc_euclidean_cub_*.pt", "fc_euclidean_flowers_*.pt"],
                # Dataset-specific BCE keys (prevent cross-dataset fold selection)
                "fc_bce_cub":         ["fc_bce_cub_fc_*.pt"],
                "conv_bce_cub":       ["conv_bce_cub_conv*.pt"],
                "fc_conv_bce_cub":    ["fc_conv_bce_cub_conv*.pt"],
                "fc_bce_flowers":     ["fc_bce_flowers_fc_*.pt"],
                "conv_bce_flowers":   ["conv_bce_flowers_conv*.pt"],
                "fc_conv_bce_flowers":["fc_conv_bce_flowers_conv*.pt"],
                # Dataset-specific hinge/euclidean keys (prevent cross-dataset selection)
                "fc_hinge_cub":            ["fc_hinge_cub_fc_*.pt"],
                "fc_euclidean_cub":        ["fc_euclidean_cub_fc_*.pt"],
                "fc_hinge_flowers":        ["fc_hinge_flowers_fc_*.pt"],
                "fc_euclidean_flowers":    ["fc_euclidean_flowers_fc_*.pt"],
                "conv_hinge_cub":          ["conv_hinge_cub_conv*.pt"],
                "conv_euclidean_cub":      ["conv_euclidean_cub_conv*.pt"],
                "conv_hinge_flowers":      ["conv_hinge_flowers_conv*.pt"],
                "conv_euclidean_flowers":  ["conv_euclidean_flowers_conv*.pt"],
                "fc_conv_hinge_cub":       ["fc_conv_hinge_cub_conv*.pt"],
                "fc_conv_euclidean_cub":   ["fc_conv_euclidean_cub_conv*.pt"],
                "fc_conv_hinge_flowers":   ["fc_conv_hinge_flowers_conv*.pt"],
                "fc_conv_euclidean_flowers":["fc_conv_euclidean_flowers_conv*.pt"],
                # per-model-type per-loss keys (cross-dataset, kept for backward compat)
                "fc_bce": ["fc_bce_cub_fc_*.pt", "fc_bce_flowers_fc_*.pt"],
                "fc_hinge": ["fc_hinge_cub_fc_*.pt", "fc_hinge_flowers_fc_*.pt"],
                "fc_euclidean": ["fc_euclidean_cub_fc_*.pt", "fc_euclidean_flowers_fc_*.pt"],
                "conv_bce": ["conv_bce_cub_conv*.pt", "conv_bce_flowers_conv*.pt"],
                "conv_hinge": ["conv_hinge_cub_conv*.pt", "conv_hinge_flowers_conv*.pt"],
                "conv_euclidean": ["conv_euclidean_cub_conv*.pt", "conv_euclidean_flowers_conv*.pt"],
                "fc_conv_bce": ["fc_conv_bce_cub_conv*.pt", "fc_conv_bce_flowers_conv*.pt"],
                "fc_conv_hinge": ["fc_conv_hinge_cub_conv*.pt", "fc_conv_hinge_flowers_conv*.pt"],
                "fc_conv_euclidean": ["fc_conv_euclidean_cub_conv*.pt", "fc_conv_euclidean_flowers_conv*.pt"],
                # specific layers (for both conv and fc+conv models)
                "conv4_3": ["conv_*_cub_conv4_3_*.pt", "conv_*_flowers_conv4_3_*.pt",
                            "fc_conv_*_cub_conv4_3_*.pt", "fc_conv_*_flowers_conv4_3_*.pt"],
                "conv5_3": ["conv_*_cub_conv5_3_*.pt", "conv_*_flowers_conv5_3_*.pt",
                            "fc_conv_*_cub_conv5_3_*.pt", "fc_conv_*_flowers_conv5_3_*.pt"],
                "pool5": ["conv_*_cub_pool5_*.pt", "conv_*_flowers_pool5_*.pt",
                          "fc_conv_*_cub_pool5_*.pt", "fc_conv_*_flowers_pool5_*.pt"],
                # CUB-only fc+conv conv5_3 (Table 3: prevents Flowers or pure-conv from being selected)
                "fc_conv_cub_conv5_3": ["fc_conv_*_cub_conv5_3_*.pt"],
            }

            # Get pattern list for this key (or use key as part of pattern)
            pattern_list = patterns.get(key)
            if pattern_list:
                # Find all matching files across all patterns
                matching_files = []
                for pattern in pattern_list:
                    matching_files.extend(base.glob(pattern))

                if matching_files:
                    # Prioritize non-50/50 files (exclude files with "_0_") for standard tables
                    # Table 4 uses 50/50 split (n_unseen=0, train_ratio=0.5) which creates filenames with "_0_"
                    standard_files = [f for f in matching_files if "_0_" not in f.name]

                    def _prefix_score(p: Path) -> int:
                        """Score a file by how long a prefix of `key` matches the start of its stem."""
                        stem = p.stem
                        for i in range(len(key), 0, -1):
                            if stem.startswith(key[:i]):
                                return i
                        return 0

                    if standard_files:
                        # Sort by longest name-prefix match (descending), then alphabetically as tiebreaker
                        standard_files.sort(key=lambda p: (-_prefix_score(p), p.name))
                        return str(standard_files[0])
                    else:
                        # Fallback to all files if no standard files found (e.g., for Table 4)
                        matching_files.sort(key=lambda p: (-_prefix_score(p), p.name))
                        return str(matching_files[0])

        # 3. Fallback to old naming scheme for backward compatibility
        path = base / CHECKPOINT_NAMES.get(key, f"{key}.pt")
        if path.exists():
            return str(path)

    # 4. Fallback: search fold{i}/ subdirectories and return the most recent match
    if dir_to_use:
        base = Path(dir_to_use)
        if not base.is_absolute():
            base = CODE_ROOT / base
        fold_dirs = sorted(base.glob("fold*/"), key=lambda p: p.name)
        for fold_dir in fold_dirs:
            p = resolve_checkpoint("", str(fold_dir), key)
            if p:
                return p

    return ""


def resolve_cv_checkpoints(
    key: str,
    n_folds: int,
    checkpoint_dir: str = "",
    explicit_paths: list[str] | None = None,
) -> list[str]:
    """Return one checkpoint path per fold from ``checkpoints/fold{i}/``.

    Args:
        key: Checkpoint key (e.g. ``"fc"``, ``"bce"``).
        n_folds: Maximum number of folds to search (0 = auto-detect from dirs).
        checkpoint_dir: Override for the checkpoints root directory.
        explicit_paths: If provided, return these directly (bypass auto-detection).

    Returns:
        List of resolved checkpoint paths.  Empty strings mark missing folds.
        May be shorter than ``n_folds`` when fewer fold directories exist.
    """
    if explicit_paths:
        return [p for p in explicit_paths if p]

    dir_to_use = checkpoint_dir if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    base = Path(dir_to_use)
    if not base.is_absolute():
        base = CODE_ROOT / base

    if not base.exists():
        return []

    # Auto-detect available fold directories
    fold_dirs = sorted(base.glob("fold*/"), key=lambda p: p.name)
    if n_folds > 0:
        fold_dirs = fold_dirs[:n_folds]

    paths = []
    for fold_dir in fold_dirs:
        p = resolve_checkpoint("", str(fold_dir), key)
        if p:  # skip folds where the checkpoint is missing
            paths.append(p)

    return paths


# Paper-style: single column ~3.3in, two-column figure ~6.6in; font ~9pt
FIG_SINGLE_COL_INCH = 3.3
FIG_TWO_COL_INCH = 6.6
FIG_DPI = 150


def get_tables_dir() -> Path:
    d = RESULTS_ROOT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_tex_dir() -> Path:
    d = RESULTS_ROOT / "tex"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir() -> Path:
    d = RESULTS_ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_table_csv(tables_dir: Path, table_id: int) -> tuple[list[str], list[list[str]]] | None:
    """Read existing table CSV file. Returns (headers, rows) or None if file doesn't exist."""
    import csv
    path = tables_dir / f"Table{table_id}.csv"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return None
        return rows[0], rows[1:]


def write_table_csv(tables_dir: Path, table_id: int, headers: list[str], rows: list[list[str]]) -> Path:
    import csv
    path = tables_dir / f"Table{table_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def write_table_latex(
    tables_dir: Path,
    table_id: int,
    caption: str,
    label: str,
    header_rows: list[list[str]],
    data_rows: list[list[str]],
    col_align: str = "l",
) -> Path:
    """Write LaTeX table (booktabs) matching paper layout. header_rows: list of rows for \\thead; data_rows: body."""
    path = tables_dir / f"Table{table_id}.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{" + caption + "}",
        "\\label{tab:" + label + "}",
        "\\begin{tabular}{" + col_align + "}",
        "\\toprule",
    ]
    for row in header_rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\midrule")
    for row in data_rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compile_table_to_pdf(tex_dir: Path, table_id: int, xelatex: str = "xelatex") -> bool:
    """
    Compile a single table LaTeX file to PDF using xelatex.
    PDF is saved to results/ root, TEX files are in tex/ subdirectory.

    Args:
        tex_dir: Directory containing Table{table_id}.tex (results/tex/)
        table_id: Table number (1, 2, 3, 4, etc.)
        xelatex: xelatex command (default: "xelatex")

    Returns:
        True if compilation succeeded, False otherwise
    """
    import subprocess

    tex_path = tex_dir / f"Table{table_id}.tex"
    if not tex_path.exists():
        return False

    name = tex_path.stem
    content = tex_path.read_text(encoding="utf-8")

    preamble = r"""
\documentclass[11pt]{article}
\usepackage{booktabs}
\usepackage{fontspec}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
"""

    footer = r"""
\end{document}
"""

    wrapper = tex_dir / f"{name}_pdf.tex"
    wrapper.write_text(preamble + content + footer, encoding="utf-8")

    # Output PDF to results root, not tex subdirectory
    results_root = tex_dir.parent
    try:
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-output-directory", str(results_root), wrapper.name],
            cwd=tex_dir,
            capture_output=True,
            timeout=60,
        )
        # Check if compilation succeeded
        if result.returncode != 0:
            print(f"XeLaTeX compilation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr.decode('utf-8', errors='ignore')}")
            if result.stdout:
                print(f"Standard output:\n{result.stdout.decode('utf-8', errors='ignore')[:500]}")
            if wrapper.exists():
                wrapper.unlink()
            return False
    except FileNotFoundError:
        print(f"XeLaTeX not found. Please install LaTeX (e.g., 'brew install mactex' on macOS)")
        if wrapper.exists():
            wrapper.unlink()
        return False
    except subprocess.TimeoutExpired:
        print(f"XeLaTeX compilation timed out after 60 seconds")
        if wrapper.exists():
            wrapper.unlink()
        return False

    # Clean up auxiliary files in results root
    for ext in (".aux", ".log"):
        p = results_root / f"{name}_pdf{ext}"
        if p.exists():
            p.unlink()

    # Rename PDF to final name
    pdf_out = results_root / f"{name}_pdf.pdf"
    target = results_root / f"{name}.pdf"
    if pdf_out.exists():
        pdf_out.rename(target)
        if wrapper.exists():
            wrapper.unlink()
        return True

    if wrapper.exists():
        wrapper.unlink()
    return False


def validate_data_path(cub_root: str, wikipedia_path: str, script_name: str = "Script") -> bool:
    """
    Validate that required data paths exist.

    Returns True if all paths exist, False otherwise.
    Prints informative error messages if paths are missing.
    """
    missing = []
    if cub_root and not Path(cub_root).exists():
        missing.append(f"CUB images directory: {cub_root}")
    if wikipedia_path and not Path(wikipedia_path).exists():
        missing.append(f"Wikipedia JSONL: {wikipedia_path}")

    if missing:
        print(f"{script_name}: Missing required data paths:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def get_device(device_arg: str) -> torch.device:
    """
    Get torch device with fallback.
    """
    import torch
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
