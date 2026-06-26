"""
check_value_coverage.py
=======================
For each run folder under Archive/{category}, plot a heatmap showing
which (Value1, Value2) pairs have debate data.

When a base run has a matching _v2 folder, generates a combined 3-panel
plot (base | v2 | combined). Standalone folders get a single heatmap.

Run:
    python3 check_value_coverage.py

Outputs heatmaps into ./plots/coverage/
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", font_scale=0.85)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = SCRIPT_DIR.parent / "Archive"

CATEGORIES = {
    "Reasoning vs Non-reasoning": ARCHIVE_ROOT / "Reasoning vs Non-reasoning",
    "Runs/Untitled": ARCHIVE_ROOT / "Runs" / "Untitled",
    "Small vs Large": ARCHIVE_ROOT / "Small vs Large",
}

OUTPUT_ROOT = SCRIPT_DIR / "plots" / "coverage"

VALUE_ORDER = [
    "Care/Protection",
    "Truthfulness",
    "Professionalism",
    "Sustainability",
    "Justice/Equality/Freedom/Respect",
    "Creativity",
    "Wisdom/Learning",
    "Communication/Cooperation",
    "Privacy",
]

SHORT_VALS = [v.split("/")[0].strip() for v in VALUE_ORDER]


def shorten_value(v: str) -> str:
    return v.split("/")[0].strip()


def is_run_folder(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        p.is_file() and p.suffix == ".json" and re.fullmatch(r"\d+\.json", p.name)
        for p in path.iterdir()
    )


def scan_run_folder(run_dir: Path) -> tuple[dict[tuple[str, str], int], int]:
    """Scan a run folder and return (pair_counts, total)."""
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    for p in run_dir.iterdir():
        if not (p.is_file() and p.suffix == ".json" and re.fullmatch(r"\d+\.json", p.name)):
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        v1 = shorten_value(str(data.get("values1", "")))
        v2 = shorten_value(str(data.get("values2", "")))
        if v1 and v2:
            pair_counts[(v1, v2)] += 1
            total += 1
    return dict(pair_counts), total


def make_matrix(pair_counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    mat = pd.DataFrame(0, index=SHORT_VALS, columns=SHORT_VALS, dtype=int)
    for (v1, v2), count in pair_counts.items():
        if v1 in SHORT_VALS and v2 in SHORT_VALS:
            mat.loc[v1, v2] = count
    return mat.iloc[::-1]  # reverse y-axis


def plot_single(run_name: str, pair_counts: dict, total: int, out_path: Path):
    mat = make_matrix(pair_counts)
    annot = mat.map(lambda v: str(v) if v > 0 else "")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, ax=ax, annot=annot, fmt="", cmap="YlOrRd",
                linewidths=0.4, linecolor="white", vmin=0,
                cbar_kws={"label": "# Debates", "shrink": 0.8})
    ax.set_title(f"{run_name}  (total: {total} debates)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Value2"); ax.set_ylabel("Value1")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined(base_name: str, base_pairs: dict, base_total: int,
                  v2_pairs: dict, v2_total: int, out_path: Path):
    # Merge counts
    combined = defaultdict(int)
    for k, v in base_pairs.items(): combined[k] += v
    for k, v in v2_pairs.items(): combined[k] += v
    combined_total = base_total + v2_total

    mats = [make_matrix(base_pairs), make_matrix(v2_pairs), make_matrix(dict(combined))]
    titles = [f"Base run  ({base_total})", f"v2 run  ({v2_total})", f"Combined  ({combined_total})"]

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, mat, title in zip(axes, mats, titles):
        annot = mat.map(lambda v: str(v) if v > 0 else "")
        sns.heatmap(mat, ax=ax, annot=annot, fmt="", cmap="YlOrRd",
                    linewidths=0.4, linecolor="white", vmin=0,
                    cbar_kws={"label": "# Debates", "shrink": 0.8})
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value2"); ax.set_ylabel("Value1")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(f"{base_name}  —  Value-pair Coverage", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    # Clean old coverage plots
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        print(f"Cleaned: {OUTPUT_ROOT}")

    for category_name, category_root in CATEGORIES.items():
        if not category_root.exists():
            print(f"[SKIP] Not found: {category_root}")
            continue

        safe_cat = category_name.replace("/", "_").replace(" ", "_")
        cat_out_dir = OUTPUT_ROOT / safe_cat

        print(f"\n{'='*60}")
        print(f"Category: {category_name}")
        print(f"{'='*60}")

        run_dirs = sorted(
            [p for p in category_root.iterdir() if is_run_folder(p)],
            key=lambda p: p.name.lower(),
        )
        if not run_dirs:
            print("  No run folders found."); continue

        # Group into base + v2 pairs
        by_name: dict[str, dict[str, Path]] = {}
        for d in run_dirs:
            name = d.name
            if name.endswith("_v2"):
                base = name[:-3]
                by_name.setdefault(base, {})["v2"] = d
            elif name.endswith("_v3"):
                # skip v3 for now
                continue
            else:
                by_name.setdefault(name, {})["base"] = d

        for base_name, variants in sorted(by_name.items()):
            base_dir = variants.get("base")
            v2_dir = variants.get("v2")

            if base_dir and v2_dir:
                bp, bt = scan_run_folder(base_dir)
                vp, vt = scan_run_folder(v2_dir)
                if bt == 0 and vt == 0:
                    print(f"  [SKIP] {base_name}: no data"); continue
                out = cat_out_dir / f"{base_name}_combined.png"
                plot_combined(base_name, bp, bt, vp, vt, out)
            elif base_dir:
                bp, bt = scan_run_folder(base_dir)
                if bt == 0:
                    print(f"  [SKIP] {base_name}: no data"); continue
                out = cat_out_dir / f"{base_name}.png"
                plot_single(base_name, bp, bt, out)
            elif v2_dir:
                vp, vt = scan_run_folder(v2_dir)
                if vt == 0:
                    print(f"  [SKIP] {base_name}_v2: no data"); continue
                out = cat_out_dir / f"{base_name}_v2.png"
                plot_single(f"{base_name}_v2", vp, vt, out)

    print("\nDone!")


if __name__ == "__main__":
    main()
