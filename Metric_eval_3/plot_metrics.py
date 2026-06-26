"""
plot_metrics.py
===============
Comprehensive visualization script for all metrics in:
  - Reasoning vs Non-reasoning/metrics.json
  - Runs/Untitled/metrics.json
  - Small vs Large/metrics.json

Run:
    python plot_metrics.py

Outputs one sub-folder per category inside ./plots/
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

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

VALUE_SHORT = {v: v.split("/")[0].strip() for v in VALUE_ORDER}

CATEGORIES = {
    "Reasoning vs Non-reasoning": Path("Reasoning vs Non-reasoning/metrics.json"),
    "Runs/Untitled": Path("Runs/Untitled/metrics.json"),
    "Small vs Large": Path("Small vs Large/metrics.json"),
}

OUTPUT_ROOT = Path("plots")

CMAP_HEATMAP = "Blues"
CMAP_DIVERGING = "RdYlGn"
CMAP_COOL = "YlOrRd"

sns.set_theme(style="white", font_scale=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shorten(name: str) -> str:
    """Shorten a model name to something readable."""
    # e.g. google/gemma-3-12b-it  -> gemma-3-12b-it
    if "/" in name:
        name = name.split("/")[-1]
    return name


def shorten_value(v: str) -> str:
    return VALUE_SHORT.get(v, v.split("/")[0].strip())


def finite(x):
    """Return x if finite number, else None."""
    if x is None:
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
    except TypeError:
        return None
    return x


def save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def pivot_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    val_col: str,
    aggfunc="mean",
) -> pd.DataFrame:
    piv = df.pivot_table(index=row_col, columns=col_col, values=val_col, aggfunc=aggfunc)
    return piv


def annotated_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    fmt: str = ".2f",
    cmap=CMAP_HEATMAP,
    vmin=None,
    vmax=None,
    cbar_label: str = "",
    mask: pd.DataFrame | None = None,
):
    _applymap = getattr(data, "map", None) or data.applymap
    annot = _applymap(lambda v: f"{v:{fmt}}" if pd.notna(v) else "")
    sns.heatmap(
        data,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        mask=mask,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Persuasion Drift heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_persuasion_drift(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    df = df[df["Persuasion_Drift"].apply(lambda x: finite(x) is not None)].copy()
    if df.empty:
        print("  [SKIP] No Persuasion_Drift data.")
        return

    df["model_pair"] = df.apply(
        lambda r: f"{shorten(r['Model1'])} vs {shorten(r['Model2'])}", axis=1
    )
    df["v1_short"] = df["Value1"].map(shorten_value)
    df["v2_short"] = df["Value2"].map(shorten_value)

    pairs = sorted(df["model_pair"].unique())
    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)

    short_vals = [shorten_value(v) for v in VALUE_ORDER]

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if len(pairs) > 1 else [axes]

    for i, pair in enumerate(pairs):
        ax = axes[i]
        sub = df[df["model_pair"] == pair]
        mat = pd.DataFrame(np.nan, index=short_vals, columns=short_vals)
        for _, row in sub.iterrows():
            r, c = row["v1_short"], row["v2_short"]
            if r in mat.index and c in mat.columns:
                mat.loc[r, c] = row["Persuasion_Drift"]
        annotated_heatmap(
            ax, mat, fmt=".2f", cmap=CMAP_DIVERGING,
            vmin=-1, vmax=1, cbar_label="Persuasion Drift"
        )
        ax.set_title(pair, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value2")
        ax.set_ylabel("Value1")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Persuasion Drift — {category}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out_dir / "1_persuasion_drift.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Token Cost
# ─────────────────────────────────────────────────────────────────────────────

def plot_token_cost(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)

    # melt to long form: one row per (model, value, token_cost)
    rows = []
    for _, r in df.iterrows():
        tc1 = finite(r.get("TokenCost_Model1"))
        tc2 = finite(r.get("TokenCost_Model2"))
        if tc1 is not None:
            rows.append({"model": shorten(r["Model1"]), "value": shorten_value(r["Value1"]), "tokens": tc1})
        if tc2 is not None:
            rows.append({"model": shorten(r["Model2"]), "value": shorten_value(r["Value2"]), "tokens": tc2})

    if not rows:
        print("  [SKIP] No TokenCost data.")
        return

    long = pd.DataFrame(rows)
    models = sorted(long["model"].unique())
    values = [shorten_value(v) for v in VALUE_ORDER if shorten_value(v) in long["value"].unique()]

    # ── 2a. For each model: bar graph (value vs tokens) – all models in one image
    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = long[long["model"] == model].groupby("value")["tokens"].mean().reindex(values)
        ax.bar(sub.index, sub.values, color=sns.color_palette("Blues_d", len(sub)))
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Avg Tokens")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Token Cost (per model, by value) — {category}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir / "2a_token_cost_model_bar.png")

    # ── 2b. For each model: heatmap (values × values) cell = token cost
    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]

    all_vals_short = [shorten_value(v) for v in VALUE_ORDER]
    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = long[long["model"] == model]
        # For heatmap: x = Value2 context, y = Value1 context
        # We keep the original value columns to pivot
        df_m = df[
            (df["Model1"].apply(shorten) == model) | (df["Model2"].apply(shorten) == model)
        ].copy()
        mat = pd.DataFrame(np.nan, index=all_vals_short, columns=all_vals_short)
        for _, row in df_m.iterrows():
            v1 = shorten_value(row["Value1"])
            v2 = shorten_value(row["Value2"])
            if shorten(row["Model1"]) == model and finite(row.get("TokenCost_Model1")) is not None:
                mat.loc[v1, v2] = row["TokenCost_Model1"]
            if shorten(row["Model2"]) == model and finite(row.get("TokenCost_Model2")) is not None:
                mat.loc[v2, v1] = row["TokenCost_Model2"]
        annotated_heatmap(ax, mat, fmt=".0f", cmap=CMAP_COOL, cbar_label="Tokens")
        ax.set_title(model, fontsize=8, fontweight="bold")
        ax.set_xlabel("Value2")
        ax.set_ylabel("Value1")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Token Cost Heatmap (by model, values × values) — {category}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir / "2b_token_cost_model_heatmap.png")

    # ── 2c. For each value: bar graph (models vs tokens) – all values in one image
    vals_present = [v for v in values if v in long["value"].unique()]
    ncols = min(3, len(vals_present))
    nrows = math.ceil(len(vals_present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if len(vals_present) > 1 else [axes]

    for i, val in enumerate(vals_present):
        ax = axes_flat[i]
        sub = long[long["value"] == val].groupby("model")["tokens"].mean()
        ax.bar(sub.index, sub.values, color=sns.color_palette("Oranges_d", len(sub)))
        ax.set_title(val, fontsize=9, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Avg Tokens")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Token Cost (per value, by model) — {category}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir / "2c_token_cost_value_bar.png")

    # ── 2d. For each model pair: heatmap (models × models) cell = token cost
    df2 = df.copy()
    df2["m1s"] = df2["Model1"].apply(shorten)
    df2["m2s"] = df2["Model2"].apply(shorten)
    model_list = sorted(set(df2["m1s"].tolist() + df2["m2s"].tolist()))

    mat = pd.DataFrame(np.nan, index=model_list, columns=model_list)
    for _, row in df2.iterrows():
        tc1 = finite(row.get("TokenCost_Model1"))
        tc2 = finite(row.get("TokenCost_Model2"))
        if tc1 is not None:
            mat.loc[row["m1s"], row["m2s"]] = tc1
        if tc2 is not None:
            mat.loc[row["m2s"], row["m1s"]] = tc2

    fig, ax = plt.subplots(figsize=(max(6, len(model_list)), max(5, len(model_list))))
    annotated_heatmap(ax, mat, fmt=".0f", cmap=CMAP_COOL, cbar_label="Avg Tokens")
    ax.set_title(f"Token Cost Heatmap (models × models) — {category}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Opponent Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    save(fig, out_dir / "2d_token_cost_model_pair_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Avg Turns to Consensus
# ─────────────────────────────────────────────────────────────────────────────

def plot_avg_turns(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    df = df[df["Avg_Turns_To_Consensus"].apply(lambda x: finite(x) is not None)].copy()
    if df.empty:
        print("  [SKIP] No Avg_Turns_To_Consensus data.")
        return

    df["model_pair"] = df.apply(
        lambda r: f"{shorten(r['Model1'])} vs {shorten(r['Model2'])}", axis=1
    )
    df["v1_short"] = df["Value1"].map(shorten_value)
    df["v2_short"] = df["Value2"].map(shorten_value)

    pairs = sorted(df["model_pair"].unique())
    short_vals = [shorten_value(v) for v in VALUE_ORDER]

    # ── 3a. Bar: for each model pair, values vs turns
    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if len(pairs) > 1 else [axes]

    for i, pair in enumerate(pairs):
        ax = axes_flat[i]
        sub = df[df["model_pair"] == pair]
        # aggregate by value1 and value2
        vals_turns = defaultdict(list)
        for _, row in sub.iterrows():
            vals_turns[row["v1_short"]].append(row["Avg_Turns_To_Consensus"])
            vals_turns[row["v2_short"]].append(row["Avg_Turns_To_Consensus"])
        means = {v: np.mean(vals_turns[v]) for v in short_vals if v in vals_turns}
        ax.bar(list(means.keys()), list(means.values()), color=sns.color_palette("Greens_d", len(means)))
        ax.set_title(pair, fontsize=8, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Avg Turns to Consensus")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Avg Turns to Consensus (bar, by model pair) — {category}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir / "3a_avg_turns_bar.png")

    # ── 3b. Heatmap: for each model pair, values × values
    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes_flat = np.array(axes).flatten() if len(pairs) > 1 else [axes]

    for i, pair in enumerate(pairs):
        ax = axes_flat[i]
        sub = df[df["model_pair"] == pair]
        mat = pd.DataFrame(np.nan, index=short_vals, columns=short_vals)
        for _, row in sub.iterrows():
            mat.loc[row["v1_short"], row["v2_short"]] = row["Avg_Turns_To_Consensus"]

        # Symmetrize matrix if the models are identical
        if pair.split(" vs ")[0] == pair.split(" vs ")[1]:
            with np.errstate(all="ignore"):
                sym_arr = np.nanmean([mat.values, mat.T.values], axis=0)
            mat = pd.DataFrame(sym_arr, index=mat.index, columns=mat.columns)

        annotated_heatmap(ax, mat, fmt=".1f", cmap="YlGn", cbar_label="Avg Turns")
        ax.set_title(pair, fontsize=8, fontweight="bold")
        ax.set_xlabel("Value2")
        ax.set_ylabel("Value1")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Avg Turns to Consensus (heatmap) — {category}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir / "3b_avg_turns_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Argument Result
# ─────────────────────────────────────────────────────────────────────────────

def plot_argument_result(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    df["model_pair"] = df.apply(
        lambda r: f"{shorten(r['Model1'])} vs {shorten(r['Model2'])}", axis=1
    )
    df["v1_short"] = df["Value1"].map(shorten_value)
    df["v2_short"] = df["Value2"].map(shorten_value)

    pairs = sorted(df["model_pair"].unique())
    short_vals = [shorten_value(v) for v in VALUE_ORDER]

    for suffix, col in [("model1", "ArgumentResult_Model1"), ("model2", "ArgumentResult_Model2")]:
        sub_df = df[df[col].apply(lambda x: finite(x) is not None)].copy()
        if sub_df.empty:
            continue

        ncols = min(3, len(pairs))
        nrows = math.ceil(len(pairs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes_flat = np.array(axes).flatten() if len(pairs) > 1 else [axes]

        for i, pair in enumerate(pairs):
            ax = axes_flat[i]
            sub = sub_df[sub_df["model_pair"] == pair]
            mat = pd.DataFrame(np.nan, index=short_vals, columns=short_vals)
            for _, row in sub.iterrows():
                if finite(row[col]) is not None:
                    mat.loc[row["v1_short"], row["v2_short"]] = row[col]
            
            # Symmetrize matrix if the models are identical
            if pair.split(" vs ")[0] == pair.split(" vs ")[1]:
                with np.errstate(all="ignore"):
                    sym_arr = np.nanmean([mat.values, mat.T.values], axis=0)
                mat = pd.DataFrame(sym_arr, index=mat.index, columns=mat.columns)

            annotated_heatmap(ax, mat, fmt=".2f", cmap=CMAP_DIVERGING, cbar_label="Arg Result")
            ax.set_title(pair, fontsize=8, fontweight="bold")
            ax.set_xlabel("Value2")
            ax.set_ylabel("Value1")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Argument Result ({col}) — {category}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, out_dir / f"4_argument_result_{suffix}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Argument Strength
# ─────────────────────────────────────────────────────────────────────────────

def plot_argument_strength(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)

    rows = []
    for _, r in df.iterrows():
        s1 = finite(r.get("ArgumentStrength_Model1"))
        s2 = finite(r.get("ArgumentStrength_Model2"))
        if s1 is not None:
            rows.append({"model": shorten(r["Model1"]), "value": shorten_value(r["Value1"]), "strength": s1})
        if s2 is not None:
            rows.append({"model": shorten(r["Model2"]), "value": shorten_value(r["Value2"]), "strength": s2})

    if not rows:
        print("  [SKIP] No ArgumentStrength data.")
        return

    long = pd.DataFrame(rows)
    models = sorted(long["model"].unique())
    short_vals = [shorten_value(v) for v in VALUE_ORDER if shorten_value(v) in long["value"].unique()]

    mat = long.pivot_table(index="model", columns="value", values="strength", aggfunc="mean")
    mat = mat.reindex(columns=[v for v in [shorten_value(x) for x in VALUE_ORDER] if v in mat.columns])

    fig, ax = plt.subplots(figsize=(max(10, len(short_vals)), max(5, len(models))))
    annotated_heatmap(ax, mat, fmt=".2f", cmap="PuBuGn", cbar_label="Mean Argument Strength")
    ax.set_title(f"Argument Strength (model × value) — {category}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Model")
    fig.tight_layout()
    save(fig, out_dir / "5_argument_strength_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Persuasion Bias
#
#  View A: Fixed-size pie chart matrix  (one figure per model pair)
#    Grid: rows = Value1, cols = Value2
#    Each cell = pie with 3 slices:
#      red   = Consensus_Wins_Model1 / Consensus_Debates   (model1 wins)
#      blue  = Consensus_Wins_Model2 / Consensus_Debates   (model2 wins)
#      green = remainder                                    (no consensus winner)
#    All pies same size → colour area is the signal, not radius.
#
#  View B: Per-value aggregated win-count bar chart  (one chart, all pairs)
#    For each (model_pair, value) aggregate:
#      model1_wins  = sum of Consensus_Wins_Model1  for rows where value == Value1
#      model2_wins  = sum of Consensus_Wins_Model2  for rows where value == Value2
#      total_consensus = sum of Consensus_Debates   (same rows)
#    win_rate = wins / total_consensus  → always in [0, 1]
#    Justification: win counts are integers and sum correctly; averaging rates
#    across rows of different sizes was causing the >100% bug.
# ─────────────────────────────────────────────────────────────────────────────

def plot_persuasion_bias(records: list[dict], out_dir: Path, category: str):
    import matplotlib.patches as mpatches

    df = pd.DataFrame(records)
    df = df[
        df["Consensus_WinRate_Model1"].apply(lambda x: finite(x) is not None) |
        df["Consensus_WinRate_Model2"].apply(lambda x: finite(x) is not None)
    ].copy()

    if df.empty:
        print("  [SKIP] No Consensus_WinRate data.")
        return

    df["v1_short"]   = df["Value1"].map(shorten_value)
    df["v2_short"]   = df["Value2"].map(shorten_value)
    df["m1_short"]   = df["Model1"].apply(shorten)
    df["m2_short"]   = df["Model2"].apply(shorten)
    df["model_pair"] = df["m1_short"] + " vs " + df["m2_short"]

    # Use raw counts — always integers, never sum above total
    df["wins1"]      = df["Consensus_Wins_Model1"].fillna(0).astype(int)
    df["wins2"]      = df["Consensus_Wins_Model2"].fillna(0).astype(int)
    df["con_deb"]    = df["Consensus_Debates"].fillna(0).astype(int)

    short_vals = [shorten_value(v) for v in VALUE_ORDER]
    COLOR_M1   = "#c0392b"   # red   — model1
    COLOR_M2   = "#2c5f9e"   # blue  — model2
    COLOR_NONE = "#5dba6e"   # green — no consensus winner

    pairs = sorted(df["model_pair"].unique())

    # =========================================================================
    # VIEW A — Fixed-size pie matrix, one figure per model pair
    # =========================================================================
    for pair in pairs:
        sub = df[df["model_pair"] == pair].copy()
        if sub.empty:
            continue

        vals_present = [v for v in short_vals
                        if v in sub["v1_short"].values or v in sub["v2_short"].values]
        n = len(vals_present)
        if n == 0:
            continue

        val_idx = {v: i for i, v in enumerate(vals_present)}

        cell = 1.0          # grid unit
        pad  = 0.08
        figsize = (n * cell + 2.5, n * cell + 2.0)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_aspect("equal")
        ax.axis("off")

        pie_r = 0.38   # fixed radius in data units

        # Column / row header labels
        for i, v in enumerate(vals_present):
            # x-axis labels (col = Value2)
            ax.text(i, -0.55, v, ha="center", va="top",
                    fontsize=6.5, rotation=40, rotation_mode="anchor", color="#333")
            # y-axis labels (row = Value1)
            ax.text(-0.52, i, v, ha="right", va="center",
                    fontsize=6.5, color="#333")

        for _, row in sub.iterrows():
            v1 = row["v1_short"]
            v2 = row["v2_short"]
            if v1 not in val_idx or v2 not in val_idx:
                continue

            ri = val_idx[v1]   # row
            ci = val_idx[v2]   # col

            w1  = int(row["wins1"])
            w2  = int(row["wins2"])
            tot = int(row["con_deb"])
            if tot == 0:
                continue

            no_win = max(0, tot - w1 - w2)
            sizes  = [w1, w2, no_win]
            colors = [COLOR_M1, COLOR_M2, COLOR_NONE]

            # Draw pie via wedge patches at (ci, ri)
            start = 90.0  # start from top
            for size, color in zip(sizes, colors):
                if size == 0:
                    continue
                frac  = size / tot
                angle = frac * 360.0
                wedge = mpatches.Wedge(
                    center=(ci, ri),
                    r=pie_r,
                    theta1=start - angle,
                    theta2=start,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                ax.add_patch(wedge)

                # Label inside wedge if large enough
                if frac >= 0.12:
                    mid_angle = math.radians(start - angle / 2)
                    lx = ci + pie_r * 0.58 * math.cos(mid_angle)
                    ly = ri + pie_r * 0.58 * math.sin(mid_angle)
                    ax.text(lx, ly, f"{frac:.0%}",
                            ha="center", va="center",
                            fontsize=4.5, color="white",
                            fontweight="bold", zorder=4)
                start -= angle

            # Thin circle border
            circle = plt.Circle((ci, ri), pie_r,
                                 fill=False, edgecolor="#aaaaaa",
                                 linewidth=0.5, zorder=5)
            ax.add_patch(circle)

            # Debate count below pie
            ax.text(ci, ri - pie_r - 0.06, f"n={tot}",
                    ha="center", va="top", fontsize=4, color="#777", zorder=4)

        m1_name, m2_name = pair.split(" vs ", 1)
        legend_handles = [
            mpatches.Patch(color=COLOR_M1,   label=f"Model 1 wins  ({m1_name})"),
            mpatches.Patch(color=COLOR_M2,   label=f"Model 2 wins  ({m2_name})"),
            mpatches.Patch(color=COLOR_NONE, label="No consensus winner"),
        ]
        ax.legend(handles=legend_handles,
                  loc="lower center",
                  bbox_to_anchor=(0.5, -0.12),
                  ncol=3, fontsize=7.5, frameon=True,
                  bbox_transform=ax.transAxes)

        ax.set_title(
            f"Persuasion Bias — {pair} — {category}\n"
            f"Row = Value1 ({m1_name})  ·  Col = Value2 ({m2_name})  ·  "
            f"Pie = win share",
            fontsize=9, fontweight="bold",
            x=0.5, y=1.02, transform=ax.transAxes,
        )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        safe_pair = pair.replace("/", "_").replace(" ", "_")
        save(fig, out_dir / f"6a_persuasion_bias_matrix_{safe_pair}.png")

    # =========================================================================
    # VIEW B — Per-value aggregated win rate, compact grouped bar chart
    #
    # For each (model_pair, value):
    #   model1_total_wins     = SUM(Consensus_Wins_Model1)  where Value1 == value
    #   model1_total_debates  = SUM(Consensus_Debates)       same rows
    #   model1_win_rate       = model1_total_wins / model1_total_debates
    # Same for model2 / Value2.
    # This uses counts as denominator → always in [0,1].
    # =========================================================================
    agg_rows = []
    for _, row in df.iterrows():
        pair = row["model_pair"]
        tot  = int(row["con_deb"])
        if tot == 0:
            continue
        agg_rows.append({
            "pair":  pair,
            "value": row["v1_short"],
            "slot":  "M1",
            "wins":  int(row["wins1"]),
            "total": tot,
        })
        agg_rows.append({
            "pair":  pair,
            "value": row["v2_short"],
            "slot":  "M2",
            "wins":  int(row["wins2"]),
            "total": tot,
        })

    if not agg_rows:
        return

    agg_df = pd.DataFrame(agg_rows)
    agg = (agg_df.groupby(["pair", "value", "slot"])
                 .agg(wins=("wins", "sum"), total=("total", "sum"))
                 .reset_index())
    agg["wr"] = agg["wins"] / agg["total"]   # guaranteed [0, 1]

    vals_in = [v for v in short_vals if v in agg["value"].values]
    if not vals_in:
        return

    n_vals  = len(vals_in)
    n_pairs = len(pairs)

    # Compact layout: one cluster per value, two sub-bars per pair (M1, M2)
    slots_per_cluster = n_pairs * 2
    bar_w   = 0.7 / slots_per_cluster
    x       = np.arange(n_vals)

    pair_colors_m1 = sns.color_palette("Reds_d",  n_pairs)
    pair_colors_m2 = sns.color_palette("Blues_d", n_pairs)

    fig_w = max(9, n_vals * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    for pi, pair in enumerate(pairs):
        sub_p  = agg[agg["pair"] == pair]
        m1_name, m2_name = pair.split(" vs ", 1)

        # Offset: pairs centred within cluster, M1 left of M2
        centre_offset = (pi - (n_pairs - 1) / 2) * (bar_w * 2.1)
        off_m1 = centre_offset - bar_w * 0.55
        off_m2 = centre_offset + bar_w * 0.55

        def get_wr(val, slot):
            row = sub_p[(sub_p["value"] == val) & (sub_p["slot"] == slot)]
            return float(row["wr"].iloc[0]) if len(row) > 0 else np.nan

        m1_vals = [get_wr(v, "M1") for v in vals_in]
        m2_vals = [get_wr(v, "M2") for v in vals_in]

        b1 = ax.bar(x + off_m1, m1_vals, width=bar_w,
                    color=pair_colors_m1[pi],
                    label=f"M1: {m1_name}", zorder=3)
        b2 = ax.bar(x + off_m2, m2_vals, width=bar_w,
                    color=pair_colors_m2[pi],
                    label=f"M2: {m2_name}", zorder=3)

        for bar, v in list(zip(b1, m1_vals)) + list(zip(b2, m2_vals)):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.0%}",
                        ha="center", va="bottom",
                        fontsize=6, color="#333", zorder=4)

    ax.axhline(0.5, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.55, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(vals_in, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Win rate (wins / consensus debates)", fontsize=9)
    ax.set_xlabel("Value", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Persuasion Bias — Aggregated Win Rate per Value — {category}\n"
        "Red = Model 1 when arguing that value  ·  Blue = Model 2  ·  "
        "Rate = total wins ÷ total consensus debates",
        fontsize=9, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=6.5,
              ncol=min(4, n_pairs * 2),
              framealpha=0.9,
              title="pair · model slot", title_fontsize=6.5)

    fig.tight_layout()
    save(fig, out_dir / "6b_persuasion_bias_value_aggregated.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Value Dominance Index (VDI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vdi(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    short_vals = [shorten_value(v) for v in VALUE_ORDER]

    wins: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        v1 = shorten_value(row["Value1"])
        v2 = shorten_value(row["Value2"])

        debates = int(row.get("Debates") or 0)
        cw1 = int(row.get("Consensus_Wins_Model1") or 0)
        cw2 = int(row.get("Consensus_Wins_Model2") or 0)

        # Action1 wins → value1 wins; Action2 → value2 wins
        totals[v1] += debates
        totals[v2] += debates
        wins[v1] += cw1
        wins[v2] += cw2

    vdi: dict[str, float] = {}
    for v in short_vals:
        t = totals.get(v, 0)
        if t > 0:
            vdi[v] = wins.get(v, 0) / t
        else:
            vdi[v] = np.nan

    # Heatmap: VDI as value × value matrix
    mat = pd.DataFrame(np.nan, index=short_vals, columns=short_vals)
    for _, row in df.iterrows():
        v1 = shorten_value(row["Value1"])
        v2 = shorten_value(row["Value2"])
        d = int(row.get("Debates") or 0)
        if d == 0:
            continue
        cw1 = int(row.get("Consensus_Wins_Model1") or 0)
        cw2 = int(row.get("Consensus_Wins_Model2") or 0)
        if v1 in short_vals and v2 in short_vals:
            mat.loc[v1, v2] = cw1 / d
            mat.loc[v2, v1] = cw2 / d

    fig, ax = plt.subplots(figsize=(11, 9))
    annotated_heatmap(ax, mat, fmt=".2f", cmap=CMAP_DIVERGING, vmin=0, vmax=1, cbar_label="VDI (win rate)")
    ax.set_title(f"Value Dominance Index (VDI) — {category}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Value2 (opponent)")
    ax.set_ylabel("Value1")
    fig.tight_layout()
    save(fig, out_dir / "7_vdi_heatmap.png")

    # Also bar chart of overall VDI per value
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    vdi_vals = [vdi.get(v, np.nan) for v in short_vals]
    colors = ["#4878CF" if not math.isnan(v) else "#cccccc" for v in vdi_vals]
    bars = ax2.bar(short_vals, [0 if math.isnan(v) else v for v in vdi_vals], color=colors)
    ax2.set_ylim(0, max((v for v in vdi_vals if not math.isnan(v)), default=1) * 1.2)
    ax2.set_xlabel("Value")
    ax2.set_ylabel("VDI (overall win rate)")
    ax2.set_title(f"Value Dominance Index (overall) — {category}", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=40, labelsize=8)
    for bar, v in zip(bars, vdi_vals):
        if not math.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    save(fig2, out_dir / "7b_vdi_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Morality Score (lower-triangular heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_morality_score(records: list[dict], out_dir: Path, category: str):
    """
    Morality score = % of debates where the chosen action is unchanged (consistent).
    We approximate here by using Consensus_Rate as a proxy for consistency.
    Proper morality score requires base vs v2 comparison (see morality_score.py).
    We compute it from the records: for records with Version='none' and v2 pair,
    we check win-rate stability. As a fallback, we use Consensus_Rate.
    """
    df = pd.DataFrame(records)
    short_vals = [shorten_value(v) for v in VALUE_ORDER]

    # Use consensus_rate as a stable-decision proxy
    mat = pd.DataFrame(np.nan, index=short_vals, columns=short_vals)
    counts = pd.DataFrame(0, index=short_vals, columns=short_vals, dtype=int)

    for _, row in df.iterrows():
        v1 = shorten_value(row["Value1"])
        v2 = shorten_value(row["Value2"])
        cr = finite(row.get("Consensus_Rate"))
        if cr is None:
            continue
        if v1 not in short_vals or v2 not in short_vals:
            continue

        # Force lower triangle
        i1, i2 = short_vals.index(v1), short_vals.index(v2)
        if i1 < i2:
            r, c = v2, v1
        else:
            r, c = v1, v2

        prev = mat.loc[r, c]
        n = counts.loc[r, c]
        if math.isnan(prev):
            mat.loc[r, c] = cr * 100
        else:
            mat.loc[r, c] = (prev * n + cr * 100) / (n + 1)
        counts.loc[r, c] += 1

    # Upper triangle mask
    mask = pd.DataFrame(False, index=short_vals, columns=short_vals)
    for i, v1 in enumerate(short_vals):
        for j, v2 in enumerate(short_vals):
            if j >= i:
                mask.loc[v1, v2] = True

    _applymap2 = getattr(mat, "map", None) or mat.applymap
    annot = _applymap2(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        mat,
        ax=ax,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Morality Score (%) [consensus rate proxy]", "shrink": 0.8},
    )
    ax.set_title(f"Morality Score (lower-triangular) — {category}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Value2")
    ax.set_ylabel("Value1")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    save(fig, out_dir / "8_morality_score.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for category, metrics_path in CATEGORIES.items():
        safe_name = category.replace("/", "_").replace(" ", "_")
        out_dir = OUTPUT_ROOT / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"  Input : {metrics_path}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        records = load_metrics(metrics_path)
        if not records:
            print("  [SKIP] Empty or missing metrics.")
            continue

        print(f"  Loaded {len(records)} records.")

        plot_persuasion_drift(records, out_dir, category)
        plot_token_cost(records, out_dir, category)
        plot_avg_turns(records, out_dir, category)
        plot_argument_result(records, out_dir, category)
        plot_argument_strength(records, out_dir, category)
        plot_persuasion_bias(records, out_dir, category)
        plot_vdi(records, out_dir, category)
        plot_morality_score(records, out_dir, category)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
