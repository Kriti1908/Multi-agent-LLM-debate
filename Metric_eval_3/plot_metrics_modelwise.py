
from __future__ import annotations

import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

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
    "Reasoning vs Non-reasoning": Path("Reasoning vs Non-reasoning/metrics_new.json"),
    "Runs/Untitled": Path("Runs/Untitled/metrics_new.json"),
    "Small vs Large": Path("Small vs Large/metrics_new.json"),
}

OUTPUT_ROOT = Path("valueCollapsedPlots")

sns.set_theme(style="white", font_scale=0.9)

_RE_REASONING_VARIANT = re.compile(r"(?i)(?:[_-]?(low|high))$")
_RE_NUMERIC_SIZE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*([bm])")
_RE_LEADING_FAMILY = re.compile(r"(?i)^([a-z]+)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shorten(name: Any) -> str:
    """Shorten a model name to something readable."""
    if name is None:
        return ""
    s = str(name)
    if "/" in s:
        s = s.split("/")[-1]
    return s


def shorten_value(v: Any) -> str:
    return VALUE_SHORT.get(str(v), str(v).split("/")[0].strip())


def finite(x: Any):
    """Return x if finite number, else None."""
    if x is None:
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
    except TypeError:
        return None
    return float(x)


def save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def weight_for_row(row: pd.Series) -> float:
    """Prefer consensus debates, then debates, else 1."""
    for key in ("Consensus_Debates", "Debates"):
        w = finite(row.get(key))
        if w is not None and w > 0:
            return float(w)
    return 1.0


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals: list[float] = []
    wts: list[float] = []
    for v, w in zip(values, weights):
        if v is None or w is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(w, float) and math.isnan(w):
            continue
        if w <= 0:
            continue
        vals.append(float(v))
        wts.append(float(w))
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=wts))


def model_short_display(model: str) -> str:
    """Nice label for the legend when a family key is lowercase."""
    if not model:
        return model
    if model.isupper():
        return model
    if len(model) <= 3:
        return model.upper()
    return model[:1].upper() + model[1:]


def strip_reasoning_variant(name: str) -> str:
    """Remove trailing low/high setting tags."""
    return _RE_REASONING_VARIANT.sub("", name)


def model_family(model: Any, category: str) -> str:
    """
    Family key used for color grouping.

    - Reasoning vs Non-reasoning:
      Qwen3-32B_low / Qwen3-32B_high -> Qwen3-32B
    - Small vs Large:
      Qwen3-8B / Qwen3-32B -> qwen
    - Otherwise:
      family = shortened model name (after stripping low/high if present)
    """
    s = shorten(model)

    if category == "Reasoning vs Non-reasoning":
        return strip_reasoning_variant(s)

    if category == "Small vs Large":
        base = strip_reasoning_variant(s)
        m = _RE_LEADING_FAMILY.match(base)
        if m:
            return m.group(1).lower()
        return base.lower()

    return strip_reasoning_variant(s)


def model_sort_rank(model: Any, category: str) -> tuple:
    """
    Sort models within each family.

    - Reasoning vs Non-reasoning: low -> base -> high
    - Small vs Large: ascending size (e.g., 8B before 32B)
    - Default: alphabetical
    """
    s = shorten(model)
    s_low = s.lower()

    if category == "Reasoning vs Non-reasoning":
        if re.search(r"(?i)(?:[_-]low)$", s_low):
            setting_rank = 0
        elif re.search(r"(?i)(?:[_-]high)$", s_low):
            setting_rank = 2
        else:
            setting_rank = 1
        return (setting_rank, s_low)

    if category == "Small vs Large":
        m = _RE_NUMERIC_SIZE.search(s_low)
        if m:
            size = float(m.group(1))
            unit = m.group(2).lower()
            # Normalize to billions so 8B < 20B < 32B < 70B and 125M < 1B.
            size_b = size if unit == "b" else size / 1000.0
        else:
            size_b = float("inf")
        return (size_b, s_low)

    return (s_low,)


def family_order_and_colors(models: list[str], category: str):
    families: list[str] = []
    for m in models:
        fam = model_family(m, category)
        if fam not in families:
            families.append(fam)

    palette_name = "tab20" if len(families) <= 20 else "husl"
    palette = sns.color_palette(palette_name, len(families))
    fam_color = dict(zip(families, palette))
    fam_label = {fam: model_short_display(fam) for fam in families}
    return families, fam_color, fam_label


def plot_modelwise_bar_line(
    data: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    category: str,
    percent: bool = False,
    zero_line: bool = False,
    annotate: bool = True,
):
    if data.empty:
        print(f"  [SKIP] No data for {out_path.name}")
        return

    plot_df = data.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if plot_df.empty:
        print(f"  [SKIP] No finite data for {out_path.name}")
        return

    plot_df["family"] = plot_df["model"].apply(lambda m: model_family(m, category))
    plot_df["sort_key"] = plot_df["model"].apply(lambda m: model_sort_rank(m, category))

    family_order, fam_color, fam_label = family_order_and_colors(plot_df["model"].tolist(), category)

    ordered_rows = []
    for fam in family_order:
        fam_df = plot_df[plot_df["family"] == fam].sort_values(["sort_key", "model"], ascending=True)
        ordered_rows.append(fam_df)

    plot_df = pd.concat(ordered_rows, ignore_index=True)

    positions: list[float] = []
    labels: list[str] = []
    colors: list[Any] = []
    families: list[str] = []
    values: list[float] = []

    x = 0.0
    family_gap = 0.9
    for idx, fam in enumerate(family_order):
        fam_df = plot_df[plot_df["family"] == fam].reset_index(drop=True)
        if fam_df.empty:
            continue
        if positions:
            x += family_gap
        start = x
        for _, row in fam_df.iterrows():
            positions.append(x)
            labels.append(row["model"])
            colors.append(fam_color[fam])
            families.append(fam)
            values.append(float(row["value"]))
            x += 1.0
        end = x - 1.0
        # store center if we want to label groups later
        # (kept for future use)
        _ = (start + end) / 2.0

    fig_w = max(8.5, 0.82 * len(labels) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    bars = ax.bar(positions, values, color=colors, alpha=0.84, width=0.72)
    ax.plot(positions, values, marker="o", linewidth=1.8, color="#444444", alpha=0.7, zorder=3)

    if zero_line:
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.9, alpha=0.65, zorder=1)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        label_fmt = lambda v: f"{v:.0%}"
    else:
        label_fmt = lambda v: f"{v:.2f}"

    if annotate:
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin if ymax > ymin else 1.0
        offset = span * 0.02
        for bar, val in zip(bars, values):
            if pd.isna(val):
                continue
            y = val + offset if val >= 0 else val - offset
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                label_fmt(val),
                ha="center",
                va=va,
                fontsize=7,
                color="#333333",
            )

    legend_handles = [
        plt.Line2D([0], [0], color=fam_color[fam], lw=6, label=fam_label[fam])
        for fam in family_order
    ]
    ax.legend(
        handles=legend_handles,
        title="Family / setting",
        loc="upper right",
        fontsize=7.2,
        title_fontsize=8,
        frameon=True,
    )

    fig.tight_layout()
    save(fig, out_path)


def aggregate_modelwise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out_rows = []
    for model, g in df.groupby("model"):
        value = weighted_mean(g["value"].tolist(), g["weight"].tolist())
        out_rows.append(
            {
                "model": model,
                "value": value,
                "n": len(g),
                "weight_sum": float(g["weight"].sum()),
            }
        )
    out = pd.DataFrame(out_rows)
    return out.sort_values("value", ascending=False).reset_index(drop=True)


def _append_model_rows(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    value: Any,
    weight: float,
):
    val = finite(value)
    if val is not None:
        rows.append({"model": shorten(model), "value": val, "weight": weight})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Persuasion Drift (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_persuasion_drift(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    rows = []
    for _, row in df.iterrows():
        drift = finite(row.get("Persuasion_Drift"))
        if drift is None:
            continue
        w = weight_for_row(row)
        _append_model_rows(rows, model=row.get("Model1"), value=drift, weight=w)
        _append_model_rows(rows, model=row.get("Model2"), value=-drift, weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Persuasion Drift (signed, model-wise) — {category}",
        ylabel="Signed persuasion drift",
        out_path=out_dir / "1_persuasion_drift_modelwise.png",
        category=category,
        zero_line=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Token Cost (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_token_cost(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    rows = []
    for _, row in df.iterrows():
        w = weight_for_row(row)
        _append_model_rows(rows, model=row.get("Model1"), value=row.get("TokenCost_Model1"), weight=w)
        _append_model_rows(rows, model=row.get("Model2"), value=row.get("TokenCost_Model2"), weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Token Cost (avg tokens per argument, model-wise) — {category}",
        ylabel="Avg tokens per argument",
        out_path=out_dir / "2_token_cost_modelwise.png",
        category=category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Avg Turns to Consensus (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_avg_turns(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    rows = []
    for _, row in df.iterrows():
        turns = finite(row.get("Avg_Turns_To_Consensus"))
        if turns is None:
            continue
        w = weight_for_row(row)
        _append_model_rows(rows, model=row.get("Model1"), value=turns, weight=w)
        _append_model_rows(rows, model=row.get("Model2"), value=turns, weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Avg Turns to Consensus (reasoning steps, model-wise) — {category}",
        ylabel="Mean turns to consensus",
        out_path=out_dir / "3_avg_turns_modelwise.png",
        category=category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Argument Result (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_argument_result(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    rows = []
    for _, row in df.iterrows():
        w = weight_for_row(row)
        _append_model_rows(rows, model=row.get("Model1"), value=row.get("ArgumentResult_Model1"), weight=w)
        _append_model_rows(rows, model=row.get("Model2"), value=row.get("ArgumentResult_Model2"), weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Argument Result (response acceptance, model-wise) — {category}",
        ylabel="Mean response score (-1 to +1)",
        out_path=out_dir / "4_argument_result_modelwise.png",
        category=category,
        zero_line=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Argument Strength (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_argument_strength(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    rows = []
    for _, row in df.iterrows():
        w = weight_for_row(row)
        _append_model_rows(rows, model=row.get("Model1"), value=row.get("ArgumentStrength_Model1"), weight=w)
        _append_model_rows(rows, model=row.get("Model2"), value=row.get("ArgumentStrength_Model2"), weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Argument Strength (structured / assertive arguments, model-wise) — {category}",
        ylabel="Mean argument strength score (weak=0, strong=1)",
        out_path=out_dir / "5_argument_strength_modelwise.png",
        category=category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Persuasion Bias (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_persuasion_bias(records: list[dict], out_dir: Path, category: str):
    """
    Persuasion-bias win share.

    Uses only consensus-reached debates:
      win_share = sum(wins) / sum(Consensus_Debates)

    Any stale row where wins exceed consensus debates is skipped, because that
    indicates the underlying metrics file was generated with an older bug.
    """

    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    stats = defaultdict(lambda: {"wins": 0.0, "total": 0.0})
    skipped_invalid = 0

    for _, row in df.iterrows():
        tot = finite(row.get("Consensus_Debates"))
        if tot is None or tot <= 0:
            continue

        w1 = finite(row.get("Consensus_Wins_Model1"))
        w2 = finite(row.get("Consensus_Wins_Model2"))
        if w1 is None or w2 is None:
            continue

        if w1 + w2 > tot + 1e-9:
            skipped_invalid += 1
            continue

        m1 = shorten(row.get("Model1"))
        m2 = shorten(row.get("Model2"))

        stats[m1]["wins"] += w1
        stats[m1]["total"] += tot

        stats[m2]["wins"] += w2
        stats[m2]["total"] += tot

    if skipped_invalid:
        print(f"  [WARN] Skipped {skipped_invalid} invalid persuasion-bias rows where wins > consensus debates.")

    rows = []
    for model, s in stats.items():
        if s["total"] > 0:
            rows.append({"model": model, "value": s["wins"] / s["total"]})

    if not rows:
        print("  [SKIP] No valid persuasion-bias data.")
        return

    model_df = pd.DataFrame(rows).sort_values("value", ascending=False)

    plot_modelwise_bar_line(
        model_df,
        title=f"Persuasion Bias (win share, correctly aggregated) — {category}",
        ylabel="Win share",
        out_path=out_dir / "6_persuasion_bias_modelwise.png",
        category=category,
        percent=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Value Dominance Index (model-wise proxy)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vdi(records: list[dict], out_dir: Path, category: str):
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    # Model-wise proxy: share of consensus wins normalized by debates.
    rows = []
    for _, row in df.iterrows():
        tot = finite(row.get("Consensus_Debates"))
        if tot is None or tot <= 0:
            tot = finite(row.get("Debates"))
        if tot is None or tot <= 0:
            continue

        w1 = finite(row.get("Consensus_Wins_Model1"))
        w2 = finite(row.get("Consensus_Wins_Model2"))
        if w1 is not None:
            _append_model_rows(rows, model=row.get("Model1"), value=w1 / tot, weight=tot)
        if w2 is not None:
            _append_model_rows(rows, model=row.get("Model2"), value=w2 / tot, weight=tot)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Value Dominance Index (win rate proxy, model-wise) — {category}",
        ylabel="Wins / debate",
        out_path=out_dir / "7_vdi_modelwise.png",
        category=category,
        percent=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Morality Score (model-wise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_morality_score(records: list[dict], out_dir: Path, category: str):
    """
    Consistency across settings.

    Primary attempt:
      Compare base (Version == 'none') vs v2 rows for the same
      Model1/Model2/Value1/Value2 tuple. If the winner is unchanged,
      the pair contributes 1, otherwise 0.

    Fallback:
      If a direct merge is not possible, use Consensus_Rate as a proxy.
    """
    df = pd.DataFrame(records)
    if df.empty:
        print("  [SKIP] Empty dataframe.")
        return

    key_cols = ["Model1", "Model2", "Value1", "Value2"]

    base_df = df[df["Version"].astype(str).str.lower().isin({"none", "base", ""})].copy()
    v2_df = df[df["Version"].astype(str).str.lower().eq("v2")].copy()

    rows = []
    if not base_df.empty and not v2_df.empty:
        merged = base_df.merge(v2_df, on=key_cols, suffixes=("_base", "_v2"))
        for _, row in merged.iterrows():
            b1 = finite(row.get("Consensus_Wins_Model1_base"))
            b2 = finite(row.get("Consensus_Wins_Model2_base"))
            v1 = finite(row.get("Consensus_Wins_Model1_v2"))
            v2 = finite(row.get("Consensus_Wins_Model2_v2"))
            if None in (b1, b2, v1, v2):
                continue

            def outcome(w1: float, w2: float) -> int:
                if w1 > w2:
                    return 1
                if w2 > w1:
                    return -1
                return 0

            same = 1.0 if outcome(b1, b2) == outcome(v1, v2) else 0.0
            w = weight_for_row(pd.Series({
                "Consensus_Debates": finite(row.get("Consensus_Debates_base")) or finite(row.get("Consensus_Debates_v2")),
                "Debates": finite(row.get("Debates_base")) or finite(row.get("Debates_v2")),
            }))

            _append_model_rows(rows, model=row.get("Model1"), value=same, weight=w)
            _append_model_rows(rows, model=row.get("Model2"), value=same, weight=w)

    if not rows:
        # Fallback proxy
        for _, row in df.iterrows():
            cr = finite(row.get("Consensus_Rate"))
            if cr is None:
                continue
            w = weight_for_row(row)
            _append_model_rows(rows, model=row.get("Model1"), value=cr, weight=w)
            _append_model_rows(rows, model=row.get("Model2"), value=cr, weight=w)

    model_df = aggregate_modelwise(pd.DataFrame(rows))
    plot_modelwise_bar_line(
        model_df,
        title=f"Morality Score (consistency across settings, model-wise) — {category}",
        ylabel="Consistency score",
        out_path=out_dir / "8_morality_score_modelwise.png",
        category=category,
        percent=True,
    )


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
