from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




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


def normalise_verdict(verdict: object) -> str:
    if verdict is None:
        return ""
    return str(verdict).strip().lower()


def winner_from_verdict(
    verdict: object,
    values1: str,
    values2: str,
    swap_action_mapping: bool = False,
) -> str | None:
    verdict_text = normalise_verdict(verdict)

    if verdict_text in {"action 1", "action1", "1"}:
        return values2 if swap_action_mapping else values1
    if verdict_text in {"action 2", "action2", "2"}:
        return values1 if swap_action_mapping else values2
    if verdict_text == values1.lower():
        return values1
    if verdict_text == values2.lower():
        return values2
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# -------------------------------
# 🔹 CORE: Collect stats per pair
# -------------------------------
def collect_drift_stats(
    base_dir: Path,
    v2_dir: Path,
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    set[str],
]:
    totals: DefaultDict[tuple[str, str], int] = defaultdict(int)
    drift_toward_y: DefaultDict[tuple[str, str], int] = defaultdict(int)
    drift_toward_x: DefaultDict[tuple[str, str], int] = defaultdict(int)
    seen_values: set[str] = set()

    base_arguments_dir = base_dir / "arguments"
    v2_arguments_dir = v2_dir / "arguments"

    for base_file in sorted(base_dir.glob("*.json")):
        if base_file.name.endswith(".failed.json"):
            continue

        stem = base_file.stem
        v2_file = v2_dir / f"{stem}.json"
        base_arguments_file = base_arguments_dir / f"{stem}.arguments.json"
        v2_arguments_file = v2_arguments_dir / f"{stem}.arguments.json"

        if not (v2_file.exists() and base_arguments_file.exists() and v2_arguments_file.exists()):
            continue

        base_data = load_json(base_file)
        base_arguments = load_json(base_arguments_file)
        v2_arguments = load_json(v2_arguments_file)

        values1 = base_data.get("values1")
        values2 = base_data.get("values2")
        if not isinstance(values1, str) or not isinstance(values2, str):
            continue

        seen_values.update({values1, values2})

        base_winner = winner_from_verdict(
            base_arguments.get("final_verdict"),
            values1,
            values2,
            swap_action_mapping=False,
        )

        v2_winner = winner_from_verdict(
            v2_arguments.get("final_verdict"),
            values1,
            values2,
            swap_action_mapping=False,
        )

        if base_winner is None or v2_winner is None:
            continue

        pair = (values1, values2)
        totals[pair] += 1

        if base_winner != v2_winner:
            if v2_winner == values1:
                drift_toward_y[pair] += 1
            elif v2_winner == values2:
                drift_toward_x[pair] += 1

    return dict(totals), dict(drift_toward_y), dict(drift_toward_x), seen_values




# (keep your existing functions unchanged)
# winner_from_verdict, load_json, collect_drift_stats, build_matrices...

# -------------------------------
# FIND PAIRS
# -------------------------------
def find_folder_pairs(parent: Path):
    base_map = {}
    v2_map = {}

    for folder in parent.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name
        if name.endswith("_v2"):
            v2_map[name[:-3]] = folder
        else:
            base_map[name] = folder

    pairs = []
    for key in base_map:
        if key in v2_map:
            pairs.append((key, base_map[key], v2_map[key]))

    return pairs


# -------------------------------
# SHORT NAME
# -------------------------------
def prettify(name: str):
    return name.replace("reasoning_", "").replace("_", " ").replace(" vs ", "\nvs ")


# -------------------------------
# MULTI HEATMAP PLOT
# -------------------------------
def build_matrices(
    totals: dict[tuple[str, str], int],
    drift_toward_y: dict[tuple[str, str], int],
    drift_toward_x: dict[tuple[str, str], int],
    values: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    value_index = {value: idx for idx, value in enumerate(values)}

    counts = pd.DataFrame(0, index=values, columns=values, dtype=int)
    switched_counts = pd.DataFrame(0, index=values, columns=values, dtype=int)
    signed_drift_counts = pd.DataFrame(0, index=values, columns=values, dtype=int)

    for (values1, values2), total in totals.items():
        if values1 not in values or values2 not in values:
            continue

        toward_y = drift_toward_y.get((values1, values2), 0)
        toward_x = drift_toward_x.get((values1, values2), 0)

        switched = toward_y + toward_x
        drift_signed = toward_y - toward_x

        row_value, col_value = values1, values2

        # lower triangle logic (same as your original)
        if value_index[values1] < value_index[values2]:
            row_value, col_value = values2, values1
            drift_signed = -drift_signed

        counts.loc[row_value, col_value] += total
        switched_counts.loc[row_value, col_value] += switched
        signed_drift_counts.loc[row_value, col_value] += drift_signed

    persuasion_score = pd.DataFrame(np.nan, index=values, columns=values, dtype=float)

    for row in values:
        for col in values:
            total = counts.loc[row, col]
            if total == 0:
                continue
            persuasion_score.loc[row, col] = (
                signed_drift_counts.loc[row, col] / total
            ) * 100

    return persuasion_score, switched_counts, counts

def plot_multiple_heatmaps(results, output_path):
    n = len(results)

    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten()

    for i, (title, df) in enumerate(results):
        ax = axes[i]

        mask = df.isna()

        sns.heatmap(
            df,
            mask=mask,
            annot=True,
            fmt=".1f",
            cmap="bwr",
            center=0,
            vmin=-100,
            vmax=100,
            linewidths=0.5,
            linecolor="white",
            cbar=(i == n - 1),  # only last one shows colorbar
            ax=ax,
        )

        ax.set_title(title, fontsize=12)

    # remove empty plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    
def plot_single_heatmap(df, title, output_path):
    plt.figure(figsize=(10, 8))

    mask = df.isna()

    sns.heatmap(
        df,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="bwr",
        center=0,
        vmin=-100,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Persuasion score (%)"},
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_folder", type=Path)
    args = parser.parse_args()

    pairs = find_folder_pairs(args.parent_folder)

    if not pairs:
        raise ValueError("No matching base/v2 folder pairs found")

    for key, base_dir, v2_dir in pairs:
        print(f"[PAIR] {key}")

        totals, drift_y, drift_x, seen_values = collect_drift_stats(base_dir, v2_dir)

        values = [v for v in VALUE_ORDER if v in seen_values]

        persuasion_score, _, _ = build_matrices(
            totals, drift_y, drift_x, values
        )

        safe_name = key.replace("/", "_")
        output_file = args.parent_folder / f"{safe_name}.png"

        plot_single_heatmap(
            persuasion_score,
            prettify(key),
            output_file
        )

        print(f"✅ Saved: {output_file}")
        
if __name__ == "__main__":
    main()