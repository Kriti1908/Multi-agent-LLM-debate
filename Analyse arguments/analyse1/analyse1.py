#!/usr/bin/env python3

"""
========================================================================
Argument Embedding + UMAP/t-SNE Visualization Pipeline
========================================================================

WHAT THIS DOES
---------------
This script:

1. Recursively scans:
       ../Archive/

2. Finds all:
       arguments/*.json

3. Loads all arguments.

4. Creates sentence embeddings for every argument.

5. Generates:
       - UMAP plots
       - t-SNE plots

6. Colors points by:
       - Strong/Weak
       - Consensus/No Consensus
       - Winning/Losing side

7. Saves plots in SAME directory structure
   inside current working directory:

Example:
--------
INPUT:
../Archive/Reasoning vs Non-reasoning/reasoning_X1_vs_X2/arguments/*.json

OUTPUT:
./Reasoning vs Non-reasoning/reasoning_X1_vs_X2/plots/

========================================================================
INSTALL
========================================================================

pip install \
    sentence-transformers \
    umap-learn \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    numpy

========================================================================
RUN
========================================================================

python visualize_arguments.py

========================================================================
OUTPUT FILES
========================================================================

For each experiment folder:

plots/
    umap_strength.png
    umap_winner.png
    umap_consensus.png
    tsne_strength.png
    tsne_winner.png
    tsne_consensus.png
    metadata.csv

========================================================================
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"

RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

TSNE_PERPLEXITY = 30

POINT_SIZE = 16
ALPHA = 0.75

# ================================================================
# PATHS
# ================================================================

CURRENT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = CURRENT_DIR.parent / "Archive"

OUTPUT_ROOT = CURRENT_DIR

# ================================================================
# COLOR MAPS
# ================================================================

STRENGTH_COLORS = {
    "Strong": "#1f77b4",
    "Weak": "#ff7f0e",
    "Unknown": "#7f7f7f",
}

WINNER_COLORS = {
    "Winner": "#2ca02c",
    "Loser": "#d62728",
    "Unknown": "#7f7f7f",
}

CONSENSUS_COLORS = {
    "Consensus": "#9467bd",
    "No Consensus": "#8c564b",
    "Unknown": "#7f7f7f",
}

# ================================================================
# LOAD MODEL
# ================================================================

print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# ================================================================
# HELPERS
# ================================================================


def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed loading {path}: {e}")
        return None


def normalize_strength(x):
    if not isinstance(x, str):
        return "Unknown"

    x = x.strip().lower()

    if x == "strong":
        return "Strong"

    if x == "weak":
        return "Weak"

    return "Unknown"


def normalize_consensus(verdict):
    if not isinstance(verdict, str):
        return "Unknown"

    verdict = verdict.lower()

    if "no consensus" in verdict:
        return "No Consensus"

    return "Consensus"


def infer_winner(argument_obj, final_verdict):
    """
    Infer whether argument belongs to winning side.

    Rules:
    -------
    final_verdict == "Action 1"
        -> Agent1_* arguments are winners
        -> Agent2_* arguments are losers

    final_verdict == "Action 2"
        -> Agent2_* arguments are winners
        -> Agent1_* arguments are losers
    """

    agent = argument_obj.get("agent", "")

    if not isinstance(agent, str):
        return "Unknown"

    agent = agent.lower()

    if "action 1" in final_verdict.lower():

        if agent.startswith("agent1"):
            return "Winner"

        if agent.startswith("agent2"):
            return "Loser"

    elif "action 2" in final_verdict.lower():

        if agent.startswith("agent2"):
            return "Winner"

        if agent.startswith("agent1"):
            return "Loser"

    return "Unknown"


def create_scatter_plot(
    df,
    x_col,
    y_col,
    color_col,
    color_map,
    title,
    output_path,
):
    plt.figure(figsize=(14, 10))

    unique_values = sorted(df[color_col].dropna().unique())

    for value in unique_values:

        subset = df[df[color_col] == value]

        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=color_map.get(value, "#7f7f7f"),
            label=value,
        )

    plt.title(title, fontsize=18)
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)

    plt.legend(fontsize=12)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)

    plt.close()


# ================================================================
# FIND EXPERIMENT FOLDERS
# ================================================================

experiment_dirs = []

for root_dir in [
    ARCHIVE_DIR / "Reasoning vs Non-reasoning",
    ARCHIVE_DIR / "Runs" / "Untitled",
    ARCHIVE_DIR / "Small vs Large",
]:

    if not root_dir.exists():
        continue

    for child in root_dir.iterdir():

        if child.is_dir():

            arguments_dir = child / "arguments"

            if arguments_dir.exists():
                experiment_dirs.append(child)

print(f"\nFound {len(experiment_dirs)} experiment directories")

# ================================================================
# PROCESS EACH EXPERIMENT
# ================================================================

for exp_dir in experiment_dirs:

    print(f"\n================================================")
    print(f"Processing: {exp_dir}")
    print(f"================================================")

    arguments_dir = exp_dir / "arguments"

    output_dir = (
        OUTPUT_ROOT / exp_dir.relative_to(ARCHIVE_DIR) / "plots"
    )

    rows = []

    # ------------------------------------------------------------
    # LOAD ALL ARGUMENT FILES
    # ------------------------------------------------------------

    argument_files = sorted(arguments_dir.glob("*.json"))

    if len(argument_files) == 0:
        print("No argument files found")
        continue

    for arg_file in tqdm(argument_files):

        data = safe_load_json(arg_file)

        if data is None:
            continue

        final_verdict = data.get("final_verdict", "Unknown")
        consensus = normalize_consensus(final_verdict)

        arguments = data.get("arguments", [])

        for idx, arg in enumerate(arguments):

            argument_text = arg.get("argument", "")

            if not argument_text or len(argument_text.strip()) == 0:
                continue

            rows.append(
                {
                    "file": arg_file.name,
                    "argument_id": idx,
                    "text": argument_text,
                    "agent": arg.get("agent", "Unknown"),
                    "strength": normalize_strength(
                        arg.get("type", "Unknown")
                    ),
                    "response": arg.get("response", "Unknown"),
                    "winner_status": infer_winner(
                        arg,
                        final_verdict,
                    ),
                    "consensus": consensus,
                    "final_verdict": final_verdict,
                }
            )

    if len(rows) == 0:
        print("No valid arguments found")
        continue

    df = pd.DataFrame(rows)

    print(f"Loaded {len(df)} arguments")

    # ------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------

    print("\nRunning UMAP...")

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    umap_embeddings = umap_model.fit_transform(embeddings)

    df["umap_x"] = umap_embeddings[:, 0]
    df["umap_y"] = umap_embeddings[:, 1]

    # ------------------------------------------------------------
    # TSNE
    # ------------------------------------------------------------

    print("\nRunning t-SNE...")

    tsne_model = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, max(5, len(df) // 5)),
        metric="cosine",
        random_state=RANDOM_STATE,
        init="pca",
    )

    tsne_embeddings = tsne_model.fit_transform(embeddings)

    df["tsne_x"] = tsne_embeddings[:, 0]
    df["tsne_y"] = tsne_embeddings[:, 1]

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    df.to_csv(metadata_path, index=False)

    # ------------------------------------------------------------
    # UMAP PLOTS
    # ------------------------------------------------------------

    print("\nSaving UMAP plots...")

    create_scatter_plot(
        df,
        "umap_x",
        "umap_y",
        "strength",
        STRENGTH_COLORS,
        "UMAP — Strong vs Weak Arguments",
        output_dir / "umap_strength.png",
    )

    create_scatter_plot(
        df,
        "umap_x",
        "umap_y",
        "winner_status",
        WINNER_COLORS,
        "UMAP — Winning vs Losing Arguments",
        output_dir / "umap_winner.png",
    )

    create_scatter_plot(
        df,
        "umap_x",
        "umap_y",
        "consensus",
        CONSENSUS_COLORS,
        "UMAP — Consensus vs No Consensus",
        output_dir / "umap_consensus.png",
    )

    # ------------------------------------------------------------
    # TSNE PLOTS
    # ------------------------------------------------------------

    print("\nSaving t-SNE plots...")

    create_scatter_plot(
        df,
        "tsne_x",
        "tsne_y",
        "strength",
        STRENGTH_COLORS,
        "t-SNE — Strong vs Weak Arguments",
        output_dir / "tsne_strength.png",
    )

    create_scatter_plot(
        df,
        "tsne_x",
        "tsne_y",
        "winner_status",
        WINNER_COLORS,
        "t-SNE — Winning vs Losing Arguments",
        output_dir / "tsne_winner.png",
    )

    create_scatter_plot(
        df,
        "tsne_x",
        "tsne_y",
        "consensus",
        CONSENSUS_COLORS,
        "t-SNE — Consensus vs No Consensus",
        output_dir / "tsne_consensus.png",
    )

    # ------------------------------------------------------------
    # OPTIONAL: COMBINED PLOT
    # ------------------------------------------------------------

    print("\nSaving combined plot...")

    plt.figure(figsize=(16, 12))

    combined_labels = (
        df["strength"].astype(str)
        + " | "
        + df["winner_status"].astype(str)
    )

    unique_combined = sorted(combined_labels.unique())

    palette = sns.color_palette(
        "tab10",
        n_colors=len(unique_combined),
    )

    color_lookup = {
        label: palette[i]
        for i, label in enumerate(unique_combined)
    }

    for label in unique_combined:

        subset = df[combined_labels == label]

        plt.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=[color_lookup[label]],
            label=label,
        )

    plt.title(
        "UMAP — Strength + Winner Combined",
        fontsize=18,
    )

    plt.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig(
        output_dir / "umap_combined.png",
        dpi=300,
    )

    plt.close()

    print(f"\nSaved plots to:\n{output_dir}")

print("\n================================================")
print("DONE")
print("================================================")