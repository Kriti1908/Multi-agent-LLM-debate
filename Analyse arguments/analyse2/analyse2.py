#!/usr/bin/env python3

"""
========================================================================
ADVANCED ARGUMENT VISUALIZATION PIPELINE
========================================================================

GOAL
----
Create meaningful argument-space visualizations where:
    - Strong Winner
    - Weak Winner
    - Strong Loser
    - Weak Loser

show visible separation.

KEY IMPROVEMENTS
----------------
Instead of only semantic embeddings, this pipeline uses:

1. Semantic embeddings
2. Rhetorical features
3. PCA compression
4. SUPERVISED UMAP

This creates separation based on:
    persuasion style,
    reasoning structure,
    argumentative strength,
    winning behavior

instead of ONLY topic similarity.

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
    numpy \
    spacy \
    textstat

python -m spacy download en_core_web_sm

========================================================================
RUN
========================================================================

python advanced_argument_visualization.py

========================================================================
DIRECTORY STRUCTURE EXPECTED
========================================================================

Current file:
    ./advanced_argument_visualization.py

Dataset:
    ../Archive/

========================================================================
OUTPUT
========================================================================

Creates SAME directory structure locally:

./Reasoning vs Non-reasoning/.../plots/
./Runs/Untitled/.../plots/
./Small vs Large/.../plots/

Each contains:
    - supervised_umap_combined.png
    - metadata.csv

========================================================================
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import textstat
import umap.umap_ as umap

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"

RANDOM_STATE = 42

PCA_COMPONENTS = 50

UMAP_NEIGHBORS = 40
UMAP_MIN_DIST = 0.15

POINT_SIZE = 18
ALPHA = 0.72

# ================================================================
# PATHS
# ================================================================

CURRENT_DIR = Path(__file__).resolve().parent

ARCHIVE_DIR = CURRENT_DIR.parent.parent / "Archive"

OUTPUT_ROOT = CURRENT_DIR

# ================================================================
# LOAD MODELS
# ================================================================

print("\nLoading sentence transformer...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("\nLoading spaCy model...")

nlp = spacy.load("en_core_web_sm")

# ================================================================
# RHETORICAL FEATURE WORD LISTS
# ================================================================

CAUSAL_WORDS = {
    "because",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "causes",
    "results",
    "leads",
    "implies",
}

HEDGE_WORDS = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "likely",
    "suggests",
}

LEGAL_WORDS = {
    "law",
    "legal",
    "policy",
    "regulation",
    "gdpr",
    "ccpa",
    "compliance",
    "constitutional",
}

MITIGATION_WORDS = {
    "prevent",
    "mitigate",
    "reduce",
    "protect",
    "safeguard",
    "minimize",
}

CONFIDENCE_WORDS = {
    "must",
    "clearly",
    "definitely",
    "certainly",
    "undeniably",
    "absolutely",
}

# ================================================================
# COLOR MAP
# ================================================================

LABEL_COLORS = {
    "Strong_Winner": "#1f77b4",
    "Weak_Winner": "#6baed6",
    "Strong_Loser": "#d62728",
    "Weak_Loser": "#ff9896",
    "Unknown": "#7f7f7f",
}

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


def infer_winner(argument_obj, final_verdict):
    """
    Infer winner using:
        Action 1 -> Agent1 wins
        Action 2 -> Agent2 wins
    """

    agent = argument_obj.get("agent", "")

    if not isinstance(agent, str):
        return "Unknown"

    agent = agent.lower()

    final_verdict = str(final_verdict).lower()

    if "action 1" in final_verdict:

        if agent.startswith("agent1"):
            return "Winner"

        if agent.startswith("agent2"):
            return "Loser"

    elif "action 2" in final_verdict:

        if agent.startswith("agent2"):
            return "Winner"

        if agent.startswith("agent1"):
            return "Loser"

    return "Unknown"


def rhetorical_features(text):

    text_lower = text.lower()

    words = re.findall(r"\w+", text_lower)

    doc = nlp(text)

    num_tokens = len(words)

    unique_ratio = len(set(words)) / max(1, num_tokens)

    return np.array([

        # length
        num_tokens,

        # readability
        textstat.flesch_reading_ease(text),

        # causal reasoning
        sum(w in CAUSAL_WORDS for w in words),

        # hedge language
        sum(w in HEDGE_WORDS for w in words),

        # legal/policy grounding
        sum(w in LEGAL_WORDS for w in words),

        # mitigation language
        sum(w in MITIGATION_WORDS for w in words),

        # confidence language
        sum(w in CONFIDENCE_WORDS for w in words),

        # verbs
        sum(tok.pos_ == "VERB" for tok in doc),

        # adjectives
        sum(tok.pos_ == "ADJ" for tok in doc),

        # numeric/statistical evidence
        sum(tok.like_num for tok in doc),

        # lexical diversity
        unique_ratio,

    ])


def create_plot(df, output_path):

    plt.figure(figsize=(16, 12))

    unique_labels = sorted(df["combined_label"].unique())

    for label in unique_labels:

        subset = df[df["combined_label"] == label]

        plt.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=LABEL_COLORS.get(label, "#7f7f7f"),
            label=label,
        )

    plt.title(
        "Supervised UMAP — Argument Strength + Debate Outcome",
        fontsize=20,
    )

    plt.xlabel("UMAP Dimension 1", fontsize=15)
    plt.ylabel("UMAP Dimension 2", fontsize=15)

    plt.legend(fontsize=12)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=400)

    plt.close()

# ================================================================
# FIND EXPERIMENT DIRECTORIES
# ================================================================

experiment_dirs = []

candidate_roots = [
    ARCHIVE_DIR / "Reasoning vs Non-reasoning",
    ARCHIVE_DIR / "Runs" / "Untitled",
    ARCHIVE_DIR / "Small vs Large",
]

for root_dir in candidate_roots:

    if not root_dir.exists():
        continue

    for child in root_dir.iterdir():

        if child.is_dir():

            arguments_dir = child / "arguments"

            if arguments_dir.exists():
                experiment_dirs.append(child)

print(f"\nFound {len(experiment_dirs)} experiment directories")

# ================================================================
# PROCESS EACH DIRECTORY
# ================================================================

for exp_dir in experiment_dirs:

    print("\n====================================================")
    print(f"Processing: {exp_dir}")
    print("====================================================")

    arguments_dir = exp_dir / "arguments"

    output_dir = (
        OUTPUT_ROOT
        / exp_dir.relative_to(ARCHIVE_DIR)
        / "plots"
    )

    rows = []

    # ============================================================
    # LOAD ARGUMENT FILES
    # ============================================================

    argument_files = sorted(arguments_dir.glob("*.json"))

    if len(argument_files) == 0:

        print("No argument files found")

        continue

    for arg_file in tqdm(argument_files):

        data = safe_load_json(arg_file)

        if data is None:
            continue

        final_verdict = data.get("final_verdict", "Unknown")

        arguments = data.get("arguments", [])

        for idx, arg in enumerate(arguments):

            argument_text = arg.get("argument", "")

            if not argument_text:
                continue

            strength = normalize_strength(
                arg.get("type", "Unknown")
            )

            winner_status = infer_winner(
                arg,
                final_verdict,
            )

            combined_label = (
                f"{strength}_{winner_status}"
            )

            rows.append({
                "file": arg_file.name,
                "argument_id": idx,
                "text": argument_text,
                "agent": arg.get("agent", "Unknown"),
                "strength": strength,
                "winner_status": winner_status,
                "combined_label": combined_label,
                "response": arg.get("response", "Unknown"),
                "final_verdict": final_verdict,
            })

    if len(rows) == 0:

        print("No valid arguments found")

        continue

    df = pd.DataFrame(rows)

    print(f"\nLoaded {len(df)} arguments")

    # ============================================================
    # SEMANTIC EMBEDDINGS
    # ============================================================

    print("\nGenerating semantic embeddings...")

    semantic_embeddings = embedding_model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ============================================================
    # RHETORICAL FEATURES
    # ============================================================

    print("\nGenerating rhetorical features...")

    rhetorical_matrix = np.vstack([
        rhetorical_features(text)
        for text in tqdm(df["text"])
    ])

    # ============================================================
    # COMBINE FEATURES
    # ============================================================

    print("\nCombining features...")

    X = np.hstack([
        semantic_embeddings,
        rhetorical_matrix,
    ])

    # ============================================================
    # NORMALIZE
    # ============================================================

    print("\nNormalizing...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ============================================================
    # PCA
    # ============================================================

    print("\nRunning PCA...")

    pca = PCA(
        n_components=min(PCA_COMPONENTS, X_scaled.shape[1]),
        random_state=RANDOM_STATE,
    )

    X_pca = pca.fit_transform(X_scaled)

    # ============================================================
    # LABELS
    # ============================================================

    label_to_id = {
        label: idx
        for idx, label in enumerate(
            sorted(df["combined_label"].unique())
        )
    }

    y_labels = np.array([
        label_to_id[x]
        for x in df["combined_label"]
    ])

    # ============================================================
    # SUPERVISED UMAP
    # ============================================================

    print("\nRunning supervised UMAP...")

    umap_model = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,

        metric="cosine",

        target_metric="categorical",

        target_weight=0.35,

        random_state=RANDOM_STATE,
    )

    umap_embeddings = umap_model.fit_transform(
        X_pca,
        y=y_labels,
    )

    df["umap_x"] = umap_embeddings[:, 0]
    df["umap_y"] = umap_embeddings[:, 1]

    # ============================================================
    # SAVE METADATA
    # ============================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    df.to_csv(metadata_path, index=False)

    # ============================================================
    # PLOT
    # ============================================================

    print("\nCreating visualization...")

    create_plot(
        df,
        output_dir / "supervised_umap_combined.png",
    )

    print(f"\nSaved results to:\n{output_dir}")

# ================================================================
# DONE
# ================================================================

print("\n====================================================")
print("DONE")
print("====================================================")