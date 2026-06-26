#!/usr/bin/env python3
"""
Semantic-only argument clustering and win-rate analysis.

What this version does:
- Uses ONLY semantic embeddings to create clusters.
- Does NOT use win/lose, strong/weak, or rhetorical features for clustering.
- After clustering, computes win rate for each semantic cluster.
- Saves metadata, cluster summary, and a UMAP plot colored by semantic cluster.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

POINT_SIZE = 18
ALPHA = 0.72

# ================================================================
# PATHS
# ================================================================

CURRENT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = CURRENT_DIR.parent.parent / "Archive"
OUTPUT_ROOT = CURRENT_DIR

# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading sentence transformer...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

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


def choose_best_k(X, k_min=2, k_max=10, random_state=42):
    """
    Pick K for KMeans using silhouette score.
    Falls back safely if the dataset is too small.
    """
    n = len(X)
    if n < 3:
        return 1

    upper = min(k_max, n - 1)
    best_k = k_min
    best_score = -1

    for k in range(k_min, upper + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def create_plot(df, output_path):
    plt.figure(figsize=(16, 12))

    unique_clusters = sorted(df["cluster_id"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        subset = df[df["cluster_id"] == cluster_id]
        plt.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=POINT_SIZE,
            alpha=ALPHA,
            color=cmap(i),
            label=f"Cluster {cluster_id}",
        )

    plt.title("Semantic UMAP — Clusters from Semantic Embeddings Only", fontsize=20)
    plt.xlabel("UMAP Dimension 1", fontsize=15)
    plt.ylabel("UMAP Dimension 2", fontsize=15)
    plt.legend(fontsize=10, loc="best")
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
    output_dir = OUTPUT_ROOT / exp_dir.relative_to(ARCHIVE_DIR) / "plots"
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

            winner_status = infer_winner(arg, final_verdict)

            rows.append(
                {
                    "file": arg_file.name,
                    "argument_id": idx,
                    "text": argument_text,
                    "agent": arg.get("agent", "Unknown"),
                    "winner_status": winner_status,
                    "response": arg.get("response", "Unknown"),
                    "final_verdict": final_verdict,
                }
            )

    if len(rows) == 0:
        print("No valid arguments found")
        continue

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} arguments")

    known_mask = df["winner_status"].isin(["Winner", "Loser"])
    df["is_winner"] = (df["winner_status"] == "Winner").astype(int)
    df["is_known_outcome"] = known_mask.astype(int)

    # ============================================================
    # SEMANTIC EMBEDDINGS ONLY
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
    # SCALE + PCA
    # ============================================================

    print("\nNormalizing semantic embeddings...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(semantic_embeddings)

    print("\nRunning PCA...")
    pca = PCA(
        n_components=min(PCA_COMPONENTS, X_scaled.shape[1], max(2, len(df) - 1)),
        random_state=RANDOM_STATE,
    )
    X_pca = pca.fit_transform(X_scaled)

    # ============================================================
    # SEMANTIC CLUSTERING
    # ============================================================

    print("\nChoosing number of semantic clusters...")
    best_k = choose_best_k(
        X_pca,
        k_min=MIN_CLUSTERS,
        k_max=min(MAX_CLUSTERS, len(df) - 1),
        random_state=RANDOM_STATE,
    )

    print(f"Using {best_k} semantic clusters")

    if best_k == 1:
        df["cluster_id"] = 0
    else:
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        df["cluster_id"] = kmeans.fit_predict(X_pca)

    # ============================================================
    # UMAP VISUALIZATION
    # ============================================================

    print("\nRunning unsupervised UMAP on semantic features...")
    umap_model = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    umap_embeddings = umap_model.fit_transform(X_pca)
    df["umap_x"] = umap_embeddings[:, 0]
    df["umap_y"] = umap_embeddings[:, 1]

    # ============================================================
    # CLUSTER WIN-RATE SUMMARY
    # ============================================================

    print("\nComputing cluster win rates...")

    summary = (
        df.groupby("cluster_id")
        .agg(
            total_arguments=("text", "size"),
            known_outcomes=("is_known_outcome", "sum"),
            winners=("is_winner", "sum"),
        )
        .reset_index()
    )

    summary["win_rate"] = np.where(
        summary["known_outcomes"] > 0,
        summary["winners"] / summary["known_outcomes"],
        np.nan,
    )

    summary["losers"] = summary["known_outcomes"] - summary["winners"]

    cluster_labels = (
        df.groupby("cluster_id")["text"]
        .apply(lambda s: s.iloc[0][:120].replace("\n", " ") if len(s) else "")
        .reset_index(name="sample_text")
    )

    summary = summary.merge(cluster_labels, on="cluster_id", how="left")

    # Add per-row cluster win rate back to metadata
    df = df.merge(summary[["cluster_id", "win_rate", "total_arguments"]], on="cluster_id", how="left")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "metadata.csv", index=False)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    # ============================================================
    # PLOT
    # ============================================================

    print("\nCreating visualization...")
    create_plot(df, output_dir / "semantic_clusters_umap.png")

    print(f"\nSaved results to:\n{output_dir}")
    print(summary[["cluster_id", "total_arguments", "known_outcomes", "winners", "losers", "win_rate"]])

# ================================================================
# DONE
# ================================================================

print("\n====================================================")
print("DONE")
print("====================================================")