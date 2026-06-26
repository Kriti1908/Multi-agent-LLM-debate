#!/usr/bin/env python3
"""
Semantic-only argument clustering with cluster labels and win-rate annotations.

What this version does:
- Uses ONLY semantic embeddings to cluster arguments.
- Does NOT use win/lose or strong/weak as clustering inputs.
- Uses UMAP + HDBSCAN for semantic clustering.
- Generates human-readable cluster labels from cluster-specific TF-IDF terms.
- Annotates each cluster on the plot with:
    * semantic label
    * win rate
    * cluster size
- Saves metadata and cluster summary CSV files.

Install:
    pip install sentence-transformers umap-learn scikit-learn pandas matplotlib tqdm hdbscan
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "hdbscan is required for this script. Install it with: pip install hdbscan"
    ) from e

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"
RANDOM_STATE = 42

# PCA is only for denoising before clustering/UMAP; still semantic-only
PCA_COMPONENTS = 50

# UMAP for clustering space (higher-dimensional)
CLUSTER_UMAP_NEIGHBORS = 15
CLUSTER_UMAP_MIN_DIST = 0.0
CLUSTER_UMAP_COMPONENTS = 8

# UMAP for 2D visualization
VIS_UMAP_NEIGHBORS = 30
VIS_UMAP_MIN_DIST = 0.10

# HDBSCAN settings
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 5

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


def choose_min_cluster_size(n):
    """
    Scale min_cluster_size gently with dataset size.
    """
    if n < 80:
        return max(5, n // 8)
    if n < 250:
        return 15
    if n < 1000:
        return 20
    return 30


def cluster_name_from_terms(terms):
    """
    Turn top terms into a compact readable label.
    """
    terms = [t.strip() for t in terms if t and t.strip()]
    cleaned = []
    seen = set()
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            cleaned.append(t)
            seen.add(tl)

    if not cleaned:
        return "misc"

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} / {cleaned[1]}"

    return f"{cleaned[0]} / {cleaned[1]} / {cleaned[2]}"


def extract_cluster_terms(df, top_n=3):
    """
    Create semantic labels for clusters using cluster-specific TF-IDF terms.

    We concatenate all texts in a cluster, then compute TF-IDF over the cluster-docs
    and take the strongest terms for each cluster.
    """
    cluster_ids = [cid for cid in sorted(df["cluster_id"].unique()) if cid != -1]

    if len(cluster_ids) == 0:
        return {}

    cluster_docs = []
    for cid in cluster_ids:
        texts = df.loc[df["cluster_id"] == cid, "text"].tolist()
        cluster_docs.append(" ".join(texts))

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    count_matrix = vectorizer.fit_transform(cluster_docs)

    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    terms = np.array(vectorizer.get_feature_names_out())
    cluster_to_terms = {}

    for idx, cid in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray().ravel()
        if np.count_nonzero(row) == 0:
            cluster_to_terms[cid] = ["misc"]
            continue

        top_idx = row.argsort()[::-1][:top_n * 2]
        top_terms = []
        for j in top_idx:
            term = terms[j]
            if term not in top_terms:
                top_terms.append(term)
            if len(top_terms) >= top_n:
                break

        cluster_to_terms[cid] = top_terms if top_terms else ["misc"]

    return cluster_to_terms


def build_cluster_summary(df, cluster_terms):
    """
    Compute win rate and semantic label for each cluster.
    """
    rows = []

    for cid, group in df.groupby("cluster_id"):
        total = len(group)
        known = int((group["winner_status"].isin(["Winner", "Loser"])).sum())
        winners = int((group["winner_status"] == "Winner").sum())
        losers = int((group["winner_status"] == "Loser").sum())

        win_rate = (winners / known) if known > 0 else np.nan

        if cid == -1:
            label = "Noise / Unassigned"
            terms = []
        else:
            terms = cluster_terms.get(cid, ["misc"])
            label = cluster_name_from_terms(terms)

        rows.append(
            {
                "cluster_id": cid,
                "cluster_label": label,
                "top_terms": " | ".join(terms),
                "total_arguments": total,
                "known_outcomes": known,
                "winners": winners,
                "losers": losers,
                "win_rate": win_rate,
                "win_rate_pct": (win_rate * 100.0) if pd.notna(win_rate) else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["cluster_id"],
        ascending=True,
        kind="stable",
    )

    return summary


def create_plot(df, summary, output_path):
    """
    Plot UMAP points colored by semantic cluster.
    Cluster labels + win rates are written on the plot itself.
    """
    plt.figure(figsize=(18, 13))

    unique_clusters = [c for c in sorted(df["cluster_id"].unique()) if c != -1]
    cmap = plt.get_cmap("tab20", max(1, len(unique_clusters)))

    # Plot clusters
    for i, cid in enumerate(unique_clusters):
        subset = df[df["cluster_id"] == cid]
        plt.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=POINT_SIZE,
            alpha=ALPHA,
            color=cmap(i),
            label=f"C{cid}",
            edgecolors="none",
        )

    # Noise points, if any
    noise = df[df["cluster_id"] == -1]
    if len(noise) > 0:
        plt.scatter(
            noise["umap_x"],
            noise["umap_y"],
            s=POINT_SIZE,
            alpha=0.35,
            color="lightgray",
            label="Noise",
            edgecolors="none",
        )

    # Annotate cluster centroids with semantic label + win rate
    for _, row in summary.iterrows():
        cid = int(row["cluster_id"])
        if cid == -1:
            continue

        subset = df[df["cluster_id"] == cid]
        if len(subset) == 0:
            continue

        cx = subset["umap_x"].mean()
        cy = subset["umap_y"].mean()

        label = row["cluster_label"]
        wr = row["win_rate_pct"]
        total = int(row["total_arguments"])

        if pd.isna(wr):
            wr_text = "win rate: n/a"
        else:
            wr_text = f"win rate: {wr:.1f}%"

        text = f"C{cid}: {label}\n{wr_text} | n={total}"

        plt.annotate(
            text,
            xy=(cx, cy),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.82,
            ),
        )

    plt.title("Semantic UMAP — HDBSCAN Clusters from Semantic Embeddings Only", fontsize=20)
    plt.xlabel("UMAP Dimension 1", fontsize=15)
    plt.ylabel("UMAP Dimension 2", fontsize=15)
    plt.legend(fontsize=10, loc="best", frameon=True)
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

    argument_files = sorted(arguments_dir.glob("*.json"))
    if len(argument_files) == 0:
        print("No argument files found")
        continue

    # ------------------------------------------------------------
    # LOAD ARGUMENTS
    # ------------------------------------------------------------
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
    # OPTIONAL DENOISING PCA
    # ============================================================

    print("\nNormalizing and reducing with PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(semantic_embeddings)

    pca_components = min(PCA_COMPONENTS, X_scaled.shape[1], max(2, len(df) - 1))
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # ============================================================
    # UMAP SPACE FOR CLUSTERING
    # ============================================================

    print("\nBuilding semantic clustering space with UMAP...")
    cluster_umap = umap.UMAP(
        n_neighbors=CLUSTER_UMAP_NEIGHBORS,
        min_dist=CLUSTER_UMAP_MIN_DIST,
        n_components=CLUSTER_UMAP_COMPONENTS,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    X_cluster = cluster_umap.fit_transform(X_pca)

    # ============================================================
    # HDBSCAN SEMANTIC CLUSTERING
    # ============================================================

    print("\nClustering with HDBSCAN...")
    min_cluster_size = choose_min_cluster_size(len(df))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    df["cluster_id"] = clusterer.fit_predict(X_cluster)

    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in set(df["cluster_id"]) else 0)
    noise_count = int((df["cluster_id"] == -1).sum())
    print(f"Found {n_clusters} clusters; noise points: {noise_count}")

    # ============================================================
    # 2D UMAP FOR VISUALIZATION
    # ============================================================

    print("\nRunning 2D UMAP for visualization...")
    vis_umap = umap.UMAP(
        n_neighbors=VIS_UMAP_NEIGHBORS,
        min_dist=VIS_UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    vis_embeddings = vis_umap.fit_transform(X_pca)
    df["umap_x"] = vis_embeddings[:, 0]
    df["umap_y"] = vis_embeddings[:, 1]

    # ============================================================
    # CLUSTER LABELS + WIN RATES
    # ============================================================

    print("\nExtracting semantic cluster labels...")
    cluster_terms = extract_cluster_terms(df, top_n=3)
    summary = build_cluster_summary(df, cluster_terms)

    # Backfill labels/terms into each row
    label_map = summary.set_index("cluster_id")["cluster_label"].to_dict()
    terms_map = summary.set_index("cluster_id")["top_terms"].to_dict()
    wr_map = summary.set_index("cluster_id")["win_rate_pct"].to_dict()

    df["cluster_label"] = df["cluster_id"].map(label_map).fillna("Noise / Unassigned")
    df["cluster_terms"] = df["cluster_id"].map(terms_map).fillna("")
    df["cluster_win_rate_pct"] = df["cluster_id"].map(wr_map)

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
    create_plot(df, summary, output_dir / "semantic_clusters_umap.png")

    print(f"\nSaved results to:\n{output_dir}")
    print(summary[["cluster_id", "cluster_label", "total_arguments", "known_outcomes", "winners", "losers", "win_rate_pct"]])

# ================================================================
# DONE
# ================================================================

print("\n====================================================")
print("DONE")
print("====================================================")