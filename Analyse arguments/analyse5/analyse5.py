#!/usr/bin/env python3
"""
Semantic-only argument clustering with exactly 10 clusters.

What this version does:
- Uses ONLY semantic embeddings to cluster arguments.
- Forces 10 semantic clusters (or fewer only if the dataset is too small).
- Does NOT use win/lose, strong/weak, or rhetorical features for clustering.
- After clustering, computes win rate for each cluster.
- Generates human-readable cluster labels from cluster-specific TF-IDF terms.
- Writes cluster label + win rate on the plot itself.
- Saves metadata and cluster summary CSV files.
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"
RANDOM_STATE = 42

TARGET_CLUSTERS = 10
PCA_COMPONENTS = 50

# UMAP for 2D visualization only
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10

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


def choose_n_clusters(n_rows, target=10):
    """
    Return exactly 10 clusters when possible.
    Fall back safely only if the dataset is too small.
    """
    if n_rows < 3:
        return 1
    return min(target, n_rows - 1)


def make_cluster_label(terms):
    """
    Turn top TF-IDF terms into a compact human-readable label.
    """
    cleaned = []
    seen = set()

    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            cleaned.append(term)
            seen.add(key)

    if not cleaned:
        return "misc"

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} / {cleaned[1]}"

    return f"{cleaned[0]} / {cleaned[1]} / {cleaned[2]}"


def extract_cluster_terms(df, top_n=3):
    """
    For each cluster, concatenate all texts inside it and extract the strongest
    cluster-specific TF-IDF terms.
    """
    cluster_ids = [cid for cid in sorted(df["cluster_id"].unique()) if cid != -1]
    if not cluster_ids:
        return {}

    cluster_docs = []
    for cid in cluster_ids:
        texts = df.loc[df["cluster_id"] == cid, "text"].tolist()
        cluster_docs.append(" ".join(texts))

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=6000,
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

        top_idx = row.argsort()[::-1]
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
    Build a summary table with:
    - semantic label
    - top terms
    - cluster size
    - win rate
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
            label = make_cluster_label(terms)

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
    Plot 2D UMAP points colored by semantic cluster.
    Place one annotation per cluster with:
    - cluster semantic label
    - win rate
    - cluster size
    """
    plt.figure(figsize=(18, 13))

    unique_clusters = [c for c in sorted(df["cluster_id"].unique()) if c != -1]
    cmap = plt.get_cmap("tab10", max(1, len(unique_clusters)))

    # Scatter points
    for i, cid in enumerate(unique_clusters):
        subset = df[df["cluster_id"] == cid]
        plt.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=POINT_SIZE,
            alpha=ALPHA,
            color=cmap(i % 10),
            label=f"C{cid}",
            edgecolors="none",
        )

    noise = df[df["cluster_id"] == -1]
    if len(noise) > 0:
        plt.scatter(
            noise["umap_x"],
            noise["umap_y"],
            s=POINT_SIZE,
            alpha=0.30,
            color="lightgray",
            label="Noise",
            edgecolors="none",
        )

    # Fixed annotation offsets to reduce overlap
    offsets = [
        (28, 28),
        (-32, 28),
        (28, -34),
        (-32, -34),
        (42, 0),
        (-42, 0),
        (0, 42),
        (0, -42),
        (48, 18),
        (-48, 18),
    ]

    summary_non_noise = summary[summary["cluster_id"] != -1].copy()
    summary_non_noise = summary_non_noise.sort_values("cluster_id")

    # Cluster annotations
    for idx, (_, row) in enumerate(summary_non_noise.iterrows()):
        cid = int(row["cluster_id"])
        subset = df[df["cluster_id"] == cid]
        if len(subset) == 0:
            continue

        cx = subset["umap_x"].mean()
        cy = subset["umap_y"].mean()

        label = row["cluster_label"]
        wr = row["win_rate_pct"]
        total = int(row["total_arguments"])

        wr_text = "win rate: n/a" if pd.isna(wr) else f"win rate: {wr:.1f}%"
        text = f"C{cid}: {label}\n{wr_text} | n={total}"

        dx, dy = offsets[idx % len(offsets)]

        plt.annotate(
            text,
            xy=(cx, cy),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="black",
                alpha=0.86,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                lw=0.7,
                alpha=0.5,
                shrinkA=0,
                shrinkB=4,
            ),
        )

    plt.title(
        "Semantic UMAP — 10 Semantic Clusters from Embeddings Only",
        fontsize=20,
    )
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
    # PCA
    # ============================================================

    print("\nNormalizing and reducing with PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(semantic_embeddings)

    pca_components = min(PCA_COMPONENTS, X_scaled.shape[1], max(2, len(df) - 1))
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # ============================================================
    # FIXED 10-CLUSTER SEMANTIC CLUSTERING
    # ============================================================

    n_clusters = choose_n_clusters(len(df), TARGET_CLUSTERS)
    print(f"\nClustering into {n_clusters} semantic clusters with KMeans...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    df["cluster_id"] = kmeans.fit_predict(X_pca)

    # ============================================================
    # 2D UMAP FOR VISUALIZATION
    # ============================================================

    print("\nRunning 2D UMAP for visualization...")
    vis_umap = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
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
    print(
        summary[
            [
                "cluster_id",
                "cluster_label",
                "total_arguments",
                "known_outcomes",
                "winners",
                "losers",
                "win_rate_pct",
            ]
        ]
    )

# ================================================================
# DONE
# ================================================================

print("\n====================================================")
print("DONE")
print("====================================================")