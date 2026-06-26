#!/usr/bin/env python3
"""
Semantic Argument Clustering + Win Rate Analysis
================================================

For each experiment directory:
  1. Load and label arguments (winner/loser)
  2. Embed with sentence-transformers
  3. Find optimal k (2-10) via silhouette score
  4. K-Means cluster on embeddings
  5. Auto-label clusters with TF-IDF top terms
  6. Project to 2D with unsupervised UMAP
  7. Plot: UMAP scatter (colored by cluster) + win rate bar chart

INSTALL
-------
pip install sentence-transformers umap-learn scikit-learn \
            pandas matplotlib seaborn tqdm numpy

RUN
---
python semantic_cluster_winrate.py
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

EMBEDDING_MODEL   = "all-mpnet-base-v2"
RANDOM_STATE      = 42
MAX_CLUSTERS      = 10
MIN_CLUSTERS      = 2
UMAP_NEIGHBORS    = 30
UMAP_MIN_DIST     = 0.1
POINT_SIZE        = 22
ALPHA             = 0.75
TOP_TFIDF_TERMS   = 4     # words per cluster label

# ================================================================
# PATHS
# ================================================================

CURRENT_DIR  = Path(__file__).resolve().parent
ARCHIVE_DIR  = CURRENT_DIR.parent.parent / "Archive"
OUTPUT_ROOT  = CURRENT_DIR

# ================================================================
# LOAD MODEL  (once, shared across all experiments)
# ================================================================

print("\nLoading sentence transformer …")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ================================================================
# HELPERS  — taken verbatim from original pipeline
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
    Action 1  →  Agent1 wins
    Action 2  →  Agent2 wins
    """
    agent = argument_obj.get("agent", "")
    if not isinstance(agent, str):
        return "Unknown"
    agent         = agent.lower()
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


# ================================================================
# CLUSTER LABELLING  —  TF-IDF top terms per cluster
# ================================================================

def label_clusters_tfidf(texts, cluster_ids, n_terms=TOP_TFIDF_TERMS):
    """
    Treat each cluster as one document (concatenation of its texts).
    Run TF-IDF across clusters; the highest-scoring terms for each
    cluster form the semantic label.

    Returns
    -------
    dict  {cluster_id: "term1 term2 term3 term4"}
    """
    unique_clusters = sorted(set(cluster_ids))

    # Concatenate all argument texts per cluster
    cluster_docs = {
        cid: " ".join(t for t, c in zip(texts, cluster_ids) if c == cid)
        for cid in unique_clusters
    }

    corpus = [cluster_docs[cid] for cid in unique_clusters]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)   # (n_clusters, vocab)
    feature_names = np.array(vectorizer.get_feature_names_out())

    labels = {}
    for i, cid in enumerate(unique_clusters):
        row      = tfidf_matrix[i].toarray().ravel()
        top_idx  = row.argsort()[::-1][:n_terms]
        top_terms = feature_names[top_idx]
        labels[cid] = " · ".join(top_terms)

    return labels


# ================================================================
# OPTIMAL K  —  silhouette sweep
# ================================================================

def find_optimal_k(embeddings, k_min=MIN_CLUSTERS, k_max=MAX_CLUSTERS):
    """
    Sweep k from k_min to k_max, score each with silhouette score
    (computed on a subsample of max 3000 points for speed).
    Returns the best k and a dict of {k: score}.
    """
    n = len(embeddings)
    sample_size = min(n, 3000)

    rng          = np.random.default_rng(RANDOM_STATE)
    sample_idx   = rng.choice(n, size=sample_size, replace=False)
    X_sample     = embeddings[sample_idx]

    scores = {}
    for k in range(k_min, min(k_max, n - 1) + 1):
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sample)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X_sample, labels, metric="cosine")
        print(f"  k={k:2d}  silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n  → Best k = {best_k}  (score={scores[best_k]:.4f})")
    return best_k, scores


# ================================================================
# PLOTTING
# ================================================================

PALETTE = sns.color_palette("tab10", MAX_CLUSTERS)


def plot_umap_clusters(df, cluster_labels, output_path):
    """
    2D UMAP scatter coloured by cluster.
    Cluster semantic label printed at centroid.
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    unique_clusters = sorted(df["cluster"].unique())

    for cid in unique_clusters:
        sub   = df[df["cluster"] == cid]
        color = PALETTE[cid % len(PALETTE)]
        label = cluster_labels[cid]

        ax.scatter(
            sub["umap_x"], sub["umap_y"],
            s=POINT_SIZE, alpha=ALPHA,
            color=color,
            label=f"C{cid}: {label}",
            zorder=2,
        )

        # centroid annotation
        cx = sub["umap_x"].mean()
        cy = sub["umap_y"].mean()
        ax.annotate(
            f"C{cid}",
            xy=(cx, cy),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                alpha=0.85,
                linewidth=1.5,
            ),
            zorder=3,
        )

    ax.set_title(
        "Semantic Argument Clusters  (Unsupervised UMAP)",
        fontsize=18, pad=14,
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=13)
    ax.set_ylabel("UMAP Dimension 2", fontsize=13)

    # Legend outside plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Cluster  ·  Top TF-IDF Terms",
        title_fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_win_rates(df, cluster_labels, output_path):
    """
    Horizontal bar chart:  win rate per cluster, ordered high → low.
    Bars coloured by cluster id. Argument count annotated.
    Only includes clusters where winner_status is known
    (i.e. excludes 'Unknown').
    """
    known = df[df["winner_status"] != "Unknown"].copy()

    if known.empty:
        print("  [WARN] No known winner_status entries — skipping win rate plot.")
        return

    stats = (
        known
        .groupby("cluster")
        .agg(
            total     = ("winner_status", "count"),
            wins      = ("winner_status", lambda s: (s == "Winner").sum()),
        )
        .assign(win_rate=lambda d: d["wins"] / d["total"])
        .reset_index()
        .sort_values("win_rate", ascending=True)   # ascending for horizontal barh
    )

    # Build display labels
    stats["label"] = stats["cluster"].apply(
        lambda c: f"C{c}: {cluster_labels[c]}"
    )

    fig, ax = plt.subplots(figsize=(14, max(5, len(stats) * 0.75)))

    bars = ax.barh(
        y      = stats["label"],
        width  = stats["win_rate"],
        color  = [PALETTE[int(c) % len(PALETTE)] for c in stats["cluster"]],
        edgecolor = "white",
        linewidth  = 0.6,
        alpha = 0.88,
    )

    # Annotate: win rate % + (n=X)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        w = bar.get_width()
        ax.text(
            w + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w * 100:.1f}%  (n={row['total']})",
            va="center", ha="left",
            fontsize=10,
        )

    # 50 % reference line
    ax.axvline(0.5, color="black", linewidth=1.2,
               linestyle="--", alpha=0.55, label="50 % baseline")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Win Rate", fontsize=13)
    ax.set_title(
        "Win Rate by Semantic Cluster\n"
        "(fraction of arguments in each cluster that belong to the winning agent)",
        fontsize=15, pad=12,
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ================================================================
# FIND EXPERIMENT DIRECTORIES  —  identical to original
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
        if child.is_dir() and (child / "arguments").exists():
            experiment_dirs.append(child)

print(f"\nFound {len(experiment_dirs)} experiment directories")

# ================================================================
# MAIN LOOP
# ================================================================

for exp_dir in experiment_dirs:

    print("\n" + "=" * 60)
    print(f"Experiment: {exp_dir.name}")
    print("=" * 60)

    arguments_dir = exp_dir / "arguments"
    output_dir    = OUTPUT_ROOT / exp_dir.relative_to(ARCHIVE_DIR) / "plots"

    # ----------------------------------------------------------
    # 1. LOAD ARGUMENTS
    # ----------------------------------------------------------

    rows = []

    for arg_file in tqdm(sorted(arguments_dir.glob("*.json")), desc="Loading"):

        data = safe_load_json(arg_file)
        if data is None:
            continue

        final_verdict = data.get("final_verdict", "Unknown")
        arguments     = data.get("arguments", [])

        for idx, arg in enumerate(arguments):

            text = arg.get("argument", "")
            if not text:
                continue

            rows.append({
                "file"         : arg_file.name,
                "argument_id"  : idx,
                "text"         : text,
                "agent"        : arg.get("agent", "Unknown"),
                "strength"     : normalize_strength(arg.get("type", "Unknown")),
                "winner_status": infer_winner(arg, final_verdict),
                "final_verdict": final_verdict,
            })

    if not rows:
        print("  No valid arguments — skipping.")
        continue

    df = pd.DataFrame(rows)
    print(f"\n  Loaded {len(df)} arguments")

    # ----------------------------------------------------------
    # 2. EMBED
    # ----------------------------------------------------------

    print("\n  Generating embeddings …")
    embeddings = embedding_model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-sphere → cosine == dot product
    )

    # ----------------------------------------------------------
    # 3. FIND OPTIMAL K
    # ----------------------------------------------------------

    print("\n  Silhouette sweep for optimal k …")
    best_k, sil_scores = find_optimal_k(embeddings)

    # ----------------------------------------------------------
    # 4. K-MEANS ON FULL EMBEDDING SET
    # ----------------------------------------------------------

    print(f"\n  Running K-Means (k={best_k}) …")
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # ----------------------------------------------------------
    # 5. TF-IDF CLUSTER LABELS
    # ----------------------------------------------------------

    print("\n  Computing TF-IDF cluster labels …")
    cluster_labels = label_clusters_tfidf(
        df["text"].tolist(),
        df["cluster"].tolist(),
    )

    print("\n  Cluster labels:")
    for cid, lbl in sorted(cluster_labels.items()):
        n     = (df["cluster"] == cid).sum()
        wrate = (
            df[(df["cluster"] == cid) & (df["winner_status"] != "Unknown")]
            .assign(w=lambda d: d["winner_status"] == "Winner")
            ["w"].mean()
        )
        print(f"    C{cid}: {lbl}  (n={n}, win_rate={wrate:.2%})")

    # ----------------------------------------------------------
    # 6. UNSUPERVISED UMAP  (for visualisation only)
    # ----------------------------------------------------------

    print("\n  Running unsupervised UMAP …")
    umap_model = umap.UMAP(
        n_neighbors  = UMAP_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        n_components = 2,
        metric       = "cosine",
        random_state = RANDOM_STATE,
        # No target / y — purely unsupervised
    )
    coords = umap_model.fit_transform(embeddings)

    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    # ----------------------------------------------------------
    # 7. SAVE METADATA
    # ----------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "metadata.csv", index=False)

    # Add cluster label column for readability
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    # ----------------------------------------------------------
    # 8. PLOTS
    # ----------------------------------------------------------

    print("\n  Plotting …")

    plot_umap_clusters(
        df,
        cluster_labels,
        output_dir / "semantic_clusters_umap.png",
    )

    plot_win_rates(
        df,
        cluster_labels,
        output_dir / "cluster_win_rates.png",
    )

    # ----------------------------------------------------------
    # 9. SILHOUETTE SUMMARY
    # ----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))
    ks = sorted(sil_scores)
    ax.plot(ks, [sil_scores[k] for k in ks], marker="o", linewidth=2)
    ax.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs k", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_sweep.png", dpi=200)
    plt.close()

# ================================================================
# DONE
# ================================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)