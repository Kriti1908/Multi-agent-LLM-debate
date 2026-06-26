#!/usr/bin/env python3
"""
SEMANTIC WIN PREDICTOR — 4 Visualization Methods
=================================================

Method 1: WINNING DIRECTION SPECTRUM
  Project arguments onto winner_centroid − loser_centroid axis.
  Show vocabulary at each end + win rate per quintile.

Method 2: LOG-ODDS CLUSTER ENRICHMENT
  K-Means clusters. Compute log₂(P(cluster|winner)/P(cluster|loser)).
  Shows which semantic topics are over-represented among winners.

Method 3: PER-DEBATE ADVANTAGE VECTORS
  For each debate, compute winner_emb_mean − loser_emb_mean.
  Cluster these vectors → types of winning semantic moves.

Method 4: WIN PROBABILITY LANDSCAPE
  UMAP projection + micro-clustering. Color continuously by local win rate.
  Shows WHERE in semantic space winning arguments live.

Install:
  pip install sentence-transformers umap-learn scikit-learn \
              pandas matplotlib seaborn tqdm numpy
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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ====================================================================
# CONFIG
# ====================================================================
EMBEDDING_MODEL = "all-mpnet-base-v2"
RANDOM_STATE    = 42
MACRO_K         = 10    # clusters for Methods 1 & 2
MICRO_K         = 40    # fine-grained clusters for Method 4
N_TFIDF_TERMS   = 5
UMAP_NEIGHBORS  = 30
UMAP_MIN_DIST   = 0.10
POINT_SIZE      = 18
ALPHA_SCATTER   = 0.70

# ====================================================================
# PATHS
# ====================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = CURRENT_DIR.parent.parent / "Archive"
OUTPUT_ROOT = CURRENT_DIR

# ====================================================================
# MODEL (loaded once, shared across all experiments)
# ====================================================================
print("Loading sentence transformer …")
model = SentenceTransformer(EMBEDDING_MODEL)


# ====================================================================
# HELPERS
# ====================================================================

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [ERROR] {path}: {e}")
        return None


def normalize_strength(x):
    if not isinstance(x, str):
        return "Unknown"
    x = x.strip().lower()
    return "Strong" if x == "strong" else ("Weak" if x == "weak" else "Unknown")


def infer_winner(arg_obj, final_verdict):
    """Taken verbatim from original pipeline."""
    agent = arg_obj.get("agent", "")
    if not isinstance(agent, str):
        return "Unknown"
    agent = agent.lower()
    fv    = str(final_verdict).lower()
    if "action 1" in fv:
        if agent.startswith("agent1"): return "Winner"
        if agent.startswith("agent2"): return "Loser"
    elif "action 2" in fv:
        if agent.startswith("agent2"): return "Winner"
        if agent.startswith("agent1"): return "Loser"
    return "Unknown"


def tfidf_top_terms(texts_by_group: dict, n_terms: int = N_TFIDF_TERMS) -> dict:
    """
    Inter-group TF-IDF so returned terms are *distinctive* per group.
    texts_by_group: {group_id: [text, ...]}
    Returns: {group_id: "term1 · term2 · ..."}
    """
    groups  = sorted(texts_by_group)
    corpus  = [" ".join(texts_by_group[g]) or "empty" for g in groups]
    vec     = TfidfVectorizer(
        max_features=8000, stop_words="english",
        ngram_range=(1, 2), min_df=1,
    )
    mat   = vec.fit_transform(corpus)
    feats = np.array(vec.get_feature_names_out())
    result = {}
    for i, g in enumerate(groups):
        row     = mat[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:n_terms]
        result[g] = " · ".join(feats[top_idx])
    return result


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path.name}")


# ====================================================================
# DISCOVER EXPERIMENT DIRECTORIES
# ====================================================================
experiment_dirs = []
for root in [
    ARCHIVE_DIR / "Reasoning vs Non-reasoning",
    ARCHIVE_DIR / "Runs" / "Untitled",
    ARCHIVE_DIR / "Small vs Large",
]:
    if not root.exists():
        continue
    for child in root.iterdir():
        if child.is_dir() and (child / "arguments").exists():
            experiment_dirs.append(child)

print(f"\nFound {len(experiment_dirs)} experiment directories\n")


# ====================================================================
# MAIN LOOP
# ====================================================================
for exp_dir in experiment_dirs:

    print(f"{'='*60}")
    print(f"Experiment: {exp_dir.name}")
    print(f"{'='*60}")

    out = OUTPUT_ROOT / exp_dir.relative_to(ARCHIVE_DIR) / "plots_semantic_win"

    # ------------------------------------------------------------------
    # LOAD ARGUMENTS
    # ------------------------------------------------------------------
    rows = []
    for f in tqdm(sorted((exp_dir / "arguments").glob("*.json")), desc="  Loading"):
        data = safe_load_json(f)
        if data is None:
            continue
        fv = data.get("final_verdict", "Unknown")
        for i, arg in enumerate(data.get("arguments", [])):
            text = arg.get("argument", "")
            if not text:
                continue
            rows.append(dict(
                file          = f.name,
                arg_idx       = i,
                text          = text,
                agent         = arg.get("agent", "Unknown"),
                strength      = normalize_strength(arg.get("type", "Unknown")),
                winner_status = infer_winner(arg, fv),
                final_verdict = fv,
            ))

    if not rows:
        print("  No arguments — skipping.\n")
        continue

    df_all = pd.DataFrame(rows)
    print(f"  Total arguments  : {len(df_all)}")

    # Keep only arguments with known outcome
    df = df_all[df_all["winner_status"] != "Unknown"].copy().reset_index(drop=True)
    print(f"  Known W/L status : {len(df)}")

    if len(df) < 50:
        print("  Too few known-outcome arguments — skipping.\n")
        continue

    # ------------------------------------------------------------------
    # EMBED  (unit-norm → cosine distance == 1 - dot product)
    # ------------------------------------------------------------------
    print("  Embedding arguments …")
    emb = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )   # shape: (N, 768)

    winner_mask = (df["winner_status"] == "Winner").values   # bool array
    loser_mask  = (df["winner_status"] == "Loser").values


    # ==================================================================
    # METHOD 1 — WINNING DIRECTION SPECTRUM
    # ==================================================================
    print("\n  [Method 1] Winning Direction Spectrum …")

    w_centroid = emb[winner_mask].mean(axis=0)
    l_centroid = emb[loser_mask].mean(axis=0)
    w_dir      = w_centroid - l_centroid
    w_dir     /= np.linalg.norm(w_dir)           # unit-length direction

    scores          = emb @ w_dir                 # scalar per argument
    df["win_score"] = scores

    q_labels  = ["Q1 (Loser-like)", "Q2", "Q3", "Q4", "Q5 (Winner-like)"]
    df["quintile"] = pd.qcut(scores, q=5, labels=q_labels)

    # -- 1A: Overlapping score distributions for winners vs losers ------
    fig, ax = plt.subplots(figsize=(11, 6))
    for status, color, lw in [("Winner", "#1565C0", 2.5), ("Loser", "#B71C1C", 2.5)]:
        s = df.loc[df["winner_status"] == status, "win_score"]
        ax.hist(s, bins=80, alpha=0.45, color=color, density=True,
                label=f"{status}  (n={len(s):,})")
        ax.axvline(s.mean(), color=color, lw=lw, linestyle="--",
                   label=f"{status} mean")
    ax.set_xlabel("Projection onto Winning Semantic Direction", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Method 1A — Argument Score along Winner→Loser Semantic Axis\n"
        "Dashed = group mean; separation → semantic direction predicts outcome",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    sns.despine()
    save(fig, out / "m1a_score_distribution.png")

    # -- 1B: Win rate per quintile bar chart ----------------------------
    qs = (
        df.groupby("quintile", observed=True)
        .agg(
            total    = ("winner_status", "count"),
            win_rate = ("winner_status", lambda s: (s == "Winner").mean()),
        )
        .reset_index()
    )

    colors_q = ["#7f0000", "#c62828", "#f9a825", "#2e7d32", "#1b5e20"]
    fig, ax  = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        qs["quintile"].astype(str), qs["win_rate"],
        color=colors_q, edgecolor="white", linewidth=0.8, width=0.6,
    )
    ax.axhline(0.5, color="black", lw=1.4, linestyle="--",
               alpha=0.6, label="50 % baseline")
    for bar, row in zip(bars, qs.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.007,
            f"{row.win_rate*100:.1f}%\n(n={row.total:,})",
            ha="center", va="bottom", fontsize=9.5,
        )
    ax.set_ylim(0, 0.74)
    ax.set_xlabel("Quintile on Winning Semantic Axis", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title(
        "Method 1B — Win Rate per Quintile\n"
        "Q5 = semantically closest to average winner  |  Q1 = closest to average loser",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    sns.despine()
    save(fig, out / "m1b_quintile_winrate.png")

    # -- 1C: Vocabulary contrast Q1 vs Q5 ------------------------------
    q1_texts = df[df["quintile"] == "Q1 (Loser-like)"]["text"].tolist()
    q5_texts = df[df["quintile"] == "Q5 (Winner-like)"]["text"].tolist()

    vc = TfidfVectorizer(
        max_features=5000, stop_words="english",
        ngram_range=(1, 2), min_df=2,
    )
    tmat  = vc.fit_transform([" ".join(q1_texts), " ".join(q5_texts)])
    feats = np.array(vc.get_feature_names_out())

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, idx, title, color in [
        (axes[0], 0, "LOSER-SIDE Vocabulary  (Q1)", "#B71C1C"),
        (axes[1], 1, "WINNER-SIDE Vocabulary  (Q5)", "#1565C0"),
    ]:
        row     = tmat[idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:18]
        terms   = feats[top_idx]
        vals    = row[top_idx]
        ax.barh(terms[::-1], vals[::-1], color=color, alpha=0.82, edgecolor="white")
        ax.set_title(title, fontsize=14, color=color, fontweight="bold")
        ax.set_xlabel("TF-IDF Score (relative to opposite quintile)", fontsize=11)
        sns.despine(ax=ax)
    plt.suptitle(
        "Method 1C — Semantic Vocabulary: What Winning vs Losing Arguments Say",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save(fig, out / "m1c_vocabulary_contrast.png")


    # ==================================================================
    # METHOD 2 — LOG-ODDS CLUSTER ENRICHMENT
    # ==================================================================
    print("\n  [Method 2] Log-Odds Cluster Enrichment …")

    km2       = KMeans(n_clusters=MACRO_K, random_state=RANDOM_STATE, n_init=20)
    df["c2"]  = km2.fit_predict(emb)

    clabels2  = tfidf_top_terms({c: df[df["c2"]==c]["text"].tolist() for c in range(MACRO_K)})

    n_W = int(winner_mask.sum())
    n_L = int(loser_mask.sum())

    lo_rows = []
    for c in range(MACRO_K):
        sub = df[df["c2"] == c]
        nw  = int((sub["winner_status"] == "Winner").sum())
        nl  = int((sub["winner_status"] == "Loser").sum())
        nt  = len(sub)
        # Laplace-smoothed log-odds: avoids log(0) on tiny clusters
        lo  = np.log2(
            ((nw + 0.5) / (n_W + 0.5)) /
            ((nl + 0.5) / (n_L + 0.5))
        )
        lo_rows.append(dict(
            cluster  = c,
            label    = clabels2[c],
            n        = nt,
            win_rate = nw / max(1, nt),
            log_odds = lo,
        ))

    lo_df = pd.DataFrame(lo_rows).sort_values("log_odds")

    fig, ax  = plt.subplots(figsize=(15, 7))
    colors_lo = ["#1565C0" if x > 0 else "#B71C1C" for x in lo_df["log_odds"]]
    bars = ax.barh(
        lo_df["label"], lo_df["log_odds"],
        color=colors_lo, alpha=0.85, edgecolor="white", height=0.6,
    )
    ax.axvline(0, color="black", lw=1.6)
    for bar, row in zip(bars, lo_df.itertuples()):
        w   = bar.get_width()
        off = 0.008 if w >= 0 else -0.008
        ax.text(
            w + off,
            bar.get_y() + bar.get_height() / 2,
            f"WR={row.win_rate*100:.1f}%  |  n={row.n:,}",
            va="center", ha="left" if w >= 0 else "right", fontsize=9,
        )
    ax.set_xlabel(
        "Log₂ Odds Ratio  [ log₂( P(cluster | winner) / P(cluster | loser) ) ]",
        fontsize=12,
    )
    ax.set_title(
        "Method 2 — Semantic Cluster Enrichment: Winners vs Losers\n"
        "► Blue = over-represented in WINNERS  |  ◄ Red = over-represented in LOSERS",
        fontsize=13, pad=12,
    )
    sns.despine()
    save(fig, out / "m2_log_odds_enrichment.png")


    # ==================================================================
    # METHOD 3 — PER-DEBATE ADVANTAGE VECTORS
    # ==================================================================
    print("\n  [Method 3] Per-Debate Advantage Vectors …")

    adv_rows = []
    for fname in df["file"].unique():
        sub  = df[df["file"] == fname]
        wdf  = sub[sub["winner_status"] == "Winner"]
        ldf  = sub[sub["winner_status"] == "Loser"]
        if len(wdf) == 0 or len(ldf) == 0:
            continue
        adv = emb[wdf.index].mean(axis=0) - emb[ldf.index].mean(axis=0)
        adv_rows.append(dict(
            file         = fname,
            adv          = adv,
            nw           = len(wdf),
            nl           = len(ldf),
            winner_texts = wdf["text"].tolist(),
            loser_texts  = ldf["text"].tolist(),
        ))

    if len(adv_rows) < 5:
        print("  Insufficient debates for Method 3 — skipping.")
    else:
        adf     = pd.DataFrame(adv_rows)
        A       = normalize(np.vstack(adf["adv"].values))
        n_ac    = min(6, max(2, len(adf) // 5))
        km3     = KMeans(n_clusters=n_ac, random_state=RANDOM_STATE, n_init=20)
        adf["ac"] = km3.fit_predict(A)

        # Label each advantage-cluster with the winner-side TF-IDF terms
        win_texts_by_ac = {
            ac: [t for ts in adf[adf["ac"] == ac]["winner_texts"] for t in ts]
            for ac in range(n_ac)
        }
        ac_labels = tfidf_top_terms(win_texts_by_ac, n_terms=5)

        # UMAP of advantage vectors
        n_nbrs = min(15, len(adf) - 1)
        u3     = umap.UMAP(
            n_neighbors=n_nbrs, min_dist=0.1, n_components=2,
            metric="cosine", random_state=RANDOM_STATE,
        )
        a2d      = u3.fit_transform(A)
        adf["ux"] = a2d[:, 0]
        adf["uy"] = a2d[:, 1]

        pal = sns.color_palette("Set2", n_ac)
        fig, ax = plt.subplots(figsize=(13, 10))
        for ac in range(n_ac):
            s = adf[adf["ac"] == ac]
            ax.scatter(
                s["ux"], s["uy"], s=70, alpha=0.78, color=pal[ac],
                label=f"Type {ac}: {ac_labels[ac]}", zorder=2,
            )
            cx, cy = s["ux"].mean(), s["uy"].mean()
            ax.annotate(
                f"T{ac}", (cx, cy),
                fontsize=11, fontweight="bold", ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=pal[ac], lw=1.5, alpha=0.9),
            )
        ax.set_xlabel("Advantage Vector UMAP 1", fontsize=12)
        ax.set_ylabel("Advantage Vector UMAP 2", fontsize=12)
        ax.set_title(
            "Method 3 — Types of Winning Semantic Moves\n"
            "Each point = one debate's (winner_centroid − loser_centroid)\n"
            "Clusters = distinct ways winners semantically differ from losers in that debate",
            fontsize=13,
        )
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            fontsize=9, title="Winning Move Type · Top Terms",
            title_fontsize=10,
        )
        sns.despine()
        save(fig, out / "m3a_advantage_vectors.png")

        # Vocabulary contrast for the largest advantage-cluster
        big_ac   = adf.groupby("ac").size().idxmax()
        win_t    = win_texts_by_ac[big_ac]
        lose_t   = [t for ts in adf[adf["ac"] == big_ac]["loser_texts"] for t in ts]

        if win_t and lose_t:
            vc2 = TfidfVectorizer(
                max_features=5000, stop_words="english",
                ngram_range=(1, 2), min_df=1,
            )
            tm2 = vc2.fit_transform([" ".join(win_t), " ".join(lose_t)])
            f2  = np.array(vc2.get_feature_names_out())

            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            for ax, i, title, color in [
                (axes[0], 0, f"WINNER vocabulary — Type {big_ac}", "#1565C0"),
                (axes[1], 1, f"LOSER  vocabulary — Type {big_ac}", "#B71C1C"),
            ]:
                row = tm2[i].toarray().ravel()
                top = row.argsort()[::-1][:16]
                ax.barh(f2[top][::-1], row[top][::-1], color=color, alpha=0.82)
                ax.set_title(title, fontsize=13, color=color, fontweight="bold")
                ax.set_xlabel("TF-IDF Score", fontsize=11)
                sns.despine(ax=ax)
            plt.suptitle(
                f"Method 3B — Vocabulary Contrast: Largest Advantage Cluster (Type {big_ac})",
                fontsize=13, fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            save(fig, out / "m3b_advantage_vocabulary.png")


    # ==================================================================
    # METHOD 4 — WIN PROBABILITY LANDSCAPE (UMAP heatmap)
    # ==================================================================
    print("\n  [Method 4] Win Probability Landscape …")

    nn4    = min(UMAP_NEIGHBORS, len(df) - 1)
    u4     = umap.UMAP(
        n_neighbors=nn4, min_dist=UMAP_MIN_DIST, n_components=2,
        metric="cosine", random_state=RANDOM_STATE,
    )
    coords   = u4.fit_transform(emb)
    df["ux"] = coords[:, 0]
    df["uy"] = coords[:, 1]

    # Fine-grained micro-clusters (in embedding space, not 2D projection)
    n_micro  = min(MICRO_K, len(df) // 5)
    km4      = KMeans(n_clusters=n_micro, random_state=RANDOM_STATE, n_init=10)
    df["mc"] = km4.fit_predict(emb)

    mc_stats = (
        df.groupby("mc")
        .agg(
            total = ("winner_status", "count"),
            wins  = ("winner_status", lambda s: (s == "Winner").sum()),
        )
        .assign(wr=lambda d: d["wins"] / d["total"])
    )
    df["local_wr"] = df["mc"].map(mc_stats["wr"])

    fig, ax = plt.subplots(figsize=(14, 11))
    sc = ax.scatter(
        df["ux"], df["uy"],
        c=df["local_wr"],
        cmap="RdYlBu",
        s=POINT_SIZE, alpha=ALPHA_SCATTER,
        vmin=0.35, vmax=0.65,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Local Win Rate (micro-cluster)", fontsize=11)

    # Annotate extreme micro-clusters (WR > 60% or < 40%)
    for mc_id, mc_row in mc_stats.iterrows():
        if mc_row["wr"] > 0.60 or mc_row["wr"] < 0.40:
            sub2    = df[df["mc"] == mc_id]
            cx, cy  = sub2["ux"].mean(), sub2["uy"].mean()
            color   = "#0D47A1" if mc_row["wr"] > 0.60 else "#B71C1C"
            mc_top  = tfidf_top_terms(
                {0: sub2["text"].tolist()}, n_terms=3
            )[0]
            ax.annotate(
                f"{mc_row['wr']*100:.0f}%\n{mc_top}",
                (cx, cy),
                fontsize=7, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.85, lw=1),
            )

    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(
        "Method 4 — Win Probability Landscape\n"
        "Blue = winner-enriched semantic region  |  Red = loser-enriched region\n"
        "(Annotated: micro-clusters with win rate > 60% or < 40%)",
        fontsize=13,
    )
    sns.despine()
    save(fig, out / "m4_win_landscape.png")

    # ------------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------------
    df_save = df.drop(columns=["c2", "mc"], errors="ignore")
    df_save.to_csv(out / "metadata.csv", index=False)
    print(f"\n  All outputs → {out}\n")


print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")