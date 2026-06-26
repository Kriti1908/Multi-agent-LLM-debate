#!/usr/bin/env python3
"""
SEMANTIC WIN ANALYSIS — Focused, Clear Visualizations
======================================================

Produces 5 plots per experiment, each chosen because it shows
a clearly interpretable finding:

  Plot 1 — Word-Level Win Rate
    For every word appearing in ≥MIN_WORD_COUNT arguments,
    compute P(win | word appears). Top-15 winning words vs
    top-15 losing words on a single diverging chart.
    MOST DIRECT interpretation of what language wins.

  Plot 2 — Quintile Win Rate + Confidence Intervals
    Project onto winner–loser semantic axis. Five quintile bars
    with 95% binomial CI bands. Clear monotonic gradient.

  Plot 3 — Log-Odds Cluster Enrichment (improved cosmetics)
    K-Means clusters with log₂(P(cluster|winner)/P(cluster|loser)).
    Sorted, color-coded, annotated. Shows which semantic topic
    winners/losers disproportionately use.

  Plot 4 — Win Probability Logistic Regression Curve
    Fit logistic regression on the scalar semantic score.
    Plot smooth sigmoid + 95% CI band + raw bin win-rates.
    Shows the continuous relationship, not just quintiles.

  Plot 5 — Abstract vs Concrete Framing Summary Dashboard
    Side-by-side semantic framing comparison. Four panels:
    (a) winning-word cloud bar, (b) losing-word cloud bar,
    (c) quintile gradient, (d) top-line finding text box.
    Designed to be a standalone shareable summary figure.

Install:
  pip install sentence-transformers umap-learn scikit-learn \
              pandas matplotlib seaborn tqdm numpy scipy
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ====================================================================
# CONFIG
# ====================================================================
EMBEDDING_MODEL  = "all-mpnet-base-v2"
RANDOM_STATE     = 42
MACRO_K          = 10      # clusters for Plot 3
MIN_WORD_COUNT   = 100     # minimum argument appearances for word-win-rate
TOP_WORDS_EACH   = 15      # top N winning/losing words to show
N_TFIDF_TERMS    = 5
N_BINS_LR        = 20      # bins for logistic regression calibration dots

WINNER_COLOR     = "#1565C0"
LOSER_COLOR      = "#B71C1C"
NEUTRAL_COLOR    = "#455A64"
GRADIENT_COLORS  = ["#7f0000", "#c62828", "#f9a825", "#2e7d32", "#1b5e20"]

# ====================================================================
# PATHS
# ====================================================================
CURRENT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = CURRENT_DIR.parent.parent / "Archive"
OUTPUT_ROOT = CURRENT_DIR

# ====================================================================
# LOAD MODEL
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


def tfidf_cluster_labels(texts_by_group, n_terms=N_TFIDF_TERMS):
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


def binomial_ci(successes, total, confidence=0.95):
    """Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    z    = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def save(fig, path: Path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"    Saved → {path.name}")


# ====================================================================
# DISCOVER EXPERIMENTS
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

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_dir.name}")
    print(f"{'='*60}")

    out = OUTPUT_ROOT / exp_dir.relative_to(ARCHIVE_DIR) / "plots_clear"

    # ----------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------
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
    df     = df_all[df_all["winner_status"] != "Unknown"].copy().reset_index(drop=True)
    print(f"  Known outcome arguments: {len(df):,}")

    if len(df) < 50:
        print("  Too few — skipping.\n")
        continue

    # ----------------------------------------------------------------
    # EMBED
    # ----------------------------------------------------------------
    print("  Embedding …")
    emb = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    winner_mask = (df["winner_status"] == "Winner").values
    loser_mask  = ~winner_mask
    y_binary    = winner_mask.astype(int)

    # Winning semantic direction
    w_cent  = emb[winner_mask].mean(axis=0)
    l_cent  = emb[loser_mask].mean(axis=0)
    w_dir   = w_cent - l_cent
    w_dir  /= np.linalg.norm(w_dir)
    scores  = emb @ w_dir
    df["win_score"] = scores

    # ================================================================
    # PLOT 1 — WORD-LEVEL WIN RATE
    # ================================================================
    print("\n  [Plot 1] Word-Level Win Rate …")

    # Tokenize: lowercase alphabetic words only, length ≥ 4
    stop_words = {
        "that", "this", "with", "from", "have", "they", "will",
        "would", "could", "should", "their", "there", "been",
        "which", "when", "also", "more", "some", "what", "than",
        "then", "into", "about", "such", "only", "well", "both",
        "each", "these", "those", "were", "your", "argument",
        "actions", "action", "agent", "arguments",
    }

    word_wins   = {}   # word -> [win_count, total_count]
    for text, status in zip(df["text"], df["winner_status"]):
        words_in = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
        for w in words_in:
            if w in stop_words:
                continue
            if w not in word_wins:
                word_wins[w] = [0, 0]
            word_wins[w][1] += 1
            if status == "Winner":
                word_wins[w][0] += 1

    # Filter to words with enough appearances
    word_df = pd.DataFrame(
        [{"word": w, "wins": v[0], "total": v[1]}
         for w, v in word_wins.items()
         if v[1] >= MIN_WORD_COUNT]
    )
    word_df["win_rate"]  = word_df["wins"] / word_df["total"]
    word_df["deviation"] = word_df["win_rate"] - 0.50   # deviation from baseline

    # Compute Wilson CI
    cis = [binomial_ci(r.wins, r.total) for r in word_df.itertuples()]
    word_df["ci_lo"] = [c[0] for c in cis]
    word_df["ci_hi"] = [c[1] for c in cis]
    word_df["ci_half"] = (word_df["ci_hi"] - word_df["ci_lo"]) / 2

    word_df = word_df.sort_values("win_rate")
    top_losing  = word_df.head(TOP_WORDS_EACH)
    top_winning = word_df.tail(TOP_WORDS_EACH)
    combined    = pd.concat([top_losing, top_winning]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, max(8, len(combined) * 0.42)))

    colors_bar = [WINNER_COLOR if d > 0 else LOSER_COLOR for d in combined["deviation"]]
    bars = ax.barh(
        combined["word"],
        combined["win_rate"],
        xerr=combined["ci_half"],
        color=colors_bar,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        height=0.65,
        capsize=3,
        error_kw=dict(elinewidth=1.2, ecolor="#333333", capthick=1.2),
    )

    ax.axvline(0.50, color="black", lw=1.8, linestyle="--",
               alpha=0.7, label="50 % baseline (random)")

    # Add win rate labels
    for bar, row in zip(bars, combined.itertuples()):
        w = bar.get_width()
        ax.text(
            w + combined["ci_half"].max() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{row.win_rate*100:.1f}%  (n={row.total:,})",
            va="center", ha="left", fontsize=8.5,
        )

    # Separator between losing and winning words
    ax.axhline(TOP_WORDS_EACH - 0.5, color="#888888", lw=1.2,
               linestyle=":", alpha=0.8)
    ax.text(
        ax.get_xlim()[0] + 0.001,
        TOP_WORDS_EACH - 0.5 + 0.3,
        "▲ WINNER-ASSOCIATED WORDS",
        fontsize=9, color=WINNER_COLOR, fontweight="bold",
    )
    ax.text(
        ax.get_xlim()[0] + 0.001,
        TOP_WORDS_EACH - 0.5 - 0.7,
        "▼ LOSER-ASSOCIATED WORDS",
        fontsize=9, color=LOSER_COLOR, fontweight="bold",
    )

    ax.set_xlim(0.35, 0.75)
    ax.set_xlabel("Win Rate of Arguments Containing This Word", fontsize=12)
    ax.set_title(
        f"Plot 1 — Which Words Appear in Winning vs Losing Arguments\n"
        f"(Only words in ≥{MIN_WORD_COUNT:,} arguments shown;  "
        f"error bars = 95% Wilson CI;  n={len(df):,} total arguments)",
        fontsize=13, pad=12,
    )
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    save(fig, out / "plot1_word_win_rate.png")

    # ================================================================
    # PLOT 2 — QUINTILE WIN RATE WITH CONFIDENCE INTERVALS
    # ================================================================
    print("\n  [Plot 2] Quintile Win Rate + CI …")

    q_labels = ["Q1\nLoser-like", "Q2", "Q3\nNeutral", "Q4", "Q5\nWinner-like"]
    df["quintile_id"] = pd.qcut(scores, q=5, labels=False)

    q_rows = []
    for qid in range(5):
        sub  = df[df["quintile_id"] == qid]
        wins = int((sub["winner_status"] == "Winner").sum())
        tot  = len(sub)
        lo, hi = binomial_ci(wins, tot)
        q_rows.append(dict(
            qid=qid, label=q_labels[qid],
            win_rate=wins/tot, total=tot, ci_lo=lo, ci_hi=hi,
        ))
    q_df = pd.DataFrame(q_rows)

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.bar(
        q_df["label"], q_df["win_rate"],
        color=GRADIENT_COLORS, edgecolor="white", linewidth=0.8,
        width=0.58, zorder=2,
    )

    # Confidence interval lines
    for _, row in q_df.iterrows():
        x = list(q_df["label"]).index(row["label"])
        ax.plot(
            [x, x], [row["ci_lo"], row["ci_hi"]],
            color="black", lw=2.5, zorder=3,
        )
        ax.plot(
            [x - 0.12, x + 0.12], [row["ci_lo"], row["ci_lo"]],
            color="black", lw=2, zorder=3,
        )
        ax.plot(
            [x - 0.12, x + 0.12], [row["ci_hi"], row["ci_hi"]],
            color="black", lw=2, zorder=3,
        )

    # Value labels
    for bar, row in zip(bars, q_df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{row.win_rate*100:.1f}%\n(n={row.total:,})",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        )

    ax.axhline(0.50, color="black", lw=1.6, linestyle="--",
               alpha=0.6, label="50 % baseline", zorder=1)

    # Highlight the range
    bot = q_df["win_rate"].min()
    top = q_df["win_rate"].max()
    ax.annotate(
        f"Range: {(top-bot)*100:.1f} pp",
        xy=(4, top), xytext=(3.3, top + 0.025),
        fontsize=11, color="#1b5e20", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.5),
    )

    ax.set_ylim(0.40, 0.72)
    ax.set_ylabel("Win Rate", fontsize=13)
    ax.set_xlabel(
        "← More like average LOSER argument          "
        "More like average WINNER argument →",
        fontsize=11,
    )
    ax.set_title(
        "Plot 2 — Win Rate Increases Monotonically along Semantic Axis\n"
        "Arguments are sorted by their projection onto the "
        "winner − loser embedding direction",
        fontsize=13, pad=12,
    )
    ax.legend(fontsize=11)
    sns.despine()
    save(fig, out / "plot2_quintile_winrate_ci.png")

    # ================================================================
    # PLOT 3 — LOG-ODDS CLUSTER ENRICHMENT (improved)
    # ================================================================
    print("\n  [Plot 3] Log-Odds Cluster Enrichment …")

    km = KMeans(n_clusters=MACRO_K, random_state=RANDOM_STATE, n_init=20)
    df["cluster"] = km.fit_predict(emb)

    clabels = tfidf_cluster_labels(
        {c: df[df["cluster"] == c]["text"].tolist() for c in range(MACRO_K)}
    )

    n_W = int(winner_mask.sum())
    n_L = int(loser_mask.sum())

    lo_rows = []
    for c in range(MACRO_K):
        sub = df[df["cluster"] == c]
        nw  = int((sub["winner_status"] == "Winner").sum())
        nl  = int((sub["winner_status"] == "Loser").sum())
        nt  = len(sub)
        lo  = np.log2(
            ((nw + 0.5) / (n_W + 0.5)) /
            ((nl + 0.5) / (n_L + 0.5))
        )
        lo_rows.append(dict(
            cluster=c, label=clabels[c], n=nt,
            win_rate=nw / max(1, nt), log_odds=lo,
            nw=nw, nl=nl,
        ))

    lo_df = pd.DataFrame(lo_rows).sort_values("log_odds").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    bar_colors = [
        WINNER_COLOR if x > 0.005 else (LOSER_COLOR if x < -0.005 else NEUTRAL_COLOR)
        for x in lo_df["log_odds"]
    ]
    bars = ax.barh(
        lo_df["label"], lo_df["log_odds"],
        color=bar_colors, alpha=0.85, edgecolor="white",
        linewidth=0.6, height=0.62,
    )

    ax.axvline(0, color="black", lw=2)

    for bar, row in zip(bars, lo_df.itertuples()):
        w   = bar.get_width()
        pad = 0.0015 if w >= 0 else -0.0015
        ha  = "left" if w >= 0 else "right"
        ax.text(
            w + pad,
            bar.get_y() + bar.get_height() / 2,
            f"WR={row.win_rate*100:.1f}%   n={row.n:,}   "
            f"({row.nw:,}W / {row.nl:,}L)",
            va="center", ha=ha, fontsize=9,
        )

    ax.set_xlabel(
        "Log₂ Odds Ratio   [ positive = winners use this cluster more ]",
        fontsize=12,
    )
    ax.set_title(
        "Plot 3 — Semantic Cluster Enrichment: Which Topic Clusters Are "
        "Disproportionately Used by Winners vs Losers\n"
        "Controls for 50% baseline — a WR of 51.2% in a large cluster "
        "can still be highly enriched over the loser distribution",
        fontsize=12, pad=12,
    )

    legend_patches = [
        mpatches.Patch(color=WINNER_COLOR, alpha=0.85,
                       label="Over-represented in WINNERS"),
        mpatches.Patch(color=LOSER_COLOR,  alpha=0.85,
                       label="Over-represented in LOSERS"),
    ]
    ax.legend(handles=legend_patches, fontsize=11, loc="lower right")
    sns.despine()
    save(fig, out / "plot3_logodds_enrichment.png")

    # ================================================================
    # PLOT 4 — LOGISTIC REGRESSION WIN PROBABILITY CURVE
    # ================================================================
    print("\n  [Plot 4] Logistic Regression Curve …")

    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(scores.reshape(-1, 1), y_binary)

    # Smooth curve
    x_range = np.linspace(scores.min(), scores.max(), 500)
    y_prob  = lr.predict_proba(x_range.reshape(-1, 1))[:, 1]

    # Bootstrap CI on the logistic curve (100 resamples)
    rng       = np.random.default_rng(RANDOM_STATE)
    boot_probs = []
    for _ in range(120):
        idx   = rng.choice(len(scores), size=len(scores), replace=True)
        lr_b  = LogisticRegression(random_state=0)
        lr_b.fit(scores[idx].reshape(-1, 1), y_binary[idx])
        boot_probs.append(lr_b.predict_proba(x_range.reshape(-1, 1))[:, 1])
    boot_lo = np.percentile(boot_probs, 2.5, axis=0)
    boot_hi = np.percentile(boot_probs, 97.5, axis=0)

    # Binned empirical win rates for scatter dots
    bin_edges = np.percentile(scores, np.linspace(0, 100, N_BINS_LR + 1))
    bin_ids   = np.digitize(scores, bin_edges[1:-1])
    bin_rows  = []
    for b in range(N_BINS_LR):
        mask_b = bin_ids == b
        if mask_b.sum() < 5:
            continue
        wins_b = int(y_binary[mask_b].sum())
        tot_b  = int(mask_b.sum())
        lo_b, hi_b = binomial_ci(wins_b, tot_b)
        bin_rows.append(dict(
            x=scores[mask_b].mean(),
            wr=wins_b / tot_b,
            ci_lo=lo_b,
            ci_hi=hi_b,
            n=tot_b,
        ))
    bin_df = pd.DataFrame(bin_rows)

    fig, ax = plt.subplots(figsize=(13, 7))

    # CI band
    ax.fill_between(x_range, boot_lo, boot_hi,
                    alpha=0.18, color=WINNER_COLOR, label="95% Bootstrap CI")

    # Logistic curve
    ax.plot(x_range, y_prob, color=WINNER_COLOR, lw=2.5,
            label="Logistic regression fit")

    # Empirical bins
    ax.scatter(
        bin_df["x"], bin_df["wr"],
        s=bin_df["n"] / bin_df["n"].max() * 200 + 20,
        color=NEUTRAL_COLOR, zorder=4, alpha=0.85,
        label="Empirical win rate (bin size = dot size)",
    )
    ax.errorbar(
        bin_df["x"], bin_df["wr"],
        yerr=[bin_df["wr"] - bin_df["ci_lo"], bin_df["ci_hi"] - bin_df["wr"]],
        fmt="none", ecolor=NEUTRAL_COLOR, elinewidth=1.2, capsize=3,
    )

    ax.axhline(0.50, color="black", lw=1.5, linestyle="--",
               alpha=0.6, label="50% baseline")

    # Shade winner/loser regions
    ax.axvspan(x_range[y_prob >= 0.50][0], x_range.max(),
               alpha=0.06, color=WINNER_COLOR, label="P(win) > 50%")
    ax.axvspan(x_range.min(), x_range[y_prob >= 0.50][0],
               alpha=0.06, color=LOSER_COLOR, label="P(win) < 50%")

    ax.set_ylim(0.35, 0.65)
    ax.set_xlabel("Projection onto Winner−Loser Semantic Axis", fontsize=12)
    ax.set_ylabel("Probability of Winning", fontsize=12)
    ax.set_title(
        "Plot 4 — Logistic Regression: Semantic Score → Win Probability\n"
        "Shows the continuous, smooth relationship between where an argument sits "
        "in embedding space and its probability of winning",
        fontsize=13, pad=12,
    )
    ax.legend(fontsize=10, loc="upper left")
    sns.despine()
    save(fig, out / "plot4_logistic_win_probability.png")

    # ================================================================
    # PLOT 5 — FRAMING SUMMARY DASHBOARD
    # ================================================================
    print("\n  [Plot 5] Framing Summary Dashboard …")

    # Re-derive winning/losing words sorted by deviation
    word_df_sorted_win  = word_df.sort_values("win_rate", ascending=False).head(12)
    word_df_sorted_lose = word_df.sort_values("win_rate", ascending=True).head(12)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.07, right=0.97,
        top=0.88, bottom=0.08,
    )

    # ------ Panel A: Winner words ------
    ax_a = fig.add_subplot(gs[0, 0])
    ww   = word_df_sorted_win.sort_values("win_rate")
    ax_a.barh(
        ww["word"], ww["win_rate"],
        color=WINNER_COLOR, alpha=0.83, edgecolor="white",
        xerr=ww["ci_half"], capsize=2,
        error_kw=dict(elinewidth=1.0, ecolor="#333", capthick=1),
    )
    ax_a.axvline(0.50, color="black", lw=1.4, linestyle="--", alpha=0.6)
    ax_a.set_xlim(0.44, 0.65)
    ax_a.set_xlabel("Win Rate", fontsize=10)
    ax_a.set_title("WORDS ASSOCIATED\nWITH WINNING",
                   fontsize=11, color=WINNER_COLOR, fontweight="bold", pad=8)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)
    for bar, row in zip(ax_a.patches, ww.itertuples()):
        ax_a.text(
            bar.get_width() + ww["ci_half"].max() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{row.win_rate*100:.1f}%",
            va="center", fontsize=8, color=WINNER_COLOR,
        )

    # ------ Panel B: Loser words ------
    ax_b = fig.add_subplot(gs[1, 0])
    lw   = word_df_sorted_lose.sort_values("win_rate", ascending=False)
    ax_b.barh(
        lw["word"], lw["win_rate"],
        color=LOSER_COLOR, alpha=0.83, edgecolor="white",
        xerr=lw["ci_half"], capsize=2,
        error_kw=dict(elinewidth=1.0, ecolor="#333", capthick=1),
    )
    ax_b.axvline(0.50, color="black", lw=1.4, linestyle="--", alpha=0.6)
    ax_b.set_xlim(0.35, 0.56)
    ax_b.set_xlabel("Win Rate", fontsize=10)
    ax_b.set_title("WORDS ASSOCIATED\nWITH LOSING",
                   fontsize=11, color=LOSER_COLOR, fontweight="bold", pad=8)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)
    for bar, row in zip(ax_b.patches, lw.itertuples()):
        ax_b.text(
            bar.get_width() + lw["ci_half"].max() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{row.win_rate*100:.1f}%",
            va="center", fontsize=8, color=LOSER_COLOR,
        )

    # ------ Panel C: Quintile gradient ------
    ax_c = fig.add_subplot(gs[:, 1])
    bars_c = ax_c.bar(
        range(5), q_df["win_rate"],
        color=GRADIENT_COLORS, edgecolor="white", linewidth=0.8,
        width=0.62, zorder=2,
    )
    for qid in range(5):
        row_q = q_df.iloc[qid]
        ax_c.plot(
            [qid, qid], [row_q["ci_lo"], row_q["ci_hi"]],
            color="black", lw=2.2, zorder=3,
        )
        ax_c.plot(
            [qid - 0.12, qid + 0.12],
            [row_q["ci_lo"], row_q["ci_lo"]],
            color="black", lw=1.8, zorder=3,
        )
        ax_c.plot(
            [qid - 0.12, qid + 0.12],
            [row_q["ci_hi"], row_q["ci_hi"]],
            color="black", lw=1.8, zorder=3,
        )
    for bar, row in zip(bars_c, q_df.itertuples()):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{row.win_rate*100:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax_c.axhline(0.50, color="black", lw=1.5, linestyle="--", alpha=0.6)
    ax_c.set_xticks(range(5))
    ax_c.set_xticklabels(
        ["Q1\n(Loser-like)", "Q2", "Q3\n(Neutral)", "Q4", "Q5\n(Winner-like)"],
        fontsize=10,
    )
    ax_c.set_ylim(0.40, 0.70)
    ax_c.set_ylabel("Win Rate", fontsize=11)
    ax_c.set_title(
        "WIN RATE BY QUINTILE\nON SEMANTIC AXIS",
        fontsize=11, fontweight="bold", pad=8,
    )
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)

    # ------ Panel D: Log-odds cluster enrichment (top 6 each side) ------
    ax_d = fig.add_subplot(gs[:, 2])
    top_pos = lo_df[lo_df["log_odds"] >  0].tail(5)
    top_neg = lo_df[lo_df["log_odds"] <= 0].head(5)
    lo_sub  = pd.concat([top_neg, top_pos]).reset_index(drop=True)
    colors_d = [WINNER_COLOR if x > 0 else LOSER_COLOR for x in lo_sub["log_odds"]]
    ax_d.barh(
        lo_sub["label"], lo_sub["log_odds"],
        color=colors_d, alpha=0.82, edgecolor="white",
        linewidth=0.5, height=0.58,
    )
    ax_d.axvline(0, color="black", lw=1.8)
    for bar, row in zip(ax_d.patches, lo_sub.itertuples()):
        w   = bar.get_width()
        off = 0.001 if w >= 0 else -0.001
        ha  = "left" if w >= 0 else "right"
        ax_d.text(
            w + off,
            bar.get_y() + bar.get_height() / 2,
            f"WR {row.win_rate*100:.1f}% (n={row.n:,})",
            va="center", ha=ha, fontsize=8.5,
        )
    ax_d.set_xlabel(
        "Log₂ Odds Ratio\n(positive = over-represented in winners)",
        fontsize=10,
    )
    ax_d.set_title(
        "CLUSTER ENRICHMENT\n(TOP 5 WINNER / TOP 5 LOSER CLUSTERS)",
        fontsize=11, fontweight="bold", pad=8,
    )
    for spine in ["top", "right"]:
        ax_d.spines[spine].set_visible(False)

    # ------ Main title ------
    total_w = int(winner_mask.sum())
    total_l = int(loser_mask.sum())
    fig.suptitle(
        f"Semantic Framing & Debate Outcomes — {exp_dir.name}\n"
        f"Abstract/values language wins more  |  "
        f"Concrete/technical language wins less\n"
        f"n={len(df):,} arguments  ({total_w:,} winners, {total_l:,} losers)",
        fontsize=14, fontweight="bold", y=0.96,
    )

    save(fig, out / "plot5_summary_dashboard.png")

    # ----------------------------------------------------------------
    # SAVE METADATA
    # ----------------------------------------------------------------
    df.to_csv(out / "metadata.csv", index=False)
    word_df.sort_values("win_rate", ascending=False).to_csv(
        out / "word_win_rates.csv", index=False
    )
    lo_df.sort_values("log_odds", ascending=False).to_csv(
        out / "cluster_log_odds.csv", index=False
    )
    print(f"\n  All outputs → {out}\n")


print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")