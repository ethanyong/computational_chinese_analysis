"""Step 7 — Desk divergence and convergence (research questions Q1 and Q2).

1. Distinguishability (primary measure).  For every year and pair of desks, an L2-regularized
   logistic regression is trained to tell the two desks' articles apart.  Score = out-of-fold
   ROC AUC from 5-fold stratified cross-validation repeated 3 times (scores averaged per article).
   Every cell uses at most 150 randomly drawn articles per desk, so training size (which
   raises AUC mechanically) is equal across years; cells below the cap are flagged.  Because
   one random draw makes the AUC depend on which articles were drawn, every cell averages 10
   independent draws.
   AUC = 0.5: the desks' prose is indistinguishable; AUC = 1: perfectly separable.
     * Style space: relative frequencies of the 300 most frequent function words (adverbs,
       conjunctions, prepositions, particles; pronouns and localizers are excluded because
       they leak topic) — "how it is written" with topic largely removed, and with no
       hand-picked word list.
     * Full-vocabulary space: TF-IDF over the 5,000 most frequent words — topic + style.
   The regularization strength is chosen once (pooled CV) and then held fixed so that AUCs
   are comparable across years; standardization is fitted inside each training fold to
   avoid leakage.  95% CIs: percentile bootstrap of the out-of-fold predictions (200 resamples
   per draw, bounds averaged over draws); they reflect test-set sampling only, not refitting,
   so they are slightly narrow.
2. Convergence test.  If the China desk converged on the Singapore desk after the
   restructuring, AUC(China vs Singapore) should fall and AUC(China vs World) rise after 2021.
   Tested at the year level (Welch t-test of yearly AUCs, post 2022-25 vs pre years from 2013
   (realtime) or 2016 (print desks) to 2021).
3. Unbiased centroid distance (robustness).  Squared Euclidean distance between desk mean
   vectors in z-scored style space, bias-corrected:  ||m_A - m_B||^2 - tr(S_A)/n_A - tr(S_B)/n_B
   (expected value 0 when the desks do not differ, unlike the raw distance, which grows with
   sampling noise).
4. Cosine diagnostic: why a cosine between low-dimensional marker-rate vectors is not a usable
   distance (it is ~0.999 even between two halves of one desk), next to a full-vocabulary
   TF-IDF cosine and its split-half same-desk reference.
5. PCA of desk-year centroids in style space (realtime corpus), explained variance reported.

Corpus roles: the realtime corpus (free, full text, 2013-2025) is primary for the modern era.
Print-desk full texts after 2016 survive in the archive mainly for free (non-paywalled)
articles, a self-selected subset, so those years are shown dashed and interpreted cautiously.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from config import INTERIM, FEATURES, RESULTS, SEED
from plotstyle import DESK_COLOR, DESK_MARKER, DESK_LABEL, INK2, events, save

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(SEED)
PAIRS = [("china", "world"), ("china", "singapore"), ("china", "sea"),
         ("world", "singapore"), ("world", "sea"), ("singapore", "sea")]
N_BOOT = 200     # bootstrap resamples per draw
N_CAP = 150     # articles per desk per cell
N_DRAWS = 10    # independent random subsamples per cell, averaged


def load():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    art = art[art.fulltext_ok]
    fw = pd.read_parquet(f"{FEATURES}/function_words.parquet")
    tok = pd.read_parquet(f"{FEATURES}/tokens.parquet", columns=["article_id", "tokens", "n_tok"])
    mk = pd.read_parquet(f"{FEATURES}/markers.parquet",
                         columns=["article_id", "hedge", "attribution", "obligation", "permission"])
    df = art.merge(tok, on="article_id").merge(mk, on="article_id")
    df = df[df.n_tok >= 100].reset_index(drop=True)
    fw = fw.set_index("article_id").loc[df.article_id]
    return df, fw.values.astype(float), list(fw.columns)


def oof_scores(X, y, C, reps=3, sparse=False):
    s = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(X, y):
            scaler = MaxAbsScaler() if sparse else StandardScaler()
            clf = make_pipeline(scaler, LogisticRegression(C=C, max_iter=3000))
            clf.fit(X[tr], y[tr])
            s[te] += clf.decision_function(X[te]) / reps
    return s


def boot_auc(y, s):
    n = len(y)
    out = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def choose_C(X, y, sparse=False):
    best = None
    for C in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        auc = roc_auc_score(y, oof_scores(X, y, C, reps=1, sparse=sparse))
        print(f"      C={C}: pooled CV AUC={auc:.3f}")
        if best is None or auc > best[1]:
            best = (C, auc)
    return best[0]


def unbiased_dist(A, B):
    ma, mb = A.mean(0), B.mean(0)
    return float(((ma - mb) ** 2).sum() - A.var(0, ddof=1).sum() / len(A) - B.var(0, ddof=1).sum() / len(B))


def distinguishability(df, Xfw, Xtf):
    print("Choosing regularization (pooled, China vs World) ...")
    cw = df.desk.isin(["china", "world"]).values
    sub = np.where(cw)[0]
    sub = RNG.choice(sub, min(6000, len(sub)), replace=False)
    yy = (df.desk.values[sub] == "china").astype(int)
    C_fw = choose_C(Xfw[sub], yy)
    C_tf = choose_C(Xtf[sub], yy, sparse=True)
    print(f"   fixed C: style={C_fw}, full-vocabulary={C_tf}")
    # step 10 reuses the style-space value, so the directional test uses the same classifier
    pd.DataFrame({"space": ["style", "fullvocab"], "C": [C_fw, C_tf]}).to_csv(f"{RESULTS}/classifier_C.csv", index=False)

    Z = (Xfw - Xfw.mean(0)) / (Xfw.std(0) + 1e-12)
    rows = []
    for (corpus, year), g in df.groupby(["corpus", "pub_year"]):
        for a, b in PAIRS:
            # equal training size in every cell: a classifier's AUC grows with n, so without a
            # cap, years/desks with more articles would look "more distinguishable"
            ia_all, ib_all = g.index[g.desk == a].values, g.index[g.desk == b].values
            if len(ia_all) < 40 or len(ib_all) < 40:
                continue
            row = dict(corpus=corpus, year=year, pair=f"{a}-{b}", n_a=min(N_CAP, len(ia_all)),
                       n_b=min(N_CAP, len(ib_all)), below_cap=bool(min(len(ia_all), len(ib_all)) < N_CAP))
            # one random subsample of <= N_CAP per desk makes the AUC depend on the draw (the
            # Monte-Carlo spread is ~0.02-0.03), so each cell averages N_DRAWS independent draws;
            # the CI is the mean of the per-draw bootstrap bounds
            for name, X, C, sp in [("style", Xfw, C_fw, False), ("fullvocab", Xtf, C_tf, True)]:
                if name == "fullvocab" and not a == "china":
                    continue
                aucs, los, his = [], [], []
                for _ in range(N_DRAWS):
                    ia = RNG.choice(ia_all, min(N_CAP, len(ia_all)), replace=False)
                    ib = RNG.choice(ib_all, min(N_CAP, len(ib_all)), replace=False)
                    idx = np.concatenate([ia, ib])
                    y = (df.desk.values[idx] == a).astype(int)
                    s = oof_scores(X[idx], y, C, sparse=sp)
                    lo, hi = boot_auc(y, s)
                    aucs.append(roc_auc_score(y, s)); los.append(lo); his.append(hi)
                row.update({f"auc_{name}": np.mean(aucs), f"auc_{name}_lo": np.mean(los),
                            f"auc_{name}_hi": np.mean(his), f"auc_{name}_draw_sd": np.std(aucs, ddof=1)})
            # the bias-corrected distance does not grow with n, so it uses every article in the cell
            idx = np.concatenate([ia_all, ib_all])
            y = (df.desk.values[idx] == a).astype(int)
            A, B = Z[idx][y == 1], Z[idx][y == 0]
            row["dist_unbiased"] = unbiased_dist(A, B)
            bs = []
            for _ in range(300):
                bs.append(unbiased_dist(A[RNG.integers(0, len(A), len(A))], B[RNG.integers(0, len(B), len(B))]))
            row["dist_lo"], row["dist_hi"] = np.percentile(bs, [2.5, 97.5])
            rows.append(row)
        print(f"   {corpus} {year} done", flush=True)
    return pd.DataFrame(rows)


def convergence_tests(res):
    out = []
    for corpus in ["print", "realtime"]:
        for pair in ["china-world", "china-singapore", "china-sea", "world-singapore", "world-sea"]:
            for metric in ["auc_style", "auc_fullvocab", "dist_unbiased"]:
                if metric not in res:
                    continue
                s = res[(res.corpus == corpus) & (res.pair == pair) & (res.year >= 2013)].dropna(subset=[metric])
                pre, post = s[s.year <= 2021][metric], s[s.year >= 2022][metric]
                if len(pre) < 3 or len(post) < 2:
                    continue
                t = stats.ttest_ind(post, pre, equal_var=False)
                out.append(dict(corpus=corpus, pair=pair, metric=metric, pre_mean=pre.mean(), post_mean=post.mean(),
                                change=post.mean() - pre.mean(), p_welch=t.pvalue, n_pre=len(pre), n_post=len(post)))
    return pd.DataFrame(out)


def cosine_diagnostic(df, Xtf):
    """4-marker-rate cosine vs. full-vocabulary cosine, each with a noise-matched reference.

    A cosine between two sample centroids is biased downward by sampling noise, so neither its
    raw value nor a percentile bootstrap is a fair yardstick.  Instead each of 200 random splits
    compares (half of China vs half of World) with (half of China vs the other half of China):
    both cosines are computed from equally sized samples, so their difference isolates the real
    desk difference.  Reported: mean of each over splits and the 2.5/97.5 percentiles.
    """
    rows = []
    feats = ["hedge", "attribution", "obligation", "permission"]
    R = (df[feats].div(df.n_tok, axis=0) * 1000).values
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    m4 = lambda i: R[i].mean(0)
    mtf = lambda i: np.asarray(Xtf[i].mean(0)).ravel()
    for year, g in df[df.corpus == "realtime"].groupby("pub_year"):
        c, w = g.index[g.desk == "china"].values, g.index[g.desk == "world"].values
        if len(c) < 40 or len(w) < 40:
            continue
        r = dict(year=year, cos4_full=cos(m4(c), m4(w)), cos_tfidf_full=cos(mtf(c), mtf(w)))
        out = {k: [] for k in ["b4", "w4", "btf", "wtf"]}
        h = min(len(c), len(w)) // 2
        for _ in range(200):
            pc, pw = RNG.permutation(c), RNG.permutation(w)
            c1, c2, w1 = pc[:h], pc[h:2 * h], pw[:h]
            out["b4"].append(cos(m4(c1), m4(w1)))
            out["w4"].append(cos(m4(c1), m4(c2)))
            out["btf"].append(cos(mtf(c1), mtf(w1)))
            out["wtf"].append(cos(mtf(c1), mtf(c2)))
        for k, v in out.items():
            r[f"{k}_mean"], r[f"{k}_lo"], r[f"{k}_hi"] = np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
        r["diff_tfidf_mean"] = np.mean(np.array(out["wtf"]) - np.array(out["btf"]))
        r["diff4_mean"] = np.mean(np.array(out["w4"]) - np.array(out["b4"]))
        rows.append(r)
    return pd.DataFrame(rows)


def pca_centroids(df, Xfw, words):
    """PCA of desk-year centroids in z-scored function-word space (realtime corpus, 2013-2025)."""
    d = df[df.corpus == "realtime"]
    Z = (Xfw - Xfw[d.index].mean(0)) / (Xfw[d.index].std(0) + 1e-12)
    cen = pd.DataFrame(Z[d.index], index=d.index).groupby([d.desk, d.pub_year]).mean()
    pca = PCA(2).fit(cen.values)
    P = pd.DataFrame(pca.transform(cen.values), index=cen.index, columns=["PC1", "PC2"]).reset_index()
    load = pd.DataFrame(pca.components_.T, index=words, columns=["PC1", "PC2"])
    return P, pca.explained_variance_ratio_, load


# ------------------------------------------------------------------ figures
def fig_auc(res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    panels = [("style", "realtime", "Realtime — function words only (style)"),
              ("fullvocab", "realtime", "Realtime — all words (topic + style)"),
              ("style", "print", "Print desks — function words only (style)")]
    for ax, (feat, corpus, title) in zip(axes, panels):
        for pair, desk in [("china-world", "world"), ("china-singapore", "singapore"), ("china-sea", "sea")]:
            s = res[(res.corpus == corpus) & (res.pair == pair)].dropna(subset=[f"auc_{feat}"])
            if s.empty:
                continue
            for part, ls in ((s[s.year <= 2012], "-"), (s[s.year >= 2013], "-" if corpus == "realtime" else "--")):
                if part.empty:
                    continue
                ax.plot(part.year, part[f"auc_{feat}"], color=DESK_COLOR[desk], marker=DESK_MARKER[desk], lw=1.6, ls=ls,
                        label=f"China vs {DESK_LABEL[desk].replace(' desk', '')}" if part.year.max() >= 2016 else None)
                ax.fill_between(part.year, part[f"auc_{feat}_lo"], part[f"auc_{feat}_hi"], color=DESK_COLOR[desk], alpha=.12, lw=0)
        ax.axhline(0.5, color=INK2, lw=1)
        ax.set_title(title, fontsize=10)
        events(ax, label=False)
        if corpus == "print":
            ax.axvspan(2012.5, 2015.5, color="#f0efec", zorder=0)
            ax.text(2016.2, 0.47, "dashed: free (non-paywalled) articles only", fontsize=7.5, color=INK2)
    axes[0].text(2013, 0.505, "0.5 = indistinguishable", fontsize=7.5, color=INK2, va="bottom")
    axes[0].set_ylabel("Cross-validated AUC")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 3. How distinguishable is China-desk writing from other desks? (bands = 95% bootstrap CI; "
                 "vertical rules = ThinkChina, SPH restructuring, WaPo report)", x=.01, ha="left")
    save(fig, "fig03_desk_distinguishability_auc")


def fig_cosine(cd):
    """Content similarity: full-vocabulary (TF-IDF) cosine between half of the China desk and
    half of the World desk, against two halves of the China desk as a same-desk reference.
    (The 4-marker-rate cosine is kept in results/cosine_diagnostic.csv only.)"""
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ax.plot(cd.year, cd["btf_mean"], color="#2a78d6", marker="o", label="Half of China desk vs half of World desk")
    ax.fill_between(cd.year, cd["btf_lo"], cd["btf_hi"], color="#2a78d6", alpha=.15, lw=0)
    ax.plot(cd.year, cd["wtf_mean"], color=INK2, marker="s", ms=4, lw=1.4, label="Two halves of China desk (reference)")
    ax.fill_between(cd.year, cd["wtf_lo"], cd["wtf_hi"], color=INK2, alpha=.10, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cosine similarity (TF-IDF)")
    ax.legend(fontsize=8, loc="lower left")
    events(ax, label=False)
    fig.suptitle("Figure 2. Content similarity between the China and World desks, realtime corpus "
                 "(bands = 95% range over 200 random splits)", x=.01, ha="left", fontsize=10)
    save(fig, "fig02_content_similarity")


def fig_pca(P, evr):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for desk, g in P.groupby("desk"):
        g = g.sort_values("pub_year")
        ax.plot(g.PC1, g.PC2, color=DESK_COLOR[desk], lw=1, alpha=.6)
        ax.scatter(g.PC1, g.PC2, color=DESK_COLOR[desk], marker=DESK_MARKER[desk], s=28, label=DESK_LABEL[desk], zorder=3)
        for _, r in g[g.pub_year.isin([2013, 2021, 2025])].iterrows():
            ax.annotate(str(r.pub_year), (r.PC1, r.PC2), fontsize=7.5, color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variance between desk-years)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.legend(fontsize=8)
    ax.set_title("Figure S1. Desk-year style centroids, 2013-2025 (function words, realtime corpus)", loc="left", fontsize=10)
    save(fig, "figS1_style_pca_trajectories")


def main():
    df, Xfw, words = load()
    print(f"{len(df):,} full-text articles, {Xfw.shape[1]} function words")
    tf = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None,
                         max_features=5000, sublinear_tf=True, min_df=5)
    Xtf = tf.fit_transform(df.tokens)

    res = distinguishability(df, Xfw, Xtf)
    res.to_csv(f"{RESULTS}/divergence_by_year.csv", index=False)
    conv = convergence_tests(res)
    conv.to_csv(f"{RESULTS}/convergence_tests.csv", index=False)
    cd = cosine_diagnostic(df, Xtf)
    cd.to_csv(f"{RESULTS}/cosine_diagnostic.csv", index=False)
    P, evr, load_ = pca_centroids(df, Xfw, words)
    P.to_csv(f"{RESULTS}/pca_desk_year_centroids.csv", index=False)
    load_.to_csv(f"{RESULTS}/pca_loadings.csv")

    fig_auc(res)
    fig_cosine(cd)
    fig_pca(P, evr)

    pd.set_option("display.width", 220)
    print("\nAUC (style) by year:\n", res.pivot_table(index=["corpus", "year"], columns="pair", values="auc_style").round(3).to_string())
    print("\nAUC (full vocab) by year:\n", res.pivot_table(index=["corpus", "year"], columns="pair", values="auc_fullvocab").round(3).to_string())
    print("\nConvergence tests:\n", conv.round(4).to_string())
    print("\nCosine diagnostic:\n", cd.round(4).to_string())
    print("\nPCA explained variance:", evr.round(3))
    print("Top PC1 loadings:", load_.PC1.abs().sort_values(ascending=False).head(12).index.tolist())


if __name__ == "__main__":
    main()
