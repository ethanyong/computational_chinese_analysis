"""Step 10 — Which desk moved?  A directional test of divergence (Q1) and convergence (Q2).

AUC is symmetric: a rise in AUC(China vs World) after 2021 cannot say whether the China desk or
the World desk changed, and it cannot test the convergence hypothesis Q2 ("the China desk
moved toward the Singapore desk").  This step fixes the axis in advance:

  1. Fit the China-vs-X classifier (same function-word features and the same fixed
     regularization as step 7) on PRE-restructuring articles only (published before 1 Dec 2021).
  2. Score every article on that fixed axis with the SAME models.  Pre-period articles are split
     into 5 folds; each is scored by the model fitted on the other 4 (out-of-fold), and every
     post-period article is scored by the average of those 5 fold models.
     (Scoring post-period articles with one model refitted on all pre-period articles would be
     wrong here: with L2 regularization the coefficients grow with the training size, so that
     model's scores are ~12% larger in magnitude, which mechanically pushes post-period China
     articles toward 1 and the other desk's toward 0 — a spurious "divergence".)
  3. Scores are rescaled so that the pre-period mean of the China desk = 1 and of desk X = 0.
  4. Per desk, an interrupted time series separates a break from a pre-existing drift:
         score_i = b0 + b1 * (t_i - 2021.92) + b2 * post_i + e_i
     b1 = drift per year before the break, b2 = level shift at December 2021 net of the drift.
     Standard errors are clustered by publication year (13 clusters, t with 12 df), because
     news shocks are shared by articles from the same year.
  5. Desk gap.  House-style changes move both desks the same way along the axis, so the
     quantity Q1/Q2 are actually about is the DISTANCE between the desks on the axis:
         D_y = mean score(China, year y) - mean score(X, year y),   D = 1 before 2021 on average.
     Year-level OLS  D_y = c0 + c1 * (y - 2021.92) + c2 * post_y  (13 years, classical SEs).
     Q1 (divergence from World) predicts c2 > 0 on the China-World axis; Q2 (convergence on
     Singapore) predicts c2 < 0 on the China-Singapore axis.

Corpus: realtime, full text (the corpus with complete, unselected texts across the break).
Outputs: results/who_moved.csv, results/who_moved_gap.csv, results/who_moved_by_year.csv,
         figures/fig04_which_desk_moved.png
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import INTERIM, FEATURES, RESULTS, SEED, RESTRUCTURE_DATE
from plotstyle import DESK_COLOR, DESK_MARKER, DESK_LABEL, INK2, events, save

C_STYLE = float(pd.read_csv(f"{RESULTS}/classifier_C.csv").set_index("space").loc["style", "C"])
T_BREAK = 2021 + (pd.Timestamp(RESTRUCTURE_DATE).dayofyear - 1) / 365   # 1 Dec 2021 = 2021.915


def load():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    tok = pd.read_parquet(f"{FEATURES}/tokens.parquet", columns=["article_id", "n_tok"])
    art = art[(art.corpus == "realtime") & art.fulltext_ok].merge(tok, on="article_id")
    art = art[art.n_tok >= 100].reset_index(drop=True)
    fw = pd.read_parquet(f"{FEATURES}/function_words.parquet").set_index("article_id").loc[art.article_id]
    art["post"] = art.pub_date >= pd.Timestamp(RESTRUCTURE_DATE)
    art["t"] = art.pub_date.dt.year + (art.pub_date.dt.dayofyear - 1) / 365 - T_BREAK
    return art, fw.values.astype(float)


def axis_scores(art, X, a, b):
    d = art[art.desk.isin([a, b])]
    pre, post = d[~d.post].index.values, d[d.post].index.values
    y_pre = (art.desk.values[pre] == a).astype(int)
    score = pd.Series(np.nan, index=d.index)
    post_scores = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[pre], y_pre):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C_STYLE, max_iter=5000)).fit(X[pre][tr], y_pre[tr])
        score.loc[pre[te]] = m.decision_function(X[pre][te])
        post_scores.append(m.decision_function(X[post]))
    score.loc[post] = np.mean(post_scores, axis=0)
    # rescale so that the pre-period mean of desk b = 0 and of desk a = 1
    lo_mean = score[pre][y_pre == 0].mean()
    gap = score[pre][y_pre == 1].mean() - lo_mean
    return ((score - lo_mean) / gap).rename("score"), gap


def clustered(formula, data):
    return smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data.pub_year.values}, use_t=True)


def main():
    art, X = load()
    print(f"C = {C_STYLE}; break at t = {T_BREAK:.3f}")
    rows, gaps, yearly = [], [], []
    for a, b in [("china", "world"), ("china", "singapore")]:
        s, raw_gap = axis_scores(art, X, a, b)
        d = art.loc[s.index].assign(score=s, post_i=lambda x: x.post.astype(int))
        for desk in [a, b]:
            g = d[d.desk == desk]
            shift = clustered("score ~ post_i", g)
            its = clustered("score ~ t + post_i", g)
            ci_s, ci_i = shift.conf_int(), its.conf_int()
            rows.append(dict(axis=f"{a}-vs-{b}", desk=desk, n_pre=int((~g.post).sum()), n_post=int(g.post.sum()),
                             raw_gap=raw_gap, pre_mean=g[~g.post].score.mean(), post_mean=g[g.post].score.mean(),
                             shift=shift.params["post_i"], lo=ci_s.loc["post_i", 0], hi=ci_s.loc["post_i", 1],
                             trend_per_year=its.params["t"], trend_lo=ci_i.loc["t", 0], trend_hi=ci_i.loc["t", 1],
                             its_level_shift=its.params["post_i"], its_lo=ci_i.loc["post_i", 0],
                             its_hi=ci_i.loc["post_i", 1], its_p=its.pvalues["post_i"]))
            for yr, gy in g.groupby("pub_year"):
                m = gy.score.mean()
                se = gy.score.std(ddof=1) / np.sqrt(len(gy))
                yearly.append(dict(axis=f"{a}-vs-{b}", desk=desk, year=yr, mean=m, lo=m - 1.96 * se, hi=m + 1.96 * se))
        # distance between the desks on the fixed axis, by year
        yr = pd.DataFrame([r for r in yearly if r["axis"] == f"{a}-vs-{b}"]).pivot(index="year", columns="desk", values="mean")
        D = pd.DataFrame({"year": yr.index, "D": (yr[a] - yr[b]).values})
        D["t"], D["post"] = D.year + 0.5 - T_BREAK, (D.year >= 2022).astype(int)
        m = smf.ols("D ~ t + post", D).fit()
        ci = m.conf_int()
        gaps.append(dict(axis=f"{a}-vs-{b}", pre_mean=D[D.post == 0].D.mean(), post_mean=D[D.post == 1].D.mean(),
                         trend_per_year=m.params["t"], trend_lo=ci.loc["t", 0], trend_hi=ci.loc["t", 1],
                         level_shift=m.params["post"], lo=ci.loc["post", 0], hi=ci.loc["post", 1],
                         p=m.pvalues["post"], n_years=len(D)))
        for _, r in D.iterrows():
            yearly.append(dict(axis=f"{a}-vs-{b}", desk="gap", year=int(r.year), mean=r.D, lo=np.nan, hi=np.nan))
    res, gp, yr = pd.DataFrame(rows), pd.DataFrame(gaps), pd.DataFrame(yearly)
    res.to_csv(f"{RESULTS}/who_moved.csv", index=False)
    gp.to_csv(f"{RESULTS}/who_moved_gap.csv", index=False)
    yr.to_csv(f"{RESULTS}/who_moved_by_year.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    print("\nDesk gap on the fixed axis (year level):\n", gp.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, axis_name, title in [(axes[0], "china-vs-world", "Axis: China (1) vs World (0)"),
                                 (axes[1], "china-vs-singapore", "Axis: China (1) vs Singapore (0)")]:
        for desk, g in yr[(yr.axis == axis_name) & (yr.desk != "gap")].groupby("desk"):
            ax.plot(g.year, g["mean"], color=DESK_COLOR[desk], marker=DESK_MARKER[desk], lw=1.6, label=DESK_LABEL[desk])
            ax.fill_between(g.year, g.lo, g.hi, color=DESK_COLOR[desk], alpha=.15, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mean position\n(2013-Nov 2021 desk means = 1 and 0)")
        ax.axhline(1, color=INK2, lw=.8)
        ax.axhline(0, color=INK2, lw=.8)
        ax.legend(fontsize=8, loc="center left")
        events(ax, label=False)
    ax = axes[2]
    for axis_name, desk, lab in [("china-vs-world", "world", "China minus World"),
                                 ("china-vs-singapore", "singapore", "China minus Singapore")]:
        g = yr[(yr.axis == axis_name) & (yr.desk == "gap")]
        ax.plot(g.year, g["mean"], color=DESK_COLOR[desk], marker=DESK_MARKER[desk], lw=1.6, label=lab)
    ax.axhline(1, color=INK2, lw=.8)
    ax.set_ylim(0, None)
    ax.set_title("Distance between the desks on each axis", fontsize=10)
    ax.set_ylabel("Distance (2013-Nov 2021 average = 1)")
    ax.legend(fontsize=8, loc="lower left")
    events(ax, label=False)
    fig.suptitle("Figure 4. Which desk moved? Yearly mean position of each desk's articles on a style axis "
                 "fixed before the restructuring (realtime corpus; 95% CI)", x=.01, ha="left")
    save(fig, "fig04_which_desk_moved")


if __name__ == "__main__":
    main()
