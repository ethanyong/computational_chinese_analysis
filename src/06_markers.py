"""Step 6 — Epistemic-marker analysis (research question Q3).

Quantity estimated
  For marker category c, the China-desk rate ratio relative to a comparison desk:
      IRR = (markers per token on the China desk) / (markers per token on the comparison desk)
  estimated by Poisson quasi-maximum-likelihood with log(token count) as offset and
  heteroskedasticity-robust (HC1) standard errors.  Poisson QMLE is consistent for the rate
  ratio whatever the overdispersion (Wooldridge 1999; Santos Silva & Tenreyro 2006); a
  negative-binomial fit is reported as a sensitivity check.  IRR = 1 means no difference.

Why rate ratios and not a standardized difference such as (mean_China - mean_World) / SD_World?
  That quantity is an effect size (Glass's delta) on article-level spread, not a test
  statistic: it ignores sample size, so it cannot be compared with +/-1.96.  Standardized
  effect sizes (Hedges' g with a CI) are reported in the CSV next to the rate ratios.

Analyses
  A. Per-year IRR, China vs World (also vs Singapore and SE Asia), with Benjamini-Hochberg
     FDR control across all years x categories within each corpus/comparison.
  B. Difference-in-differences: China vs World, before vs after the SPH Media restructuring
     (publication date >= 2021-12-01), desk and year fixed effects.  Two inference levels:
       B1 article level (robust SE)  — treats articles as independent;
       B2 desk-year level           — Welch test on yearly log-IRRs, which allows for
          desk-specific yearly news shocks (e.g. the World desk in 2022) and is the honest
          one when only ~10 years are observed.
     Placebo breakpoints at 1 December of every year (realtime 2014-2024, print leads 2017-2024)
     show whether 2021 is special.
  C. Per-term rate ratios (each lexicon word separately), so that no result depends on how
     words were grouped into categories; plus keyword-in-context samples for words that move.
  D. Lead-only robustness: first two paragraphs of *all* sampled articles, including
     paywall-truncated ones (whose visible text is exactly the lead).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from config import INTERIM, FEATURES, RESULTS, RESTRUCTURE_DATE
from lexicon import LEXICON, CATEGORIES
from plotstyle import DESK_COLOR, DESK_MARKER, DESK_LABEL, CAT_COLOR, INK2, events, save, zh
from glosses import gloss, pinyin_italic

warnings.filterwarnings("ignore")
POST = pd.Timestamp(RESTRUCTURE_DATE)
MIN_HITS = 5


def load():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    mk = pd.read_parquet(f"{FEATURES}/markers.parquet")
    df = art.merge(mk, on="article_id")
    df["era"] = np.where(df.corpus == "realtime", "realtime",
                         np.where(df.pub_year <= 2012, "legacy", "modern"))
    df["post"] = (df.pub_date >= POST).astype(int)
    return df


def irr_fit(d, y, focal="china", exposure="n_tok"):
    """Poisson QMLE rate ratio of `focal` desk vs the rest of `d` (one comparison desk)."""
    d = d[d[exposure] > 0]
    x = sm.add_constant((d.desk == focal).astype(float).rename("focal"))
    # a ratio of two rates is not estimable when either rate rests on a handful of events
    # (the MLE diverges as a count -> 0), so require >= MIN_HITS markers on each side
    hits = d.groupby(d.desk == focal)[y].sum()
    if x.focal.nunique() < 2 or hits.min() < MIN_HITS:
        return dict(irr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, log_irr=np.nan, se=np.nan)
    m = sm.GLM(d[y], x, family=sm.families.Poisson(), offset=np.log(d[exposure])).fit(cov_type="HC1")
    b, se = m.params["focal"], m.bse["focal"]
    return dict(irr=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues["focal"],
                log_irr=b, se=se)


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


# ------------------------------------------------------------------ A. per-year
def per_year(df, full_text=True):
    rows = []
    sub = df[df.fulltext_ok] if full_text else df
    pre = "" if full_text else "lead_"
    expo = "n_tok" if full_text else "n_tok_lead"
    for (corpus, year), g in sub.groupby(["corpus", "pub_year"]):
        for comp in ["world", "singapore", "sea"]:
            d = g[g.desk.isin(["china", comp])]
            if (d.desk == comp).sum() < 30 or (d.desk == "china").sum() < 30:
                continue
            for cat in CATEGORIES:
                r = irr_fit(d, pre + cat, exposure=expo)
                rate = d[pre + cat] / d[expo].clip(lower=1) * 1000
                gg = hedges_g(rate[d.desk == "china"], rate[d.desk == comp])
                rows.append(dict(corpus=corpus, year=year, comparison=comp, category=cat,
                                 n_china=(d.desk == "china").sum(), n_comp=(d.desk == comp).sum(),
                                 **r, g=gg[0], g_lo=gg[1], g_hi=gg[2]))
    res = pd.DataFrame(rows)
    res["q_bh"] = np.nan
    for _, idx in res.groupby(["corpus", "comparison"]).groups.items():
        ok = res.loc[idx, "p"].notna()
        res.loc[ok[ok].index, "q_bh"] = multipletests(res.loc[ok[ok].index, "p"], method="fdr_bh")[1]
    return res


# ------------------------------------------------------------------ B. DiD
def _did_data(df, corpus, years, controls, text):
    """full: archived full texts (fulltext_ok); lead: first two paragraphs of every article."""
    d = df[(df.corpus == corpus) & df.pub_year.isin(years) & df.desk.isin(("china",) + tuple(controls))].copy()
    if text == "full":
        d = d[d.fulltext_ok & (d.n_tok > 0)]
        d["expo"] = d.n_tok
    else:
        d = d[d.n_tok_lead > 0]
        d["expo"] = d.n_tok_lead
    return d


def did(df, corpus, years, break_date=POST, controls=("world",), y="hedge", text="full"):
    d = _did_data(df, corpus, years, controls, text)
    yv = y if text == "full" else f"lead_{y}"
    d["china_post"] = ((d.desk == "china") & (d.pub_date >= break_date)).astype(int)
    m = smf.glm(f"{yv} ~ C(desk) + C(pub_year) + china_post", d, family=sm.families.Poisson(),
                offset=np.log(d.expo)).fit(cov_type="HC1")
    b, se = m.params["china_post"], m.bse["china_post"]
    return dict(irr=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues["china_post"], n=len(d))


def did_nb(df, corpus, years, y):
    d = _did_data(df, corpus, years, ("world",), "full")
    d["china_post"] = ((d.desk == "china") & (d.pub_date >= POST)).astype(int)
    try:
        m = smf.negativebinomial(f"{y} ~ C(desk) + C(pub_year) + china_post", d,
                                 offset=np.log(d.expo)).fit(disp=0, maxiter=300, cov_type="HC1")
        b, se = m.params["china_post"], m.bse["china_post"]
        return dict(irr_nb=np.exp(b), lo_nb=np.exp(b - 1.96 * se), hi_nb=np.exp(b + 1.96 * se),
                    p_nb=m.pvalues["china_post"], alpha=m.params["alpha"])
    except Exception as e:  # report, never hide
        return dict(irr_nb=np.nan, lo_nb=np.nan, hi_nb=np.nan, p_nb=np.nan, alpha=np.nan, err=str(e)[:60])


def did_yearlevel(yearly, corpus, years, comp="world", cat="hedge"):
    """B2: Welch test of yearly China/World log-IRRs, post years (>=2022) vs pre years."""
    s = yearly[(yearly.corpus == corpus) & (yearly.comparison == comp) & (yearly.category == cat)
               & yearly.year.isin(years)].dropna(subset=["log_irr"])
    pre, post = s[s.year <= 2021].log_irr, s[s.year >= 2022].log_irr
    if len(pre) < 2 or len(post) < 2:
        return dict(ratio_of_irr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n_pre=len(pre), n_post=len(post))
    t = stats.ttest_ind(post, pre, equal_var=False)
    diff = post.mean() - pre.mean()
    se = np.sqrt(post.var(ddof=1) / len(post) + pre.var(ddof=1) / len(pre))
    return dict(ratio_of_irr=np.exp(diff), lo=np.exp(diff - stats.t.ppf(.975, t.df) * se),
                hi=np.exp(diff + stats.t.ppf(.975, t.df) * se), p=t.pvalue, n_pre=len(pre), n_post=len(post))


# ------------------------------------------------------------------ C. per term
def per_term(df):
    rows = []
    blocks = {"2006-12 (legacy print)": (df.corpus == "print") & (df.era == "legacy"),
              "2013-Nov 2021 (pre)": (df.corpus == "realtime") & (df.post == 0),
              "Dec 2021-2025 (post)": (df.corpus == "realtime") & (df.post == 1)}
    for cat, terms in LEXICON.items():
        for t in terms:
            for bname, mask in blocks.items():
                d = df[mask & df.fulltext_ok & df.desk.isin(["china", "world"])]
                r = irr_fit(d, f"t_{t}")
                n_hits = int(d[f"t_{t}"].sum())
                rate_c = d.loc[d.desk == "china", f"t_{t}"].sum() / d.loc[d.desk == "china", "n_tok"].sum() * 1e4
                rate_w = d.loc[d.desk == "world", f"t_{t}"].sum() / d.loc[d.desk == "world", "n_tok"].sum() * 1e4
                rows.append(dict(category=cat, term=t, block=bname, hits=n_hits,
                                 rate_china_per10k=rate_c, rate_world_per10k=rate_w, **r))
    res = pd.DataFrame(rows)
    ok = res.p.notna()
    res.loc[ok, "q_bh"] = multipletests(res.loc[ok, "p"], method="fdr_bh")[1]
    return res


def gap_decomposition(df):
    """How much of each category's China-World gap (per 10,000 tokens) each word contributes.

    gap_c = rate_World(c) - rate_China(c), pooled over the block; a word's contribution is its
    own World-minus-China rate difference, so contributions sum to the category gap."""
    rows = []
    blocks = {"2006-12 (legacy print)": (df.corpus == "print") & (df.era == "legacy"),
              "2013-2025 (realtime)": df.corpus == "realtime"}
    for bname, mask in blocks.items():
        d = df[mask & df.fulltext_ok & df.desk.isin(["china", "world"])]
        tc, tw = d.loc[d.desk == "china", "n_tok"].sum(), d.loc[d.desk == "world", "n_tok"].sum()
        for cat, terms in LEXICON.items():
            gap = (d.loc[d.desk == "world", cat].sum() / tw - d.loc[d.desk == "china", cat].sum() / tc) * 1e4
            for t in terms:
                c = (d.loc[d.desk == "world", f"t_{t}"].sum() / tw - d.loc[d.desk == "china", f"t_{t}"].sum() / tc) * 1e4
                rows.append(dict(block=bname, category=cat, term=t, category_gap_per10k=gap,
                                 term_gap_per10k=c, share_of_gap=c / gap if gap else np.nan))
    return pd.DataFrame(rows)


def kwic(df, terms=("不得", "据"), k=20):
    """Random keyword-in-context lines for marker words whose China/World ratio moves, so a
    reader can see what the counts are made of (realtime China desk, before and after)."""
    rng = np.random.default_rng(20260407)
    lines = ["# Marker words in context: realtime China desk (random occurrences)\n"]
    rt = df[(df.corpus == "realtime") & df.fulltext_ok & (df.desk == "china")]
    for t in terms:
        for era, m in [("2013-Nov 2021", rt.post == 0), ("Dec 2021-2025", rt.post == 1)]:
            sub = rt[m & (rt[f"t_{t}"] > 0)]
            ctx = []
            for txt in sub.text:
                i = txt.find(t)
                while i >= 0:
                    ctx.append(txt[max(0, i - 30):i + 20].replace("\n", " "))
                    i = txt.find(t, i + 1)
            pick = rng.choice(len(ctx), min(k, len(ctx)), replace=False) if ctx else []
            lines.append(f"\n## {t}, {era} ({len(sub)} articles with the token; substring contexts shown)\n")
            lines += [f"- ...{ctx[j]}..." for j in pick]
    open(f"{RESULTS}/markers_kwic.md", "w", encoding="utf-8").write("\n".join(lines))


# ------------------------------------------------------------------ figures
def _rate_ci(g, y, expo):
    r = sm.GLM(g[y], np.ones((len(g), 1)), family=sm.families.Poisson(), offset=np.log(g[expo])).fit(cov_type="HC1")
    b, se = r.params.iloc[0], r.bse.iloc[0]
    return np.exp(b) * 1000, np.exp(b - 1.96 * se) * 1000, np.exp(b + 1.96 * se) * 1000


def fig_levels(df):
    """Rates per 1,000 tokens by desk and year, both unselected text units, with 95% CIs."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True)
    rows = [("print", "lead", "Print desks: article leads\n(all sampled articles)"),
            ("realtime", "full", "Realtime desks: full text")]
    for r_i, (corpus, unit, rlab) in enumerate(rows):
        d = df[df.corpus == corpus]
        d = d[d.n_tok_lead > 0] if unit == "lead" else d[d.fulltext_ok & (d.n_tok > 0)]
        expo = "n_tok_lead" if unit == "lead" else "n_tok"
        for c_i, cat in enumerate(["hedge", "booster", "obligation"]):
            ax = axes[r_i, c_i]
            y = f"lead_{cat}" if unit == "lead" else cat
            for desk in ["china", "world", "singapore", "sea"]:
                pts = [(yr, *_rate_ci(g, y, expo)) for yr, g in d[d.desk == desk].groupby("pub_year") if g[y].sum() > 0]
                if not pts:
                    continue
                s = pd.DataFrame(pts, columns=["y", "m", "lo", "hi"])
                for part in (s[s.y <= 2012], s[s.y >= 2013]):
                    if part.empty:
                        continue
                    ax.plot(part.y, part.m, color=DESK_COLOR[desk], marker=DESK_MARKER[desk], lw=1.6,
                            label=DESK_LABEL[desk] if part.y.max() >= 2016 else None)
                    ax.fill_between(part.y, part.lo, part.hi, color=DESK_COLOR[desk], alpha=.12, lw=0)
            if corpus == "print":
                ax.axvspan(2012.5, 2015.5, color="#f0efec", zorder=0)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{cat.capitalize()} per 1,000 tokens", fontsize=10)
            events(ax, label=(r_i == 0 and c_i == 0))
        axes[r_i, 0].set_ylabel(rlab, fontsize=9)
        axes[r_i, -1].legend(loc="upper left", fontsize=8, bbox_to_anchor=(1, 1))
    fig.suptitle("Figure 5. Marker rates by desk (bands = 95% CI; grey band = no archived print desk articles 2013-15)",
                 x=.01, ha="left")
    save(fig, "fig05_marker_rates_by_desk")


def fig_yearly_irr(lead, full):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    series = [(lead, "print", "#2a78d6", "o", -0.12, "Print, leads"),
              (full, "realtime", "#eb6834", "s", 0.12, "Realtime, full text")]
    for ax, cat in zip(axes, CATEGORIES):
        for res, corpus, col, mk, off, lab in series:
            s = res[(res.corpus == corpus) & (res.comparison == "world") & (res.category == cat)].dropna(subset=["irr"])
            sig = s.q_bh < 0.05
            ax.errorbar(s.year + off, s.irr, yerr=[s.irr - s.lo, s.hi - s.irr], fmt="none", ecolor=col, elinewidth=1, alpha=.7)
            ax.scatter(s.year[sig] + off, s.irr[sig], color=col, marker=mk, s=26, zorder=3, label=lab)
            ax.scatter(s.year[~sig] + off, s.irr[~sig], facecolor="white", edgecolor=col, marker=mk, s=26, zorder=3)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_yscale("log")
        ax.set_ylim(0.25, 4)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.minorticks_off()
        ax.set_title(cat.capitalize())
        events(ax, label=False)
    axes[0].set_ylabel("China / World rate ratio (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1, 1.02))
    fig.suptitle("Figure 6. China-desk marker rates relative to the World desk, by year "
                 "(95% CI; filled = significant after FDR correction, hollow = not)", x=.01, ha="left")
    save(fig, "fig06_china_world_rate_ratios")


def fig_placebo(pl):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4), sharey="row")
    for r_i, (corpus, lab) in enumerate([("realtime", "Realtime, full text"), ("print", "Print, leads")]):
        for ax, cat in zip(axes[r_i], CATEGORIES):
            s = pl[(pl.category == cat) & (pl.corpus == corpus)]
            col = np.where(s.break_year == 2021, "#2a78d6", "#8a8984")
            ax.errorbar(s.break_year, s.irr, yerr=[s.irr - s.lo, s.hi - s.irr], fmt="none", ecolor="#8a8984", elinewidth=1)
            ax.scatter(s.break_year, s.irr, c=col, s=30, zorder=3)
            ax.axhline(1, color=INK2, lw=1)
            ax.set_yscale("log")
            ax.set_ylim(0.2, 5)
            ax.set_yticks([0.25, 0.5, 1, 2, 4])
            ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
            ax.minorticks_off()
            ax.set_title(f"{cat.capitalize()} ({lab})", fontsize=9.5)
            ax.set_xticks(range(2014, 2025, 2))
        axes[r_i, 0].set_ylabel("DiD rate ratio (China x after)")
    fig.suptitle("Figure 7. Difference-in-differences (China vs World) with the break placed on 1 Dec of each year: "
                 "blue = actual SPH restructuring (Dec 2021), grey = placebo breaks", x=.01, ha="left")
    save(fig, "fig07_placebo_breaks")


def fig_terms(terms):
    piv = terms.pivot_table(index=["category", "term"], columns="block", values="irr", sort=False)
    q = terms.pivot_table(index=["category", "term"], columns="block", values="q_bh", sort=False)
    hits = terms.groupby(["category", "term"], sort=False).hits.sum()
    keep = hits[hits >= 40].index
    piv, q = piv.loc[keep], q.loc[keep]
    cols = ["2006-12 (legacy print)", "2013-Nov 2021 (pre)", "Dec 2021-2025 (post)"]
    piv, q = piv[cols], q[cols]
    L = np.log2(piv.values.astype(float))
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(piv) + 1.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    im = ax.imshow(np.clip(L, -1.5, 1.5), cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if np.isfinite(L[i, j]):
                star = "*" if q.values[i, j] < 0.05 else ""
                ax.text(j, i, f"{piv.values[i, j]:.2f}{star}", ha="center", va="center", fontsize=7, color="#0b0b0b")
    ax.set_xticks(range(3), ["2006-12\nprint", "2013-Nov 21\nrealtime", "Dec 21-2025\nrealtime"], fontsize=8)
    ax.set_yticks(range(len(piv)), [f"{zh(t)} ({pinyin_italic(t)}, {gloss(t)}), {c}" for c, t in piv.index], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.5)
    cb.set_label("log2 China/World rate ratio\n(blue = more on China desk)", fontsize=8)
    ax.set_title("Figure 8. Each marker word separately: China/World rate ratio\n(* = significant after FDR; words with >= 40 hits)", loc="left", fontsize=10)
    save(fig, "fig08_per_word_rate_ratios")


def main():
    df = load()
    print(f"{len(df):,} articles; {(df.fulltext_ok).sum():,} full-text")

    yearly = per_year(df, full_text=True)
    yearly.to_csv(f"{RESULTS}/markers_yearly_rate_ratios_FULLTEXT.csv", index=False)
    lead = per_year(df, full_text=False)
    lead.to_csv(f"{RESULTS}/markers_yearly_rate_ratios_LEADS.csv", index=False)

    modern, rt = list(range(2016, 2026)), list(range(2013, 2026))
    specs = [  # (corpus, years, text unit, controls, yearly table used for year-level inference)
        ("realtime", rt, "full", [("world", ("world",)), ("singapore", ("singapore",)),
                                  ("all other desks", ("world", "singapore"))], yearly),
        ("print", modern, "lead", [("world", ("world",)), ("sea", ("sea",)), ("singapore", ("singapore",)),
                                   ("all other desks", ("world", "singapore", "sea"))], lead),
        ("print", modern, "full", [("world", ("world",))], yearly),   # free-article subset only
    ]
    did_rows, plac = [], []
    for cat in CATEGORIES:
        for corpus, yrs, unit, ctrls, ytab in specs:
            for ctrl_name, ctrl in ctrls:
                r = did(df, corpus, yrs, controls=ctrl, y=cat, text=unit)
                r2 = did_yearlevel(ytab, corpus, yrs, comp=ctrl[0], cat=cat) if len(ctrl) == 1 else {}
                row = dict(category=cat, corpus=corpus, text=unit, control=ctrl_name, **r,
                           **{f"yearlevel_{k}": v for k, v in r2.items()})
                if ctrl_name == "world" and corpus == "realtime":
                    row.update(did_nb(df, corpus, yrs, cat))
                did_rows.append(row)
        for corpus, yrs, unit, first in [("realtime", rt, "full", 2014), ("print", modern, "lead", 2017)]:
            for by in range(first, 2025):
                r = did(df, corpus, yrs, break_date=pd.Timestamp(f"{by}-12-01"), y=cat, text=unit)
                plac.append(dict(category=cat, corpus=corpus, text=unit, break_year=by, **r))
    did_df = pd.DataFrame(did_rows)
    did_df.to_csv(f"{RESULTS}/markers_did.csv", index=False)
    pl = pd.DataFrame(plac)
    pl.to_csv(f"{RESULTS}/markers_did_placebo.csv", index=False)

    terms = per_term(df)
    terms.to_csv(f"{RESULTS}/markers_per_term.csv", index=False)
    dec = gap_decomposition(df)
    dec.to_csv(f"{RESULTS}/markers_gap_decomposition.csv", index=False)
    kwic(df)

    fig_levels(df)
    fig_yearly_irr(lead, yearly)
    fig_placebo(pl)
    fig_terms(terms)

    pd.set_option("display.width", 250)
    show = ["category", "corpus", "text", "control", "n", "irr", "lo", "hi", "p", "yearlevel_ratio_of_irr",
            "yearlevel_lo", "yearlevel_hi", "yearlevel_p"]
    print("\nDiD (China x after Dec 2021):\n", did_df[show].round(3).to_string())
    print("\nNB sensitivity (realtime, vs World):\n",
          did_df[(did_df.corpus == "realtime") & (did_df.control == "world")][["category", "irr", "irr_nb", "lo_nb", "hi_nb", "p_nb"]].round(3).to_string())
    print("\nPlacebo breaks:\n", pl.pivot_table(index=["corpus", "break_year"], columns="category", values="irr").round(3).to_string())
    for name, tab in [("FULL TEXT", yearly), ("LEADS", lead)]:
        w = tab[(tab.comparison == "world")]
        w = w.assign(s=w.irr.round(2).astype(str) + np.where(w.q_bh < .05, "*", ""))
        print(f"\nPer-year China/World IRR, {name} (q<.05 marked *):")
        print(w.pivot_table(index=["corpus", "year"], columns="category", values="s", aggfunc="first").to_string())
    t = terms.pivot_table(index=["category", "term"], columns="block", values="irr", sort=False).round(2)
    print("\nPer-term China/World IRR:\n", t.to_string())
    print("\nHedge-gap decomposition (World minus China, per 10k tokens):\n",
          dec[dec.category == "hedge"].round(3).to_string())


if __name__ == "__main__":
    main()
