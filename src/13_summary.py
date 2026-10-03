"""Step 13 — Summary tables with multiple-comparison control, used by the paper and REPORT.md.

Families of tests (Benjamini-Hochberg FDR applied within each family; families fixed in advance):
  F1 confirmatory  restructuring DiD, realtime full text, China vs World, 5 marker categories
                   (article-level and year-level inference, each corrected separately)
  F2 confirmatory  restructuring DiD, print leads, China vs World, 5 marker categories (same)
  F3 confirmatory  convergence/divergence tests on style AUC (year-level Welch), all desk pairs
  F4 confirmatory  directional "who moved" tests: 4 desk-axis level shifts + 2 desk-gap shifts
  F5 exploratory   source-citation ITS (5 groups x 2 corpora) + official-channel DiD at Dec 2021

Also computes the headline long-run table: pooled China/other-desk rate ratios per era with
year fixed effects (Poisson QMLE, HC1), i.e. the average gap within the same years.

Outputs: results/summary_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from config import INTERIM, FEATURES, RESULTS
from lexicon import CATEGORIES


def bh(df, p="p", q="q_bh"):
    df = df.copy()
    ok = df[p].notna()
    df[q] = np.nan
    df.loc[ok, q] = multipletests(df.loc[ok, p], method="fdr_bh")[1]
    return df


def pooled_gaps():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    mk = pd.read_parquet(f"{FEATURES}/markers.parquet")
    df = art.merge(mk, on="article_id")
    eras = [("print 2006-12, full text", (df.corpus == "print") & (df.pub_year <= 2012) & df.fulltext_ok, ""),
            ("print 2016-25, leads", (df.corpus == "print") & (df.pub_year >= 2016), "lead_"),
            ("realtime 2013-25, full text", (df.corpus == "realtime") & df.fulltext_ok, "")]
    rows = []
    for name, mask, pre in eras:
        expo = "n_tok_lead" if pre else "n_tok"
        for comp in ["world", "singapore", "sea"]:
            d = df[mask & df.desk.isin(["china", comp]) & (df[expo] > 0)].copy()
            if (d.desk == comp).sum() == 0:
                continue
            d["china"] = (d.desk == "china").astype(int)
            for cat in CATEGORIES:
                m = smf.glm(f"{pre}{cat} ~ china + C(pub_year)", d, family=sm.families.Poisson(),
                            offset=np.log(d[expo])).fit(cov_type="HC1")
                b, se = m.params["china"], m.bse["china"]
                rows.append(dict(era=name, comparison=comp, category=cat, irr=np.exp(b), lo=np.exp(b - 1.96 * se),
                                 hi=np.exp(b + 1.96 * se), p=m.pvalues["china"], n=len(d)))
    return bh(pd.DataFrame(rows))


def main():
    did = pd.read_csv(f"{RESULTS}/markers_did.csv")
    f1 = bh(did[(did.corpus == "realtime") & (did.text == "full") & (did.control == "world")])
    f2 = bh(did[(did.corpus == "print") & (did.text == "lead") & (did.control == "world")])
    f1, f2 = bh(f1, "yearlevel_p", "yearlevel_q_bh"), bh(f2, "yearlevel_p", "yearlevel_q_bh")
    pd.concat([f1.assign(family="F1"), f2.assign(family="F2")]).to_csv(f"{RESULTS}/summary_did_fdr.csv", index=False)

    conv = pd.read_csv(f"{RESULTS}/convergence_tests.csv")
    f3 = bh(conv[conv.metric == "auc_style"], p="p_welch")
    f3.to_csv(f"{RESULTS}/summary_convergence_fdr.csv", index=False)

    wm = pd.read_csv(f"{RESULTS}/who_moved.csv").rename(columns={"its_p": "p"})
    wg = pd.read_csv(f"{RESULTS}/who_moved_gap.csv").assign(desk="gap (China minus other)")
    wg = wg.rename(columns={"level_shift": "its_level_shift", "lo": "its_lo", "hi": "its_hi"})
    f4 = bh(pd.concat([wm, wg], ignore_index=True))
    f4.to_csv(f"{RESULTS}/summary_who_moved_fdr.csv", index=False)

    its = pd.read_csv(f"{RESULTS}/sources_its.csv")
    od = pd.read_csv(f"{RESULTS}/sources_official_did_placebo.csv")
    od21 = od[od.break_year == 2021].assign(corpus="realtime", group="official channels (DiD vs World)")
    f5 = bh(pd.concat([its, od21], ignore_index=True))
    f5.to_csv(f"{RESULTS}/summary_sources_fdr.csv", index=False)

    gaps = pooled_gaps()
    gaps.to_csv(f"{RESULTS}/summary_longrun_gaps.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["category", "irr", "lo", "hi", "p", "q_bh", "yearlevel_ratio_of_irr", "yearlevel_lo", "yearlevel_hi",
            "yearlevel_p", "yearlevel_q_bh"]
    print("F1 realtime DiD vs World:\n", f1[cols].round(3).to_string())
    print("\nF2 print-lead DiD vs World:\n", f2[cols].round(3).to_string())
    print("\nF3 style AUC change tests:\n", f3[["corpus", "pair", "pre_mean", "post_mean", "change", "p_welch", "q_bh"]].round(3).to_string())
    print("\nF4 who moved (ITS level shift):\n", f4[["axis", "desk", "trend_per_year", "its_level_shift", "its_lo", "its_hi", "p", "q_bh"]].round(3).to_string())
    print("\nF5 sources (exploratory):\n", f5[["corpus", "group", "pre_mean", "post_mean", "level_shift", "irr", "lo", "hi", "p", "q_bh"]].round(3).to_string())
    g = gaps.assign(s=gaps.irr.round(2).astype(str) + " [" + gaps.lo.round(2).astype(str) + "-" + gaps.hi.round(2).astype(str) + "]"
                    + np.where(gaps.q_bh < .05, "*", ""))
    print("\nLong-run China/comparison rate ratios (year FE):\n",
          g.pivot_table(index=["era", "comparison"], columns="category", values="s", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
