"""Step 9 — Data-quality report: what the archive actually contains, and an extraction audit.

Outputs
  figures/fig01_archive_coverage.png    sample sizes, full-text availability, paywall share
  results/extraction_audit.md           40 randomly chosen articles (stratified by corpus and
                                        era) with metadata and the first 300 characters of the
                                        extracted text, for a human to check extraction quality
  results/data_quality_summary.csv      per corpus-era counts and text-length quantiles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RAW, INTERIM, RESULTS, SEED
from plotstyle import DESK_COLOR, DESK_MARKER, DESK_LABEL, INK2, save


def main():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    idx = pd.read_csv(f"{RAW}/cdx_index.csv", usecols=["corpus", "desk", "pub_year"])
    art["era"] = np.where(art.corpus == "realtime", "realtime 2013-25",
                          np.where(art.pub_year <= 2012, "print 2006-12", "print 2016-25"))

    q = art.groupby("era").agg(articles=("article_id", "size"), fulltext=("fulltext_ok", "sum"),
                               share_truncated=("truncated", "mean"),
                               cjk_p25=("n_cjk", lambda s: s.quantile(.25)), cjk_median=("n_cjk", "median"),
                               cjk_p75=("n_cjk", lambda s: s.quantile(.75)))
    pop = idx[idx.pub_year.between(2006, 2025)].groupby("corpus").size()
    q.to_csv(f"{RESULTS}/data_quality_summary.csv")
    print(q.round(3).to_string(), "\nArchived article population (2006-2025):", pop.to_dict())

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    for desk in ["china", "world", "singapore", "sea"]:
        s = idx[(idx.corpus == "print") & (idx.desk == desk) & idx.pub_year.between(2006, 2025)].groupby("pub_year").size()
        s = s.reindex(range(2006, 2026), fill_value=0)
        ax.plot(s.index, s.values, color=DESK_COLOR[desk], marker=DESK_MARKER[desk], lw=1.4, label=DESK_LABEL[desk])
    ax.text(2014, 30, "no article\nURLs", ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Archived print-desk articles per year (population)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    for corpus, col, mk in [("print", "#2a78d6", "o"), ("realtime", "#eb6834", "s")]:
        s = art[art.corpus == corpus].groupby("pub_year").truncated.mean()
        for k, part in enumerate((s[s.index <= 2012], s[s.index >= 2013])):   # never draw across the 2013-15 gap
            ax.plot(part.index, part.values * 100, color=col, marker=mk, lw=1.6, label=corpus if part.index.min() == s.index.min() else None)
    ax.set_ylim(0, 100)
    ax.set_title("Sampled pages that are paywall teasers (%)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[2]
    for corpus, col, mk in [("print", "#2a78d6", "o"), ("realtime", "#eb6834", "s")]:
        s = art[art.corpus == corpus].groupby("pub_year").fulltext_ok.sum()
        for k, part in enumerate((s[s.index <= 2012], s[s.index >= 2013])):
            ax.plot(part.index, part.values, color=col, marker=mk, lw=1.6, label=corpus if part.index.min() == s.index.min() else None)
    ax.set_ylim(bottom=0)
    ax.set_title("Full-text articles in the analysis sample (all desks)", fontsize=10)
    ax.legend(fontsize=8)
    for a in axes:
        a.axvspan(2012.5, 2015.5, color="#f0efec", zorder=0)
    fig.suptitle("Figure 1. What the Wayback archive holds: print-desk full texts are scarce 2013-2023 "
                 "(no desk archive 2013-15; paywall teasers 2016-23), realtime pages are complete", x=.01, ha="left")
    save(fig, "fig01_archive_coverage")

    rng = np.random.default_rng(SEED)
    lines = ["# Extraction audit\n", "Forty articles drawn at random (10 per corpus-era stratum, 10 truncated pages) "
             "to let a reader verify that the extracted text is article prose, not navigation or boilerplate.\n"]
    picks = [art[art.era == e].sample(10, random_state=SEED) for e in sorted(art.era.unique())]
    picks.append(art[art.truncated].sample(min(10, art.truncated.sum()), random_state=SEED))
    for i, r in pd.concat(picks).iterrows():
        lines.append(f"## {r.article_id} | {r.desk} | {r.pub_date.date()} | truncated={r.truncated} | {r.n_cjk} CJK chars\n")
        lines.append(f"Source: {r.original}  \nTitle: {r.title}\n")
        lines.append("> " + r.text[:300].replace("\n", " / ") + " ...\n")
    open(f"{RESULTS}/extraction_audit.md", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
