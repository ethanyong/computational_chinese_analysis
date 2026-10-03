# Tracing China in Lianhe Zaobao: A Computational Study of the China Desk, 2006–2025

Did the China desk of Singapore's *Lianhe Zaobao* change how it writes after the December 2021
restructuring of SPH Media? This project builds an archive of 618,626 Zaobao articles from the
Internet Archive, draws a stratified sample of 28,152 articles from four desks, and compares the
China desk with the World, Singapore and Southeast Asia desks using Chinese NLP, count models and
text classifiers.

**Paper:** [Tracing_China_in_Lianhe_Zaobao.pdf](Tracing_China_in_Lianhe_Zaobao.pdf)
(source: `paper/paper.tex`). **Results summary:** [REPORT.md](REPORT.md).

## Main findings

* **No change in tone at the restructuring.** Difference-in-differences tests against the World
  desk find no significant change in hedging, boosting, obligation, permission or attribution
  after December 2021.
* **A long-standing difference in voice.** The China desk has used roughly a third fewer hedges
  than the World desk for twenty years. Much of the gap remains when articles on the same topic
  are compared (NMF topic fixed effects), and 84% of it comes from one word, 可能 (*keneng*,
  "possibly").
* **Gradual, not sudden, style change.** On a classifier-based style axis fixed before 2021, the
  China desk's style has become slowly less distinctive since about 2019, with no break in 2021.
* **A change in sourcing (exploratory).** References to official bulletins and official websites
  on the China desk rose about threefold relative to the World desk after 2021. The increase
  remains when COVID coverage and the pandemic years are excluded, while the share of citations
  going to Chinese state media fell.

## Methods at a glance

Web archive harvesting (Wayback CDX API, 43,000 pages downloaded); HTML extraction and
boilerplate removal; Chinese word segmentation (jieba, OpenCC); a lexicon of epistemic and deontic
markers; Poisson quasi-likelihood rate ratios with robust standard errors; difference-in-differences
with placebo breaks and year-level inference; cross-validated logistic-regression classifiers
(AUC) on function-word and TF-IDF features; interrupted time series with clustered standard
errors; NMF topic models; weighted log-odds with an informative Dirichlet prior; Benjamini–Hochberg
false discovery rate control.

## Repository layout

```
Tracing_China_in_Lianhe_Zaobao.pdf   the paper
REPORT.md                            results summary with the key tables
run_all.py                           runs the whole pipeline
src/                                 pipeline scripts (steps 01-13) and shared modules
paper/paper.tex                      LaTeX source of the paper
figures/                             figures with titles; figures/paper/ holds the copies used in the paper
results/                             every statistical result as CSV, plus audit and context samples (.md)
data/                                archive index, sample, raw HTML, cleaned text and features
logs/                                output of each pipeline step
```

## Run it

```
py run_all.py        # steps 1-13 (steps 1-3 download from the Wayback Machine, ~2 h, resumable)
py run_all.py 4      # re-run only the offline analysis (steps 4-13, ~1 h)
```

Requires Python 3.11+ with pandas, numpy, scipy, scikit-learn, statsmodels, jieba, opencc,
beautifulsoup4, lxml, pyarrow, aiohttp, requests and matplotlib. All randomness uses a fixed seed
(`SEED` in `src/config.py`).

| Step | Script | Output |
|---|---|---|
| 1 | `src/01_harvest_index.py` | `data/raw/cdx_index.csv`: every archived Zaobao article URL (618,626), one row per article |
| 2 | `src/02_build_sample.py` | `data/interim/sample_frame.csv`: stratified random sample, seeded |
| 3 | `src/03_download.py` | `data/raw/html/shard_*.parquet`: raw page bytes |
| 4 | `src/04_extract_clean.py` | `data/interim/articles.parquet`: clean text and exclusion log |
| 5 | `src/05_features.py` | `data/features/*`: tokens, marker counts, function-word frequencies |
| 6 | `src/06_markers.py` | marker rate ratios, restructuring DiD, placebo breaks, per-word results (Figures 5–8) |
| 7 | `src/07_style_divergence.py` | content similarity and desk distinguishability (Figures 2–3; PCA in Figure S1) |
| 8 | `src/08_distinctive_words.py` | open-vocabulary distinctive words (Figure 9) |
| 9 | `src/09_data_quality.py` | archive coverage (Figure 1) and human extraction audit |
| 10 | `src/10_who_moved.py` | which desk moved, on a fixed style axis (Figure 4) |
| 11 | `src/11_sources.py` | named-source citations, exploratory (Figure 10) |
| 12 | `src/12_topic_control.py` | hedging gap with topic fixed effects (NMF topics) |
| 13 | `src/13_summary.py` | summary tables with false discovery rate control |

Shared modules: `config.py` (paths, URL parsing, seed), `lexicon.py` (marker words),
`textutils.py` (HTML decoding and extraction), `glosses.py` (English glosses and pinyin used in
figure labels), `plotstyle.py` (figure style). Figure numbers match the paper.

The paper needs XeLaTeX (for `xeCJK`) and the Microsoft JhengHei font; any XeLaTeX distribution or
Tectonic works: `tectonic -X compile paper/paper.tex`.

## Data decisions

1. **Article URLs only.** Only article URL formats are harvested (legacy `/{zg,gj,sp,yx}/…shtml`,
   modern `/news/{desk}/story…`, realtime `/realtime/{desk}/story…`). Daily section-index pages,
   which make up almost all captures of the desks in 2013–2015, are not articles and are excluded;
   the realtime section, whose articles are archived for 2013–2025, covers those years.
2. **Publication year from the URL**, never from the capture date (19% of modern desk articles
   were first captured in a later year).
3. **Boilerplate removal.** Layout-specific body extraction, corpus-level boilerplate removal
   (any paragraph in five or more articles), and removal of paywall and promotional lines.
4. **Paywall.** Most 2016–2023 desk pages in the archive are subscriber teasers (Figure 1). The
   realtime section (free, complete texts) is the main modern corpus; desk articles are analysed
   through their *leads* (first two paragraphs), which are visible for every article. Full texts
   of free desk articles are reported only as a secondary, self-selected subset.
5. **Token-level matching.** Markers are matched on whole jieba tokens, so 據 (*ju*, "according to")
   does not match 數據 (*shuju*, "data"), 說 (*shuo*, "say") does not match 說明 (*shuoming*,
   "explain"), and 不得 (*bude*, "must not") does not match 不得不 (*budebu*, "have to").
6. **Encoding.** Raw bytes are stored and decoded once from the declared character set (GB2312 is
   read as GB18030).
7. **Print-edition teasers** on legacy pages ("全文詳見今日出版的…", "full story in today's print
   edition") are detected and excluded.
8. **Duplicates** (protocol, domain and query-string variants) collapse to one canonical article ID.

## Design decisions

* **Sampling.** Strata are corpus × desk × publication year. Within a stratum, articles are put in
  a seeded random order, and the first 300 (desk) or 200 (realtime) usable ones form the sample.
  Filtering a random order by a fixed rule leaves a simple random sample of usable articles.
* **Count models.** Poisson quasi-likelihood with a log(tokens) offset and HC1 robust standard
  errors, which is consistent for the rate ratio under overdispersion; negative binomial models
  are reported as a check. A yearly rate ratio is only estimated when each desk has at least five
  markers that year.
* **Two levels of inference for the restructuring effect.** Article-level difference-in-differences
  and a year-level Welch test on yearly log rate ratios, which allows for news events that affect
  one desk in one year (such as the 2022 war coverage on the World desk). Both are corrected for
  multiple tests, and a result is called robust only if it holds at both levels.
* **Placebo breaks** on 1 December of every year show whether December 2021 stands out.
* **Multiple comparisons.** Benjamini–Hochberg false discovery rate control within families of
  tests defined in advance (see step 13).
* **Classifier comparability.** Regularization is chosen once and then held fixed; feature scaling
  is fitted inside training folds; every cell uses at most 150 articles per desk, and each AUC
  averages ten random draws.
* **Fixed-axis scoring (step 10).** Post-2021 articles are scored by the same five fold models that
  score the pre-2021 articles. A model refitted on all pre-2021 data would have larger
  coefficients (L2 penalty, more data) and would push post-2021 scores outward, creating a false
  divergence. Standard errors are clustered by publication year.
* **Style features** are closed-class words only (adverbs, conjunctions, prepositions, particles).
  Pronouns and words of place are excluded because they reveal the topic (我國 *woguo*, "our country";
  本地 *bendi*, "local").
* **Topic control (step 12).** NMF topics are fitted on all desks together without desk labels;
  20 and 40 topics are both reported.
