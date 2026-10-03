"""Step 8 — Open-vocabulary comparison: which words distinguish the desks, and what changed?

Method: weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn
2008, "Fightin' Words", Political Analysis 16(4)).  For word w and groups i, j:

    delta_w = log[(y_wi + a_w) / (n_i + a0 - y_wi - a_w)] - log[(y_wj + a_w) / (n_j + a0 - y_wj - a_w)]
    var(delta_w) ~= 1/(y_wi + a_w) + 1/(y_wj + a_w),      z_w = delta_w / sqrt(var)

where a_w = a0 * (background frequency of w) and a0 = 10,000 pseudo-tokens of prior drawn from the
whole corpus.  The prior shrinks rare words toward zero, so the list is not dominated by
one-off names (the failure mode of raw frequency ratios), and z is comparable across words.
|z| > 3.29 corresponds to p < .001 two-sided for a single word; with thousands of words we read
the ranking descriptively, not as individual tests.

Comparisons are run on two unselected text sets: the realtime corpus (full text, 2013-2025;
primary, shown in Figure 8) and print-desk article leads (2016-2025; secondary, CSV only).
  1. China desk vs World desk, before the restructuring (2013 or 2016 - Nov 2021)
  2. China desk vs World desk, after (Dec 2021 - 2025)
  3. China desk after vs China desk before — and World desk after vs before, so that changes
     common to the whole paper (e.g. COVID vocabulary) can be told apart from China-desk change.
Tokens: jieba words of >= 2 characters (single characters are mostly particles), count >= 30.
"""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import INTERIM, FEATURES, RESULTS, RESTRUCTURE_DATE
from plotstyle import INK2, save, zh
from glosses import gloss

A0 = 10_000
MIN_COUNT = 30


def counts(token_series):
    c = Counter()
    for s in token_series:
        c.update(w for w in s.split() if len(w) >= 2)
    return c


def fightin_words(ci, cj, background):
    vocab = [w for w, n in background.items() if n >= MIN_COUNT]
    tot_bg = sum(background[w] for w in vocab)
    ni, nj = sum(ci[w] for w in vocab), sum(cj[w] for w in vocab)
    rows = []
    for w in vocab:
        a = A0 * background[w] / tot_bg
        yi, yj = ci[w], cj[w]
        d = np.log((yi + a) / (ni + A0 - yi - a)) - np.log((yj + a) / (nj + A0 - yj - a))
        v = 1 / (yi + a) + 1 / (yj + a)
        rows.append((w, yi, yj, yi / ni * 1e4, yj / nj * 1e4, d, d / np.sqrt(v)))
    return pd.DataFrame(rows, columns=["word", "count_i", "count_j", "per10k_i", "per10k_j", "log_odds", "z"]) \
        .sort_values("z", ascending=False)


def plot_pair(ax, fw, title, left, right, k=15):
    top = fw.head(k)
    bot = fw.tail(k).iloc[::-1]
    ys = np.arange(k)
    ax.barh(ys, top.z.values, color="#2a78d6", height=.7)
    ax.barh(-ys - 1.5, bot.z.values, color="#eb6834", height=.7)
    for y, (_, r) in zip(ys, top.iterrows()):
        ax.text(r.z + .3, y, f"{zh(r.word)} ({gloss(r.word)})", va="center", fontsize=7.5)
    for y, (_, r) in zip(ys, bot.iterrows()):
        ax.text(r.z - .3, -y - 1.5, f"{zh(r.word)} ({gloss(r.word)})", va="center", ha="right", fontsize=7.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel(f"z  (blue: more in {left}; orange: more in {right})", fontsize=8)
    lim = max(abs(fw.z.iloc[0]), abs(fw.z.iloc[-1])) * 2.1
    ax.set_xlim(-lim, lim)


def run(d, text_col, tag):
    post = d.pub_date >= pd.Timestamp(RESTRUCTURE_DATE)
    g = {(desk, p): counts(d[(d.desk == desk) & (post == p)][text_col]) for desk in ["china", "world"] for p in [False, True]}
    background = counts(d[text_col])
    comps = {
        "china_vs_world_pre": (g["china", False], g["world", False]),
        "china_vs_world_post": (g["china", True], g["world", True]),
        "china_post_vs_pre": (g["china", True], g["china", False]),
        "world_post_vs_pre": (g["world", True], g["world", False]),
    }
    res = {}
    for name, (ci, cj) in comps.items():
        res[name] = fightin_words(ci, cj, background)
        res[name].to_csv(f"{RESULTS}/distinctive_words_{tag}_{name}.csv", index=False)
    # China-specific change: post-vs-pre shift on the China desk minus the same shift on the
    # World desk (difference of z's; descriptive ranking only)
    m = res["china_post_vs_pre"][["word", "z"]].merge(res["world_post_vs_pre"][["word", "z"]], on="word",
                                                         suffixes=("_china", "_world"))
    m["z_diff"] = m.z_china - m.z_world
    m = m.sort_values("z_diff", ascending=False)
    m.to_csv(f"{RESULTS}/distinctive_words_{tag}_china_specific_change.csv", index=False)

    print(f"\n################ {tag}")
    for name, r in res.items():
        print(f"\n== {name}: top 25 / bottom 25")
        print(" ".join(f"{w}({z:.1f})" for w, z in zip(r.word.head(25), r.z.head(25))))
        print(" ".join(f"{w}({z:.1f})" for w, z in zip(r.word.tail(25), r.z.tail(25))))
    print("\n== China-specific change (z_china - z_world), top 30:")
    print(" ".join(f"{w}({z:.1f})" for w, z in zip(m.word.head(30), m.z_diff.head(30))))
    print("== bottom 30:")
    print(" ".join(f"{w}({z:.1f})" for w, z in zip(m.word.tail(30), m.z_diff.tail(30))))
    return res


def main():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    tok = pd.read_parquet(f"{FEATURES}/tokens.parquet", columns=["article_id", "tokens", "lead_tokens"])
    art = art.merge(tok, on="article_id")
    rt = art[(art.corpus == "realtime") & art.fulltext_ok]
    res = run(rt, "tokens", "realtime")
    leads = art[(art.corpus == "print") & (art.pub_year >= 2016)]
    run(leads, "lead_tokens", "printleads")

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.8))
    plot_pair(axes[0], res["china_vs_world_pre"], "China vs World desk, 2013-Nov 2021", "China", "World")
    plot_pair(axes[1], res["china_vs_world_post"], "China vs World desk, Dec 2021-2025", "China", "World")
    plot_pair(axes[2], res["china_post_vs_pre"], "China desk: after vs before restructuring", "after", "before")
    plot_pair(axes[3], res["world_post_vs_pre"], "World desk: after vs before (comparison)", "after", "before")
    fig.suptitle("Figure 9. Most distinctive words, realtime corpus (weighted log-odds with informative Dirichlet prior)",
                 x=.01, ha="left")
    save(fig, "fig09_distinctive_words")


if __name__ == "__main__":
    main()
