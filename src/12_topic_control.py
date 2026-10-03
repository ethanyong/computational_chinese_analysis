"""Step 12 — Is the China desk's lower hedging a matter of voice or of topic?

The World desk covers wars, elections and markets, stories that may invite speculation; the
China desk covers policy announcements and domestic news.  If the hedging gap only reflected
that difference in subject matter, it should disappear when articles on the same topic are
compared.  This step adds topic fixed effects to the long-run rate-ratio model.

Topics: non-negative matrix factorization (NMF) of TF-IDF article vectors (words of >= 2
characters, 5,000 most frequent, words in > 50% of articles dropped), fitted separately on the
realtime corpus (2013-25) and the legacy desk corpus (2006-12), full texts only.  Each article
is assigned its dominant topic.  K = 20 and K = 40 are both reported, so that no result depends
on one choice of K.  Topic models are fitted on all desks' articles together and never see desk
labels.

Models (Poisson QMLE, log tokens offset, HC1 SEs), China vs World:
  base          marker ~ china + year FE
  topic K       marker ~ china + year FE + topic FE
  shared topics marker ~ china + year FE + topic FE, only topics in which each of the two desks
                supplies at least 20% of the articles (topics both desks actually cover)
Plus the restructuring DiD (realtime, hedge and all categories) with topic FE added, in case a
post-2021 change in topic mix (e.g. COVID) hides or creates a change in stance.

Outputs: results/topic_controlled_gaps.csv, results/topic_controlled_did.csv,
         results/topics_top_words.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from config import INTERIM, FEATURES, RESULTS, SEED, RESTRUCTURE_DATE
from lexicon import CATEGORIES

KS = (20, 40)
SHARED_MIN = 0.2


def two_plus(s):
    return [w for w in s.split() if len(w) >= 2]


def fit_topics(d, tag):
    tf = TfidfVectorizer(tokenizer=two_plus, lowercase=False, token_pattern=None, max_features=5000,
                         sublinear_tf=True, min_df=5, max_df=0.5)
    X = tf.fit_transform(d.tokens)
    vocab = np.array(tf.get_feature_names_out())
    words = []
    for K in KS:
        nmf = NMF(K, init="nndsvda", random_state=SEED, max_iter=500)
        d[f"topic{K}"] = nmf.fit_transform(X).argmax(1)
        for k, comp in enumerate(nmf.components_):
            words.append(dict(corpus=tag, K=K, topic=k, n_articles=int((d[f"topic{K}"] == k).sum()),
                              share_china=(d.loc[d[f"topic{K}"] == k, "desk"] == "china").mean(),
                              top_words=" ".join(vocab[np.argsort(comp)[::-1][:12]])))
    return d, words


def irr(formula, d):
    m = smf.glm(formula, d, family=sm.families.Poisson(), offset=np.log(d.n_tok)).fit(cov_type="HC1")
    term = "china" if "china" in m.params else "china_post"
    b, se = m.params[term], m.bse[term]
    return dict(irr=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues[term], n=len(d))


def main():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    mk = pd.read_parquet(f"{FEATURES}/markers.parquet").drop(columns=["n_tok_lead"])
    tok = pd.read_parquet(f"{FEATURES}/tokens.parquet", columns=["article_id", "tokens"])
    df = art.merge(mk, on="article_id").merge(tok, on="article_id")
    df = df[df.fulltext_ok & (df.n_tok > 0)]
    corpora = {"realtime 2013-25": df[df.corpus == "realtime"],
               "desk 2006-12": df[(df.corpus == "print") & (df.pub_year <= 2012)]}

    gaps, dids, words = [], [], []
    for tag, d in corpora.items():
        d, w = fit_topics(d.reset_index(drop=True).copy(), tag)
        words += w
        cw = d[d.desk.isin(["china", "world"])].copy()
        cw["china"] = (cw.desk == "china").astype(int)
        for cat in CATEGORIES:
            specs = [("year FE only", cw, f"{cat} ~ china + C(pub_year)")]
            for K in KS:
                specs.append((f"year FE + {K} topics", cw, f"{cat} ~ china + C(pub_year) + C(topic{K})"))
            share = cw.groupby("topic40").china.mean()
            shared = share[(share >= SHARED_MIN) & (share <= 1 - SHARED_MIN)].index
            specs.append(("shared topics only (K = 40)", cw[cw.topic40.isin(shared)],
                          f"{cat} ~ china + C(pub_year) + C(topic40)"))
            for name, dd, f in specs:
                gaps.append(dict(corpus=tag, category=cat, spec=name, **irr(f, dd)))
        if tag.startswith("realtime"):
            cw["china_post"] = ((cw.desk == "china") & (cw.pub_date >= pd.Timestamp(RESTRUCTURE_DATE))).astype(int)
            for cat in CATEGORIES:
                for name, f in [("desk + year FE", f"{cat} ~ C(desk) + C(pub_year) + china_post"),
                                ("desk + year + 40 topic FE", f"{cat} ~ C(desk) + C(pub_year) + C(topic40) + china_post")]:
                    dids.append(dict(category=cat, spec=name, **irr(f, cw)))

    gaps, dids, words = pd.DataFrame(gaps), pd.DataFrame(dids), pd.DataFrame(words)
    gaps.to_csv(f"{RESULTS}/topic_controlled_gaps.csv", index=False)
    dids.to_csv(f"{RESULTS}/topic_controlled_did.csv", index=False)
    words.to_csv(f"{RESULTS}/topics_top_words.csv", index=False)
    pd.set_option("display.width", 220)
    print(gaps.round(3).to_string())
    print("\nRestructuring DiD with topic FE:\n", dids.round(3).to_string())
    # share of the base log-gap that topic composition accounts for (hedge)
    for tag in corpora:
        h = gaps[(gaps.corpus == tag) & (gaps.category == "hedge")].set_index("spec").irr
        for K in KS:
            frac = 1 - np.log(h[f"year FE + {K} topics"]) / np.log(h["year FE only"])
            print(f"{tag}: topics (K={K}) account for {frac:.0%} of the log hedging gap")


if __name__ == "__main__":
    main()
