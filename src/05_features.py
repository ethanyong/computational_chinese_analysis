"""Step 5 — Tokenize and build article-level feature tables.

Tokenization: jieba (precise mode, with part-of-speech tags) on simplified-Chinese text.
Only tokens containing at least one CJK character are kept, so punctuation, numbers and
Latin strings do not dilute the rates.  `n_tok` (number of kept tokens) is the exposure
used as the offset in every count model.

Outputs (data/features/):
  tokens.parquet        article_id, tokens (space-joined), lead_tokens (first two paragraphs)
  markers.parquet       per-term and per-category marker counts, for full text and lead
  function_words.parquet relative frequency of the 300 most frequent function words
  function_vocab.csv    those 300 words with their dominant POS tag and corpus frequency
"""
import re
from collections import Counter, defaultdict

import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from config import INTERIM, FEATURES
from lexicon import LEXICON, TERM_TO_CAT, MULTI_CHAR_TERMS

jieba.setLogLevel(60)
for t in MULTI_CHAR_TERMS + ["或将", "据称", "据报"]:
    jieba.add_word(t, freq=200000)
# 据报道 ("it is reported") must be one token: otherwise the high-frequency 据报 entry above
# yields 据报 / 道 and leaves a spurious single-character token 道
jieba.add_word("据报道", freq=300000)

CJK = re.compile(r"[一-鿿]")
# jieba POS tags treated as function words (closed grammatical classes): adverbs,
# conjunctions, prepositions, auxiliaries/particles, modal particles.  Pronouns (r*) and
# localizers (f) are excluded because they are referential and leak topic (e.g. 我国 "our
# country", 本地 "local" = Singapore), which would let a "style" classifier use subject matter.
FUNCTION_POS = {"d", "c", "p", "u", "uj", "ul", "uz", "ug", "ud", "uv", "y"}
N_FUNCTION_WORDS = 300


def tokenize(text):
    toks, pos = [], []
    for w, f in pseg.cut(text):
        if CJK.search(w):
            toks.append(w)
            pos.append(f)
    return toks, pos


def main():
    art = pd.read_parquet(f"{INTERIM}/articles.parquet")
    print(f"Tokenizing {len(art):,} articles ...")
    tok_rows, pos_count = [], defaultdict(Counter)
    for i, r in enumerate(art.itertuples()):
        paras = r.text.split("\n")
        toks, pos = tokenize(r.text)
        lead, _ = tokenize("\n".join(paras[:2]))
        for w, f in zip(toks, pos):
            pos_count[w][f] += 1
        tok_rows.append(dict(article_id=r.article_id, tokens=" ".join(toks), lead_tokens=" ".join(lead),
                             n_tok=len(toks), n_tok_lead=len(lead)))
        if i % 2000 == 0:
            print(f"   {i:,}", flush=True)
    tok = pd.DataFrame(tok_rows)
    tok.to_parquet(f"{FEATURES}/tokens.parquet", index=False)

    # ---- marker counts (token-exact matching) ----
    terms = list(TERM_TO_CAT)
    def count_terms(s):
        c = Counter(s.split())
        return [c.get(t, 0) for t in terms]
    full = pd.DataFrame(tok.tokens.map(count_terms).tolist(), columns=[f"t_{t}" for t in terms])
    lead = pd.DataFrame(tok.lead_tokens.map(count_terms).tolist(), columns=[f"lead_t_{t}" for t in terms])
    mk = pd.concat([tok[["article_id", "n_tok", "n_tok_lead"]], full, lead], axis=1)
    for cat, ts in LEXICON.items():
        mk[cat] = mk[[f"t_{t}" for t in ts]].sum(axis=1)
        mk[f"lead_{cat}"] = mk[[f"lead_t_{t}" for t in ts]].sum(axis=1)
    mk.to_parquet(f"{FEATURES}/markers.parquet", index=False)

    # ---- function-word relative frequencies ----
    total = Counter({w: sum(c.values()) for w, c in pos_count.items()})
    dominant = {w: c.most_common(1)[0][0] for w, c in pos_count.items()}
    fw = [w for w, _ in total.most_common() if dominant[w] in FUNCTION_POS][:N_FUNCTION_WORDS]
    pd.DataFrame({"word": fw, "pos": [dominant[w] for w in fw], "count": [total[w] for w in fw]}) \
        .to_csv(f"{FEATURES}/function_vocab.csv", index=False)
    idx = {w: j for j, w in enumerate(fw)}
    X = np.zeros((len(tok), len(fw)), dtype=np.float32)
    for i, s in enumerate(tok.tokens):
        ws = s.split()
        for w in ws:
            j = idx.get(w)
            if j is not None:
                X[i, j] += 1
        X[i] /= max(len(ws), 1)
    fdf = pd.DataFrame(X, columns=fw)
    fdf.insert(0, "article_id", tok.article_id.values)
    fdf.to_parquet(f"{FEATURES}/function_words.parquet", index=False)

    # sanity: lexicon terms must actually occur as jieba tokens
    hits = mk[[f"t_{t}" for t in terms]].sum().rename(lambda c: c[2:])
    print("\nToken-level lexicon hits (whole corpus):\n", hits.sort_values(ascending=False).to_string())
    print("\nFunction words (first 40):", " ".join(fw[:40]))


if __name__ == "__main__":
    main()
