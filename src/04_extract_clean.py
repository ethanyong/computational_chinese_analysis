"""Step 4 — Decode, extract and clean article text; select the final analysis sample.

Exclusion rules, applied in this order and logged per article (results/exclusions.csv):
  fetch_failed   HTTP status != 200 or empty body
  decode_error   > 0.5% U+FFFD replacement characters after decoding
  print_stub     legacy print-edition teaser ("全文详见今日出版的…")
  too_short      < 50 CJK characters of visible body text after cleaning
  duplicate      identical body text already used by a lower-ranked article

Boilerplate: after extraction, any paragraph of >= 8 characters that appears in >= 5
different articles is treated as site template text and removed from every article.

Paywalled pages are kept but flagged `truncated` (their visible text is the article's lead).
`fulltext_ok` = not truncated and >= 150 CJK characters; full-text analyses use only those.
Lead analyses (first two paragraphs) use every sampled article, so they are free of the
selection that the paywall imposes on which articles have archived full text.

Selection: within each (corpus, desk, pub_year) stratum, the first TARGET usable articles in
the random order of step 2 form the analysis sample.

Output: data/interim/articles.parquet, results/exclusions.csv, results/sample_sizes.csv
"""
import glob
import hashlib
import re
from collections import Counter

import pandas as pd
from opencc import OpenCC
from config import RAW, INTERIM, RESULTS
import textutils as T

CJK = re.compile(r"[一-鿿]")
MIN_CJK = 50            # sample usability (visible text)
MIN_CJK_FULL = 150      # full-text analyses
BOILER_MIN_LEN, BOILER_MIN_DOCS = 8, 5
t2s = OpenCC("t2s").convert


def main():
    frame = pd.read_csv(f"{INTERIM}/sample_frame.csv", dtype={"timestamp": str})
    html = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/html/shard_*.parquet"))])
    html = html.drop_duplicates("article_id", keep="last")
    df = frame.merge(html, on="article_id", how="inner")
    print(f"{len(df):,} downloaded pages of {len(frame):,} in frame")

    recs = []
    for r in df.itertuples():
        rec = dict(article_id=r.article_id, reason=None, title="", paras=[], layout="",
                   truncated=False, stub=False, bad_chars=0.0)
        if r.status != 200 or not r.content:
            rec["reason"] = "fetch_failed"
        else:
            text = T.decode_html(r.content)
            rec["bad_chars"] = text.count("�") / max(len(text), 1)
            if rec["bad_chars"] > 0.005:
                rec["reason"] = "decode_error"
            else:
                ex = T.extract(text)
                rec.update(title=t2s(ex["title"]), paras=[t2s(p) for p in ex["paragraphs"]],
                           layout=ex["layout"], truncated=ex["paywalled"], stub=ex["stub"])
                if ex["stub"]:
                    rec["reason"] = "print_stub"
        recs.append(rec)
    ex = pd.DataFrame(recs)

    # corpus-level boilerplate: paragraphs shared by many distinct articles
    counts = Counter(p for ps in ex.paras for p in set(ps) if len(p) >= BOILER_MIN_LEN)
    boiler = {p for p, c in counts.items() if c >= BOILER_MIN_DOCS}
    pd.Series(sorted(boiler)).to_csv(f"{RESULTS}/boilerplate_paragraphs_removed.csv", index=False, header=["paragraph"])
    print(f"{len(boiler)} boilerplate paragraphs removed corpus-wide")
    ex["text"] = ex.paras.map(lambda ps: "\n".join(p for p in ps if p not in boiler))
    ex["n_cjk"] = ex.text.map(lambda t: len(CJK.findall(t)))
    ex.loc[ex.reason.isna() & (ex.n_cjk < MIN_CJK), "reason"] = "too_short"

    df = df.drop(columns=["content"]).merge(ex.drop(columns="paras"), on="article_id")
    df = df.sort_values(["corpus", "desk", "pub_year", "rank"])
    df["hash"] = df.text.map(lambda t: hashlib.md5(t.encode()).hexdigest())
    dup = df.reason.isna() & df.duplicated("hash", keep="first")
    df.loc[dup, "reason"] = "duplicate"

    usable = df.reason.isna()
    df["usable_rank"] = -1
    df.loc[usable, "usable_rank"] = df[usable].groupby(["corpus", "desk", "pub_year"]).cumcount()
    df["in_sample"] = usable & (df.usable_rank < df.target)

    excl = df.groupby(["corpus", "pub_year", "desk"]).reason.value_counts().unstack(fill_value=0)
    excl.to_csv(f"{RESULTS}/exclusions.csv")
    print("\nExclusions by reason:\n", df.reason.value_counts(dropna=False).to_string())

    out = df[df.in_sample].drop(columns=["reason", "usable_rank", "hash", "in_sample"])
    out["pub_date"] = pd.to_datetime(out.pub_date, errors="coerce")
    out["fulltext_ok"] = ~out.truncated & (out.n_cjk >= MIN_CJK_FULL)
    out.to_parquet(f"{INTERIM}/articles.parquet", index=False)
    sizes = pd.crosstab([out.corpus, out.pub_year], out.desk)
    trunc = out.groupby(["corpus", "pub_year", "desk"]).truncated.mean().unstack()
    full = pd.crosstab([out.corpus, out.pub_year], out.desk, values=out.fulltext_ok, aggfunc="sum")
    full.to_csv(f"{RESULTS}/sample_sizes_fulltext.csv")
    print("\nFull-text usable articles:\n", full.to_string())
    sizes.to_csv(f"{RESULTS}/sample_sizes.csv")
    trunc.round(3).to_csv(f"{RESULTS}/paywall_truncation_share.csv")
    print(f"\nFinal sample: {len(out):,} articles\n", sizes.to_string())
    print("\nShare paywall-truncated:\n", trunc.round(2).to_string())


if __name__ == "__main__":
    main()
