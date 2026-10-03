"""Step 1 — Harvest the Wayback Machine CDX index for every Zaobao article URL family.

Design
  * Only article URL grammars are kept (no daily section-index pages like /news/china/20141003).
  * No capture-date window: an article's year comes from the date in its URL, so captures
    from any year are harvested and the *publication* year is assigned afterwards.
  * Paginated CDX queries with retries, so large prefixes are not silently truncated.
  * Query-string / protocol / domain variants are collapsed to one canonical article id.

Output: data/raw/cdx_index.csv  (one row per canonical article, earliest 200 capture kept)
"""
import time
import requests
import pandas as pd
from config import RAW, USER_AGENT, parse_article_url

CDX = "https://web.archive.org/cdx/search/cdx"

PREFIXES = (
    # legacy desk articles (both domains were used 2006-2012)
    [f"{dom}/{c}/" for dom in ("zaobao.com", "zaobao.com.sg") for c in ("zg", "gj", "sp", "yx")]
    # modern desk articles
    + [f"zaobao.com.sg/news/{d}/story" for d in ("china", "world", "singapore", "sea")]
    # realtime (instant news) articles — the only article-level archive for 2013-2015
    + [f"zaobao.com.sg/realtime/{d}/story" for d in ("china", "world", "singapore")]
)

S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def get(params, tries=6):
    for k in range(tries):
        try:
            r = S.get(CDX, params=params, timeout=180)
            if r.status_code == 200:
                return r.text
            print(f"      HTTP {r.status_code}; retry {k + 1}")
        except requests.RequestException as e:
            print(f"      {type(e).__name__}; retry {k + 1}")
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"CDX failed for {params}")


def harvest(prefix, batch=25000):
    """Stream the full result set in batches using the CDX resumeKey protocol."""
    base = dict(url=prefix, matchType="prefix", filter=["statuscode:200", "mimetype:text/html"],
                fl="timestamp,original", collapse="urlkey", limit=batch, showResumeKey="true")
    rows, key, calls = [], None, 0
    while True:
        params = {**base, **({"resumeKey": key} if key else {})}
        lines = get(params).splitlines()
        calls += 1
        # a resume key, if any, follows a blank line at the end of the response
        key = None
        if len(lines) >= 2 and lines[-2] == "":
            key = lines[-1].strip()
            lines = lines[:-2]
        rows += [l.split(" ", 1) for l in lines if " " in l]
        if not key:
            break
        time.sleep(1)
    print(f"   {prefix}: {calls} calls, {len(rows)} url-key rows", flush=True)
    return rows


def main():
    rows = []
    for pre in PREFIXES:
        rows += harvest(pre)
    df = pd.DataFrame(rows, columns=["timestamp", "original"]).drop_duplicates()
    parsed = df["original"].map(parse_article_url)
    df = df[parsed.notna()].copy()
    meta = pd.DataFrame(list(parsed.dropna()), index=df.index)
    df = pd.concat([df, meta], axis=1)
    # multi-page legacy articles: keep the base page (no suffix) or the lowest-numbered page
    df["timestamp"] = df["timestamp"].astype(str)
    # one row per article: prefer the earliest capture (closest to publication)
    df = df.sort_values(["page", "timestamp"]).drop_duplicates("article_id", keep="first")
    df["pub_year"] = df["pub_date"].str[:4].astype(int)
    df.to_csv(f"{RAW}/cdx_index.csv", index=False)
    print(f"\nSaved {len(df):,} unique articles")
    print(pd.crosstab([df.corpus, df.pub_year], df.desk).to_string())


if __name__ == "__main__":
    main()
