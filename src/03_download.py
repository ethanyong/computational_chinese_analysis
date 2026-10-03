"""Step 3 — Download the archived HTML of every sampled article from the Wayback Machine.

* Uses the `id_` playback mode, which returns the original bytes without the Wayback toolbar.
* Stores raw bytes (no decoding here — decoding is done in step 4 from the declared charset),
  so the GB2312/GBK legacy pages are decoded exactly once, from their own declaration.
* Resumable: shards are written every 500 pages and already-fetched ids are skipped.
* Polite: 6 concurrent requests, exponential back-off on 429/5xx.

Output: data/raw/html/shard_*.parquet  (article_id, status, final_url, content)
"""
import asyncio
import glob
import os
import time

import aiohttp
import pandas as pd
from config import RAW, INTERIM, USER_AGENT

OUT = f"{RAW}/html"
os.makedirs(OUT, exist_ok=True)
CONCURRENCY = 6
SHARD = 500


def done_ids():
    ids = set()
    for f in glob.glob(f"{OUT}/shard_*.parquet"):
        ids |= set(pd.read_parquet(f, columns=["article_id"]).article_id)
    return ids


async def fetch(session, sem, row):
    url = f"https://web.archive.org/web/{row.timestamp}id_/{row.original}"
    async with sem:
        for attempt in range(6):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    body = await r.read()
                    if r.status in (429, 502, 503, 504):
                        raise aiohttp.ClientResponseError(r.request_info, (), status=r.status)
                    return dict(article_id=row.article_id, status=r.status, final_url=str(r.url), content=body)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(min(60, 3 * 2 ** attempt))
        return dict(article_id=row.article_id, status=-1, final_url=url, content=b"")


async def run(todo):
    sem = asyncio.Semaphore(CONCURRENCY)
    shard_no = len(glob.glob(f"{OUT}/shard_*.parquet"))
    t0 = time.time()
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        for start in range(0, len(todo), SHARD):
            chunk = todo.iloc[start:start + SHARD]
            res = await asyncio.gather(*[fetch(session, sem, r) for r in chunk.itertuples()])
            pd.DataFrame(res).to_parquet(f"{OUT}/shard_{shard_no:04d}.parquet", compression="zstd",
                                         compression_level=12, index=False)
            shard_no += 1
            n = start + len(chunk)
            ok = sum(r["status"] == 200 for r in res)
            print(f"{n:,}/{len(todo):,} done ({ok}/{len(chunk)} ok in shard) "
                  f"{n / (time.time() - t0):.1f}/s", flush=True)


def main():
    frame = pd.read_csv(f"{INTERIM}/sample_frame.csv", dtype={"timestamp": str})
    # download the lowest ranks first, so a partial run still yields balanced strata
    frame = frame.sort_values(["rank", "corpus", "desk", "pub_year"])
    todo = frame[~frame.article_id.isin(done_ids())]
    print(f"{len(todo):,} pages to fetch ({len(frame) - len(todo):,} already done)")
    asyncio.run(run(todo))


if __name__ == "__main__":
    main()
