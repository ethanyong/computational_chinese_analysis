"""Step 2 — Draw a stratified random sample from the article index.

Strata are (corpus, desk, publication year).  Within each stratum the articles are put in a
seeded random order and the first `draw` are downloaded.  After cleaning (step 4) the first
`target` usable articles *in that random order* are kept, which is a simple random sample of
the usable articles in the stratum (a random permutation filtered by a fixed rule stays a
random permutation).  Strata are defined by publication year (never capture year), and
every stratum reaches its target size even when some downloads fail.

Output: data/interim/sample_frame.csv
"""
import math
import numpy as np
import pandas as pd
from config import RAW, INTERIM, SEED, PRINT_YEARS, REALTIME_YEARS

TARGET = {"print": 300, "realtime": 200}   # usable articles per desk-year
OVERDRAW = {"print": 1.6, "realtime": 1.35}  # extra draws to absorb stubs/dead captures


def main():
    idx = pd.read_csv(f"{RAW}/cdx_index.csv", dtype={"timestamp": str})
    keep = ((idx.corpus == "print") & idx.pub_year.isin(PRINT_YEARS)) | \
           ((idx.corpus == "realtime") & idx.pub_year.isin(REALTIME_YEARS) & idx.desk.isin(["china", "world", "singapore"]))
    idx = idx[keep].copy()

    rng = np.random.default_rng(SEED)
    idx["rand"] = rng.random(len(idx))
    idx = idx.sort_values(["corpus", "desk", "pub_year", "rand"])
    idx["rank"] = idx.groupby(["corpus", "desk", "pub_year"]).cumcount()
    idx["population"] = idx.groupby(["corpus", "desk", "pub_year"])["article_id"].transform("size")
    idx["target"] = idx["corpus"].map(TARGET)
    draw = idx["corpus"].map(lambda c: math.ceil(TARGET[c] * OVERDRAW[c]))
    frame = idx[idx["rank"] < draw].drop(columns="rand")
    frame.to_csv(f"{INTERIM}/sample_frame.csv", index=False)

    print(f"Sample frame: {len(frame):,} articles to download")
    print(pd.crosstab([frame.corpus, frame.pub_year], frame.desk).to_string())


if __name__ == "__main__":
    main()
