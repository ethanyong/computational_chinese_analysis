"""Shared configuration for the Zaobao China-desk pipeline.

All paths are relative to the project folder (the parent of src/); every script chdirs there on import.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATA = "data"
RAW = f"{DATA}/raw"            # CDX indexes and downloaded HTML
INTERIM = f"{DATA}/interim"    # cleaned text, samples
FEATURES = f"{DATA}/features"  # article-level feature tables
RESULTS = "results"            # CSV/TXT outputs of every statistical test
FIGURES = "figures"
for d in (RAW, INTERIM, FEATURES, RESULTS, FIGURES):
    os.makedirs(d, exist_ok=True)

USER_AGENT = "EAS407-research/2.0 (Princeton course project; academic, low-rate)"
SEED = 20260407

DESKS = ["china", "world", "singapore", "sea"]
LEGACY_CODES = {"zg": "china", "gj": "world", "sp": "singapore", "yx": "sea"}

# Study windows (publication years, taken from the URL — never from the capture date)
PRINT_YEARS = list(range(2006, 2013)) + list(range(2016, 2026))
REALTIME_YEARS = list(range(2013, 2026))

# Events used as candidate breakpoints
EVENTS = {"ThinkChina launch": 2019.9, "SPH Media restructuring": 2021.95, "Washington Post report": 2023.55}
RESTRUCTURE_DATE = "2021-12-01"   # SPH Media Trust (CLG) took over the media business 1 Dec 2021

# --- URL grammar -----------------------------------------------------------------
# Modern desk articles:   /news/{desk}/story{YYYYMMDD}-{id}
# Realtime articles:      /realtime/{desk}/story{YYYYMMDD}-{id}
# Legacy desk articles:   /{code}/{code}{YYMMDD}_{nnn}[_{page}].shtml|html   (code in zg/gj/sp/yx)
RE_MODERN = re.compile(r"zaobao\.com(?:\.sg)?(?::80)?/(news|realtime)/(china|world|singapore|sea|seasia)/story(\d{8})-(\d+)")
RE_LEGACY = re.compile(r"zaobao\.com(?:\.sg)?(?::80)?/(zg|gj|sp|yx)/(zg|gj|sp|yx)(\d{6})_(\d+)(?:_(\d+))?\.s?html")


def parse_article_url(url: str):
    """Return dict(corpus, desk, pub_date, article_id, page) or None for non-article URLs.

    The article_id is a canonical key so that http/https, :80, www, .com/.com.sg and
    query-string variants (utm_source, ref, amp) of the same article collapse to one.
    """
    u = str(url).lower()
    m = RE_MODERN.search(u)
    if m:
        kind, desk, ymd, num = m.groups()
        desk = "sea" if desk == "seasia" else desk
        corpus = "print" if kind == "news" else "realtime"
        return dict(corpus=corpus, desk=desk, pub_date=f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
                    article_id=f"{kind}-{desk}-{num}", page=0)
    m = RE_LEGACY.search(u)
    if m:
        c1, c2, yymmdd, num, page = m.groups()
        if c1 != c2:          # e.g. /gj/zg0011... are mis-filed cross links; skip
            return None
        yy = int(yymmdd[:2])
        year = 2000 + yy if yy < 50 else 1900 + yy
        return dict(corpus="print", desk=LEGACY_CODES[c1],
                    pub_date=f"{year}-{yymmdd[2:4]}-{yymmdd[4:]}",
                    article_id=f"legacy-{c1}{yymmdd}_{num}", page=int(page) if page else 0)
    return None
