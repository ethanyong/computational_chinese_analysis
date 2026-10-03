"""HTML decoding and article-body extraction for the three Zaobao site generations.

Design rules:
  1. Decode from raw bytes using the page's declared charset (GB2312/GBK -> GB18030 superset),
     never by re-encoding an already-decoded string.
  2. Take paragraphs only from the layout's article-body container; fall back to the
     "densest paragraph container" heuristic only if no known container exists.
  3. Flag (not silently keep) paywall teasers and print-edition stubs.
  4. Strip captions, bylines-only lines and paywall/subscription text.
Corpus-level boilerplate (paragraphs repeated across many articles) is removed later, in
step 4, because it can only be detected across documents.
"""
import re
from bs4 import BeautifulSoup

CHARSET_RE = re.compile(rb'charset\s*=\s*["\']?\s*([A-Za-z0-9_\-]+)', re.I)

# Known article-body containers, newest layout first
BODY_SELECTORS = [
    "div.article-content-rawhtml",   # 2016+ news and realtime
    "div.article-content-container", # 2016-2019 news (div#FineDining)
    "div.a_body",                    # 2013-2015 realtime
    "div#article-body", "div.article-body", "div.article-content", "td.text", "div#content",
]

PAYWALL_MARKERS = ("订户专享", "订阅或登录", "继续阅读全文", "登录后继续阅读", "订阅后继续阅读",
                   "开始订阅")   # 2016-2019 paywall prompt: "登录 | 免费试阅14天 | 开始订阅"
# Promotional lines appended to *complete* articles (e.g. realtime 2017: "想阅读更多新闻？
# 立即注册14天免费试阅"); stripped as non-prose but NOT evidence of truncation.
PROMO_MARKERS = ("免费试阅", "想阅读更多新闻", "立即注册")
STUB_MARKERS = ("全文详见", "详见今日出版", "全文刊登于", "欢迎读者购买或订阅")
# paragraphs that are not article prose
DROP_PARA = re.compile(
    r"^(请\s*$|订阅|登录|此文章为早报|已是早报订户|新用户体验价|字体大小|分享|相关新闻|热门|推荐|"
    r"早报新添|本文|欢迎读者|版权所有|新加坡报业控股|SPH Media).*"
)
DATE_LINE = re.compile(r"^[（(]?\s*\d{4}[-年./]\d{1,2}[-月./]\d{1,2}日?\s*[）)]?$")
CAPTION = re.compile(r"[（(][^（）()]{0,30}(摄|图|照片|互联网|档案照|截图|供图|美联社|法新社|路透社|彭博社|新华社|中新社)[^（）()]{0,10}[）)]\s*$")


def decode_html(raw: bytes) -> str:
    if not raw:
        return ""
    m = CHARSET_RE.search(raw[:4000])
    declared = m.group(1).decode("ascii", "ignore").lower() if m else ""
    if declared in ("gb2312", "gbk", "gb18030", "x-gbk", "cp936"):
        order = ["gb18030", "utf-8"]
    elif declared in ("big5", "big5-hkscs"):
        order = ["big5hkscs", "utf-8"]
    else:
        order = ["utf-8", "gb18030"]
    for enc in order:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: the declared/first codec with replacement characters, counted later
    return raw.decode(order[0], errors="replace")


def _paragraphs(node):
    paras = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    if not paras:  # some legacy containers hold <br>-separated text instead of <p>
        paras = [t.strip() for t in node.get_text("\n").split("\n")]
    return [re.sub(r"\s+", " ", p) for p in paras if p and p.strip()]


def _densest_container(soup):
    """Parent element whose direct <p> children carry the most text."""
    score = {}
    for p in soup.find_all("p"):
        par = p.parent
        score[id(par)] = (score.get(id(par), (0, par))[0] + len(p.get_text(strip=True)), par)
    if not score:
        return None
    return max(score.values(), key=lambda x: x[0])[1]


def extract(html: str) -> dict:
    """Return dict(title, paragraphs, layout, paywalled, stub)."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = re.split(r"\s*[|｜_]\s*", title)[0]
    full_text = soup.get_text(" ", strip=True)
    stub = any(m in full_text for m in STUB_MARKERS)
    # paywall prompts are detected inside the article container only, so that a site-wide
    # "subscribe" button in the page chrome cannot mark a free article as truncated
    body_raw = next((soup.select(sel)[0] for sel in BODY_SELECTORS if soup.select(sel)), soup)
    paywalled = any(m in body_raw.get_text(" ", strip=True) for m in PAYWALL_MARKERS)

    for tag in soup(["script", "style", "noscript", "iframe", "form", "nav", "header", "footer",
                     "figure", "figcaption", "aside", "button", "select"]):
        tag.decompose()

    node, layout = None, "densest"
    for sel in BODY_SELECTORS:
        found = soup.select(sel)
        if found:
            node, layout = found[0], sel
            break
    if node is None:
        node = _densest_container(soup)
    paras = _paragraphs(node) if node is not None else []

    clean = []
    for p in paras:
        if DROP_PARA.match(p) or CAPTION.search(p) and len(p) < 120:
            continue
        if p == title or DATE_LINE.match(p):          # legacy pages repeat headline and date
            continue
        if any(m in p for m in PAYWALL_MARKERS + STUB_MARKERS + PROMO_MARKERS):
            continue
        clean.append(p)
    return dict(title=title, paragraphs=clean, layout=layout, paywalled=paywalled, stub=stub)
