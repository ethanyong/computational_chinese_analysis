"""Epistemic / deontic / evidential marker lexicon.

Matching is on *jieba word tokens*, not substrings, so 据 does not match 数据 (data), 说 does
not match 说明 (explain) and 不得 does not match 不得不 (have to).  Multi-character markers are
added to jieba's dictionary so they are kept as single tokens; the fixed evidential phrase
据报道 ("it is reported") is one token, so it counts once, not as 据 + 报道.

Categories follow Palmer (2001) for modality and Hyland (1998, 2005) for hedges/boosters in
news and academic discourse.  Inclusion decisions, with reasons:
  * Negation / difficulty words such as 并非 ("is not"), 难以 ("hard to") and 视情况 ("depending
    on the situation") are NOT hedges: they do not express epistemic possibility.
  * 无疑 ("undoubtedly") is a BOOSTER (certainty), not an obligation word.
  * 能够 ("be able to") is dynamic ability, not deontic permission, so it is excluded.
  * A BOOSTER category is included: a hedge rate on its own cannot distinguish "less hedging"
    from "more assertion"; the hedge/booster pair is the standard design (Hyland 1998).
  * Highly ambiguous items 一定 (also "a certain degree", itself a hedge) and 肯定 (also the
    verb "affirm") are excluded from BOOSTER.
Known residual ambiguity (reported in per-term results, so readers can discount them):
  应该 is deontic ("should") or epistemic ("ought to be / probably"); 可以 is permission or
  ability; 称/说 are neutral reporting verbs.
"""

LEXICON = {
    "hedge": ["可能", "可能性", "或许", "也许", "似乎", "好像", "大概", "恐怕", "未必",
              "不一定", "不排除", "或将", "估计", "看来"],
    "booster": ["必然", "显然", "无疑", "毫无疑问", "确实", "必将", "势必", "绝对", "当然",
                "显而易见", "毋庸置疑", "不可否认"],
    "obligation": ["必须", "应当", "应该", "务必", "不得", "理应", "不应", "有必要", "有责任"],
    "permission": ["可以", "允许", "准许", "获准", "有权"],
    "attribution": ["据", "据悉", "据报", "据报道", "据称", "表示", "指出", "称", "说", "认为", "强调",
                    "透露", "报道", "引述", "声称", "宣称", "回应", "告诉", "坦言", "直言"],
}

TERM_TO_CAT = {t: c for c, ts in LEXICON.items() for t in ts}
MULTI_CHAR_TERMS = [t for t in TERM_TO_CAT if len(t) >= 3]
CATEGORIES = list(LEXICON)
