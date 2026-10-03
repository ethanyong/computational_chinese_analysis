"""English glosses for every Chinese word shown in a figure (keys are simplified Chinese, as in
the analysis).  Figures label a word as  中文 (gloss),  so no Chinese appears untranslated."""

GLOSS = {
    # ---- marker lexicon (lexicon.py)
    "可能": "possibly", "可能性": "possibility", "或许": "perhaps", "也许": "maybe", "似乎": "seemingly",
    "好像": "seem", "大概": "probably", "恐怕": "I fear", "未必": "not necessarily", "不一定": "not certain",
    "不排除": "cannot rule out", "或将": "may", "估计": "estimate", "看来": "it seems",
    "必然": "inevitably", "显然": "obviously", "无疑": "undoubtedly", "毫无疑问": "without doubt",
    "确实": "indeed", "必将": "will surely", "势必": "bound to", "绝对": "absolutely", "当然": "of course",
    "显而易见": "evidently", "毋庸置疑": "beyond doubt", "不可否认": "undeniably",
    "必须": "must", "应当": "ought to", "应该": "should", "务必": "be sure to", "不得": "must not",
    "理应": "ought to", "不应": "should not", "有必要": "necessary", "有责任": "responsible",
    "可以": "may / can", "允许": "allow", "准许": "permit", "获准": "be permitted", "有权": "have the right",
    "据": "according to", "据悉": "it is learned", "据报": "reportedly", "据报道": "it is reported",
    "据称": "allegedly", "表示": "said", "指出": "pointed out", "称": "claimed", "说": "say",
    "认为": "believe", "强调": "stress", "透露": "reveal", "报道": "report", "引述": "quote",
    "声称": "assert", "宣称": "declare", "回应": "respond", "告诉": "tell", "坦言": "admit",
    "直言": "say bluntly",
    # ---- distinctive words (Figure 8)
    "中国": "China", "台湾": "Taiwan", "香港": "Hong Kong", "大陆": "mainland", "发展": "development",
    "北京": "Beijing", "企业": "enterprise", "人民币": "renminbi", "市场": "market",
    "亿元": "100 million yuan", "万元": "10,000 yuan", "昨天": "yesterday", "今天": "today",
    "今日": "today", "当天": "that day", "主任": "director", "两岸": "cross-Strait",
    "习近平": "Xi Jinping", "冠病": "COVID-19", "死亡": "death", "袭击": "attack", "他们": "they",
    "我们": "we", "总统": "president", "马来西亚": "Malaysia", "英国": "Britain", "俄罗斯": "Russia",
    "特朗普": "Trump", "日本": "Japan", "韩国": "South Korea", "美国": "United States",
    "朝鲜": "North Korea", "早报讯": "Zaobao dateline", "网讯": "online dateline",
    "联合早报": "Lianhe Zaobao", "工作": "work", "相关": "related", "中方": "the Chinese side",
    "通报": "official bulletin", "官网": "official website", "合作": "cooperation", "普京": "Putin",
    "法新社": "AFP", "路透社": "Reuters", "加沙": "Gaza", "美元": "US dollar", "拜登": "Biden",
    "以色列": "Israel", "乌克兰": "Ukraine", "哈马斯": "Hamas", "战争": "war",
    "星期一": "Monday", "星期二": "Tuesday", "星期三": "Wednesday", "星期四": "Thursday",
    "星期五": "Friday", "星期六": "Saturday", "星期天": "Sunday", "星期日": "Sunday",
    "综合": "compiled reports", "核酸": "nucleic-acid test", "住房": "housing",
    "赖清德": "Lai Ching-te", "李家": "John Lee", "李家超": "John Lee Ka-chiu", "了解": "understand",
    "林郑": "Carrie Lam", "南海": "South China Sea", "台北": "Taipei", "明报": "Ming Pao",
    "警方": "police", "记者": "reporter", "马英九": "Ma Ying-jeou", "病例": "cases", "安倍": "Abe",
    "韩朝": "the two Koreas", "确诊": "confirmed cases", "奥巴马": "Obama", "金正恩": "Kim Jong-un",
    "隔离": "quarantine", "检测": "testing", "防疫": "epidemic control", "经济": "economy",
    "关税": "tariffs", "中央社": "Central News Agency", "游行": "protest march",
}


def gloss(word: str) -> str:
    if word not in GLOSS:
        raise KeyError(f"no English gloss for {word!r}: add it to src/glosses.py")
    return GLOSS[word]


# Hanyu pinyin without tones for the marker lexicon (shown in italics in Figure 7)
PINYIN = {
    "可能": "keneng", "可能性": "kenengxing", "或许": "huoxu", "也许": "yexu", "似乎": "sihu",
    "好像": "haoxiang", "大概": "dagai", "恐怕": "kongpa", "未必": "weibi", "不一定": "bu yiding",
    "不排除": "bu paichu", "或将": "huojiang", "估计": "guji", "看来": "kanlai",
    "必然": "biran", "显然": "xianran", "无疑": "wuyi", "毫无疑问": "haowu yiwen", "确实": "queshi",
    "必将": "bijiang", "势必": "shibi", "绝对": "juedui", "当然": "dangran", "显而易见": "xian'er yijian",
    "毋庸置疑": "wuyong zhiyi", "不可否认": "bukefouren",
    "必须": "bixu", "应当": "yingdang", "应该": "yinggai", "务必": "wubi", "不得": "bude", "理应": "liying",
    "不应": "buying", "有必要": "you biyao", "有责任": "you zeren",
    "可以": "keyi", "允许": "yunxu", "准许": "zhunxu", "获准": "huozhun", "有权": "youquan",
    "据": "ju", "据悉": "juxi", "据报": "jubao", "据报道": "ju baodao", "据称": "jucheng", "表示": "biaoshi",
    "指出": "zhichu", "称": "cheng", "说": "shuo", "认为": "renwei", "强调": "qiangdiao", "透露": "toulu",
    "报道": "baodao", "引述": "yinshu", "声称": "shengcheng", "宣称": "xuancheng", "回应": "huiying",
    "告诉": "gaosu", "坦言": "tanyan", "直言": "zhiyan",
}


def pinyin_italic(word: str) -> str:
    """Pinyin as matplotlib math text, so it renders in italics inside a plain label."""
    if word not in PINYIN:
        raise KeyError(f"no pinyin for {word!r}: add it to src/glosses.py")
    return "$\\mathit{" + PINYIN[word].replace(" ", "\\ ").replace("'", "’") + "}$"
