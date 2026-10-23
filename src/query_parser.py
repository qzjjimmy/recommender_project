"""智能推荐的"查询理解"：把中文自然语言查询解析成结构化约束。

支持三类约束：

    价格  —— "100块以内"、"不超过50元"、"20到80元"、"50元以上"
    品类  —— "家居装饰"、"家居"、"厨房用品"、"袋子"（含常见别名）
    关键词 —— 其余中文/英文词，用于在商品名称里做匹配

设计取舍：用**规则解析**而不是大模型。原因是本项目没有外网与模型依赖，
规则解析可解释、可测试、零延迟，而且解析结果能直接回显给用户
（"已识别：品类=家居装饰，价格≤100"），方便排查"为什么没搜到"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# 类别别名 → 规范类目名（规范名以 products.csv 为准）
CATEGORY_ALIASES: dict[str, str] = {
    "家居": "家居装饰",
    "家装": "家居装饰",
    "装饰": "装饰品",
    "摆件": "装饰品",
    "厨房": "厨房用品",
    "餐厨": "厨房用品",
    "餐具": "厨房用品",
    "烘焙": "厨房用品",
    "袋子": "购物袋",
    "手提袋": "购物袋",
    "购物袋": "购物袋",
    "包": "购物袋",
    "派对": "派对用品",
    "聚会": "派对用品",
    "庆典": "派对用品",
    "收纳": "收纳用品",
    "储物": "收纳用品",
    "整理": "收纳用品",
    "手工": "手工用品",
    "diy": "手工用品",
    "生活": "生活用品",
    "日用": "生活用品",
    "首饰": "饰品",
    "配件": "饰品",
    "园艺": "园艺用品",
    "花园": "园艺用品",
    "植物": "园艺用品",
    "宠物": "宠物用品",
    "文具": "文具用品",
    "办公": "办公用品",
    "玩具": "玩具",
    "婴儿": "婴儿用品",
    "家具": "家具",
    # ---- v2 新增类目的别名（3C 品牌类目 + 校园类目）----
    # 注意：这里**不写品牌名**（苹果/华为/小米…）。品牌名交给"关键词"路径处理，
    # 因为它会按商品名匹配 —— "华为手机" = 品类「手机」+ 关键词「华为」，
    # 比把品牌硬映射到某个类目更准（小米既做手机也做家电）。
    "电脑": "笔记本电脑",
    "笔记本": "笔记本电脑",
    "手提电脑": "笔记本电脑",
    "平板": "平板电脑",
    "手表": "智能穿戴",
    "手环": "智能穿戴",
    "穿戴": "智能穿戴",
    "耳机": "耳机音响",
    "音箱": "耳机音响",
    "相机": "摄影摄像",
    "单反": "摄影摄像",
    "游戏机": "游戏设备",
    "数码": "数码配件",
    "充电宝": "数码配件",
    "智能家居": "智能设备",
    "家电": "家用电器",
    "宿舍": "宿舍用品",
    "寝室": "宿舍用品",
    "教辅": "图书音像",
    "教材": "图书音像",
    "图书": "图书音像",
    "露营": "运动户外",
    "户外": "运动户外",
    "护肤": "美妆护肤",
    "化妆品": "美妆护肤",
    "母婴": "母婴用品",
    "汽车": "汽车用品",
    "车品": "汽车用品",
    "零食": "食品饮料",
    "饮料": "食品饮料",
    "珠宝": "珠宝首饰",
    "箱包": "鞋靴箱包",
    "背包": "鞋靴箱包",
    "乐器": "乐器音响",
}

# 价格表达
# 品牌别名 → 商品名里真实出现的中文品牌词。
# 商品库的品牌一律是中文（苹果/华为/小米/三星…），而用户和大模型都可能写
# "HUAWEI"/"iPhone" —— 这类词在商品名里必然 0 命中，等于白填一个关键词，
# 所以统一在这里归一到中文，规则路径与大模型路径共用同一张表。
BRAND_ALIASES: dict[str, str] = {
    # ---- 3C / 手机 ----
    "huawei": "华为",
    "honor": "荣耀",
    "apple": "苹果",
    "iphone": "苹果",
    "ipad": "苹果",
    "macbook": "苹果",
    "airpods": "苹果",
    "xiaomi": "小米",
    "redmi": "小米",
    "mi": "小米",
    "samsung": "三星",
    "galaxy": "三星",
    "sony": "索尼",
    "lenovo": "联想",
    "thinkpad": "联想",
    "oppo": "OPPO",
    "vivo": "vivo",
    "oneplus": "一加",
    "meizu": "魅族",
    "dji": "大疆",
    "canon": "佳能",
    "nikon": "尼康",
    "panasonic": "松下",
    "philips": "飞利浦",
    "dyson": "戴森",
    "nintendo": "任天堂",
    "hp": "惠普",
    "dell": "戴尔",
    "asus": "华硕",
    "acer": "宏碁",
    "logitech": "罗技",
    "anker": "安克",
    "baseus": "倍思",
    "sennheiser": "森海塞尔",
    "3m": "3M",
    # ---- 家用电器 / 生活 ----
    "midea": "美的",
    "haier": "海尔",
    "starbucks": "星巴克",
    # ---- 运动 / 服饰 / 箱包 ----
    "nike": "耐克",
    "adidas": "阿迪达斯",
    "lining": "李宁",
    "anta": "安踏",
    "decathlon": "迪卡侬",
    "kailas": "凯乐石",
    "warrior": "回力",
    "michelin": "米其林",
    # ---- 珠宝 / 美妆 / 母婴 ----
    "swarovski": "施华洛世奇",
    "pandora": "潘多拉",
    "apm": "APM",
    "chowtaifook": "周大福",
    "chowsangsang": "周生生",
    "kao": "花王",
    "pigeon": "贝亲",
}


def normalize_keyword(word: str) -> str:
    """把关键词里的品牌英文写法归一到商品名里真正出现的中文词。"""
    text = (word or "").strip()
    if not text:
        return ""
    return BRAND_ALIASES.get(text.lower(), text)


# 价格表达
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|~|—|–|到|至)\s*(\d+(?:\.\d+)?)")
_MAX_RE = re.compile(
    r"(?:不超过|不高于|(?<!不)低于|小于|少于|最多|至多|预算)\s*(\d+(?:\.\d+)?)\s*(?:元|块|块钱|rmb|RMB|￥)?"
    r"|(\d+(?:\.\d+)?)\s*(?:元|块|块钱|rmb|RMB|￥)?\s*(?:以内|以下|之内|以里|内|左右)"
)
_MIN_RE = re.compile(
    r"(?:不低于|(?<!不)高于|(?<!不)大于|(?<!不)超过|至少|最少)\s*(\d+(?:\.\d+)?)\s*(?:元|块|块钱|rmb|￥)?"
    r"|(\d+(?:\.\d+)?)\s*(?:元|块|块钱)?\s*(?:以上|起步)"
)

# 解析后剩下的连接词/语气词，不作为关键词
_STOPWORDS = {
    "的", "了", "我要", "我想", "想", "要", "查", "找", "看看", "看", "搜", "搜索",
    "推荐", "给我", "帮我", "有没有", "有什么", "什么", "一些", "一点", "个", "件",
    "商品", "东西", "产品", "货", "款", "类型", "类", "类别", "种类", "左右",
    "元", "块", "块钱", "人民币", "rmb", "以内", "以下", "以上", "之内", "之内",
    "不", "和", "与", "或", "还有", "然后", "另外", "比较", "稍微", "大概", "约",
    "价格", "价位", "预算", "钱",
    "元的", "块的", "块钱的", "元的商品", "元以内", "块以内",
    # 请求/语气词：单独成词时也不该当关键词（"我要买"这类碎片见下面的前缀清洗）
    "我", "买", "请", "帮", "帮忙", "一下", "一个", "来", "麻烦", "介绍", "需要",
    "想要", "这个", "那个", "这些", "那些", "吗", "呢", "吧", "啊", "呀", "哦", "嘛",
}


# 「请求/语气前缀」：出现在碎片开头时先剥掉，再判断剩下的是不是真关键词。
# 例："我要买8000块钱以内的华为手机" 去掉价格与品类后剩 "我要买"、"的华为"，
# 前者整条都是请求词、后者带着从品类词上剥下来的"的"，两者都要清掉。
_PREFIX_FILLERS: tuple[str, ...] = tuple(
    sorted(
        {
            "我要买", "我要", "我想买", "我想", "我想要", "想要", "想买", "请帮我",
            "帮我", "给我", "麻烦", "介绍", "推荐", "有没有", "有没", "来一个", "来点",
            "来个", "找找", "看看", "搜索", "请", "找", "查", "搜", "买", "要", "想",
            "看", "我",
        },
        key=len,
        reverse=True,
    )
)
_LEADING_FILLERS = "的了"
_TRAILING_FILLERS = "的了款类种呢吧啊呀哦"


def _clean_token(token: str) -> str:
    """把规则切分出来的碎片洗成"可能出现在商品名里"的词。

    反复剥掉开头的请求/语气词（"我要买"→""、"我买华为"→"华为"）以及前后的
    "的/了/款/类/种"，剩下为空的直接丢弃。
    """
    word = (token or "").strip()
    if not word:
        return ""
    changed = True
    while changed and word:
        changed = False
        for prefix in _PREFIX_FILLERS:
            if word.startswith(prefix):
                word = word[len(prefix):]
                changed = True
                break
        stripped = word.lstrip(_LEADING_FILLERS)
        if stripped != word:
            word = stripped
            changed = True
    return word.strip().rstrip(_TRAILING_FILLERS).strip()


@dataclass
class QueryConstraints:
    """一次查询解析出的结构化约束。"""

    raw: str = ""
    categories: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    keywords: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "rules"  # rules | llm | hybrid —— 这次结果是哪条路径产出的
    llm_notes: list[str] = field(default_factory=list)

    @property
    def has_filter(self) -> bool:
        """是否解析出了任何有效约束。"""
        return bool(self.categories or self.keywords) or self.min_price is not None or self.max_price is not None

    def price_text(self) -> str:
        """价格约束的可读描述。"""
        if self.min_price is not None and self.max_price is not None:
            return f"{self.min_price:g} ~ {self.max_price:g}"
        if self.max_price is not None:
            return f"≤ {self.max_price:g}"
        if self.min_price is not None:
            return f"≥ {self.min_price:g}"
        return "不限"


def _spans_of(pattern: re.Pattern[str], text: str) -> list[tuple[int, int, re.Match[str]]]:
    return [(m.start(), m.end(), m) for m in pattern.finditer(text)]


def parse_price(text: str) -> tuple[float | None, float | None, list[tuple[int, int]], list[str]]:
    """解析价格约束，返回 (最小值, 最大值, 命中区间, 说明)。"""
    notes: list[str] = []
    spans: list[tuple[int, int]] = []
    low: float | None = None
    high: float | None = None

    # 区间优先："20到80元"
    match = _RANGE_RE.search(text)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        if low > high:
            low, high = high, low
        spans.append((match.start(), match.end()))
        notes.append(f"识别到价格区间 {low:g} ~ {high:g}")
        return low, high, spans, notes

    match = _MAX_RE.search(text)
    if match:
        value = match.group(1) or match.group(2)
        high = float(value)
        spans.append((match.start(), match.end()))
        notes.append(f"识别到价格上限 {high:g}")

    match = _MIN_RE.search(text)
    if match:
        value = match.group(1) or match.group(2)
        low = float(value)
        spans.append((match.start(), match.end()))
        notes.append(f"识别到价格下限 {low:g}")

    return low, high, spans, notes


def parse_categories(
    text: str,
    categories: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """解析品类约束；优先匹配更长的词（"厨房用品" 优先于 "厨房"）。"""
    aliases = aliases if aliases is not None else CATEGORY_ALIASES
    known = {str(c) for c in categories if str(c) and str(c) != "nan"}

    # 候选词 = 真实类目名 + 别名，按长度降序，保证长词优先
    candidates: list[tuple[str, str]] = [(c, c) for c in known]
    candidates += [(k, v) for k, v in aliases.items() if v in known or not known]
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)

    found: list[str] = []
    spans: list[tuple[int, int]] = []
    notes: list[str] = []
    lowered = text.lower()

    for word, canonical in candidates:
        if not word:
            continue
        idx = lowered.find(word.lower())
        if idx < 0:
            continue
        # 跳过已被更长词覆盖的位置
        if any(not (idx + len(word) <= s or idx >= e) for s, e in spans):
            continue
        spans.append((idx, idx + len(word)))
        if canonical not in found:
            found.append(canonical)
            notes.append(f"识别到品类「{canonical}」（来自「{word}」）")
        if len(found) >= 3:  # 一次查询最多三个品类，避免过于宽泛
            break

    return found, spans, notes


def parse_query_rules(
    text: str,
    categories: Iterable[str] = (),
    aliases: dict[str, str] | None = None,
    vocabulary: str | None = None,
) -> QueryConstraints:
    """**规则路径**：用正则与别名表把一句中文查询解析成结构化约束。

    规则解析擅长"精确、可枚举"的表达（价格数字、已知类目名），
    零延迟、零成本、结果可解释，所以永远先跑它。

    ``vocabulary`` 可选：传入"全库商品名 + 商品编码"的小写拼接串时，会把在当前
    商品库里 0 命中的关键词直接丢掉 —— 这种词留着也只会被过滤阶段忽略，
    不如一开始就别出现在回显里（"我要买"这类碎片就是这么冒出来的）。
    """
    raw = (text or "").strip()
    result = QueryConstraints(raw=raw)
    if not raw:
        result.notes.append("查询为空，将返回热门商品")
        return result

    low, high, price_spans, price_notes = parse_price(raw)
    result.min_price, result.max_price = low, high
    result.notes += price_notes

    cats, cat_spans, cat_notes = parse_categories(raw, categories, aliases)
    result.categories = cats
    result.notes += cat_notes

    # 去掉已识别的片段，剩下的作为关键词
    cut = [False] * len(raw)
    for start, end in price_spans + cat_spans:
        for i in range(max(0, start), min(len(raw), end)):
            cut[i] = True
    rest = "".join(ch if not cut[i] else " " for i, ch in enumerate(raw))

    tokens = [t for t in re.split(r"[\s,，、;；。.!？?/|+]+", rest) if t]
    keywords: list[str] = []
    for token in tokens:
        # 剥掉请求/语气前缀与"的/了/款/类/种"前后缀：
        # "的华为" → "华为"、"我要买" → ""、"红色的袋子" → "红色"
        cleaned = _clean_token(token)
        if not cleaned or cleaned.lower() in _STOPWORDS:
            continue
        cleaned = normalize_keyword(cleaned)  # HUAWEI → 华为
        if len(cleaned) == 1 and not cleaned.isascii():
            continue  # 单个汉字多半是语气词
        if cleaned.lower() in _STOPWORDS:
            continue
        # 被类目名完全包含的词没有信息量（"家居装饰"已命中类目时，再来个"家居"只会收窄候选）
        if any(cleaned in cat for cat in result.categories):
            continue
        if vocabulary is not None and cleaned.lower() not in vocabulary:
            continue  # 全库根本没有这个词，留着只会被静默忽略
        if cleaned in keywords:
            continue
        keywords.append(cleaned)

    result.keywords = keywords[:5]
    if keywords:
        result.notes.append("识别到关键词：" + "、".join(result.keywords))

    if not result.has_filter:
        result.notes.append("未识别出明确条件，将按热度与你的偏好推荐")
    return result


# ------------------------------------------------------------ 规则 + 大模型
def needs_llm(constraints: QueryConstraints, text: str) -> bool:
    """判断规则结果是否"不够用"，需要请大模型补一手。

    只在两种情况下调用大模型，避免无谓的延迟与费用：
        1. 规则什么约束都没解析出来（说明是"便宜的好看的礼物"这类模糊表达）；
        2. 规则只解析出关键词，没有品类也没有价格（关键词很可能匹配不到商品名）。
    """
    raw = (text or "").strip()
    if len(raw) < 4:
        return False
    if not constraints.has_filter:
        return True
    return not constraints.categories and constraints.min_price is None and constraints.max_price is None


def merge_constraints(rules: QueryConstraints, llm: QueryConstraints) -> QueryConstraints:
    """合并两条路径的结果。

    取值优先级：
        价格 —— **整对以规则为准**（正则对数字最准）；只有规则一点价格线索都没有时
                 才用大模型的价格；
        品类 —— 两边取并集（大模型能补上别名表没覆盖的说法）；
        关键词 —— 两边取并集，最多 5 个。

    价格必须"整对"处理，不能逐字段回退。踩过的坑：用户说"8000块钱以内的华为手机"，
    规则给出 `max_price=8000`、`min_price=None`，而大模型把"以内"读成了
    `min_price=8000, max_price=8000`；逐字段合并会让大模型的假下限漏进来，
    于是条件变成"≥8000 且 ≤8000"，候选直接清零、页面回退规则。
    """
    merged = QueryConstraints(raw=rules.raw)
    rules_has_price = rules.min_price is not None or rules.max_price is not None
    if rules_has_price:
        merged.min_price, merged.max_price = rules.min_price, rules.max_price
    else:
        merged.min_price, merged.max_price = llm.min_price, llm.max_price
    merged.categories = list(dict.fromkeys(rules.categories + llm.categories))[:3]
    merged.keywords = list(dict.fromkeys(rules.keywords + llm.keywords))[:5]
    merged.source = "hybrid"
    merged.notes = list(rules.notes)

    added: list[str] = []
    if not rules_has_price:  # 规则读不出价格时，才会真的采纳大模型的价格
        if llm.min_price is not None:
            added.append(f"价格下限 {llm.min_price:g}")
        if llm.max_price is not None:
            added.append(f"价格上限 {llm.max_price:g}")
    if llm.categories:
        new_cats = [c for c in llm.categories if c not in rules.categories]
        if new_cats:
            added.append("品类 " + "、".join(new_cats))
    if added:
        merged.llm_notes.append("大模型补充：" + "；".join(added))
    return merged


def parse_query(
    text: str,
    categories: Iterable[str] = (),
    aliases: dict[str, str] | None = None,
    llm=None,
    vocabulary: str | None = None,
) -> QueryConstraints:
    """**混合路径**：先规则解析；规则不够用时再请大模型补充，最后合并。

    ``llm`` 需要提供 ``available`` 属性与 ``parse(text, categories)`` 方法
    （见 :mod:`src.llm_parser`）。传 ``None`` 或未配置时，等价于纯规则解析 ——
    也就是说**没有大模型系统照样能跑**，只是覆盖不了模糊表达。

    ``vocabulary`` 见 :func:`parse_query_rules`：用于丢掉全库 0 命中的关键词。
    """
    rules = parse_query_rules(text, categories, aliases, vocabulary)

    if llm is None or not getattr(llm, "available", False):
        return rules
    if not needs_llm(rules, text):
        rules.llm_notes.append("规则已解析出明确条件，未调用大模型")
        return rules

    try:
        llm_result = llm.parse(text, list(categories))
    except Exception as exc:  # noqa: BLE001 - 大模型不可用时必须静默降级
        rules.llm_notes.append(f"大模型解析失败（{type(exc).__name__}），已回退到规则结果")
        return rules

    if llm_result is None:
        rules.llm_notes.append("大模型未返回有效结果，已回退到规则结果")
        return rules
    return merge_constraints(rules, llm_result)


def describe(constraints: QueryConstraints) -> str:
    """把约束拼成一句可读的回显文字（给界面用）。"""
    parts: list[str] = []
    if constraints.categories:
        parts.append("品类：" + "、".join(constraints.categories))
    if constraints.min_price is not None or constraints.max_price is not None:
        parts.append("价格：" + constraints.price_text())
    if constraints.keywords:
        parts.append("关键词：" + "、".join(constraints.keywords))
    return "；".join(parts) if parts else "无明确条件"


def match_keywords(name: str, item_id: str, keywords: Sequence[str]) -> bool:
    """关键词是否命中商品名称或编码（不区分大小写，任一命中即可）。"""
    if not keywords:
        return True
    haystack = f"{name} {item_id}".lower()
    return any(k.lower() in haystack for k in keywords)
