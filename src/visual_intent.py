"""视觉线索到推荐查询的规范化与合成。

视觉模型负责描述图片，本模块负责把输出收敛到真实商品类目。视觉选品只按
图片中的物品与类目推荐，不保留或推断价格；最终查询仍交给现有
``SmartSearch``，不改变排序逻辑。
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.query_parser import normalize_keyword, parse_price

_GENERIC_OBJECTS = {
    "人", "人物", "男生", "女生", "男人", "女人", "房间", "室内", "背景",
    "环境", "场景", "物品", "东西", "商品",
}

# 模型偶尔会忽略提示词，把价格写进物体名或搜索词。除了复用规则解析器识别
# "100元以内" / "20到80元" 这类约束，下面再兜底清理带货币单位的金额和
# "预算/价格 + 数字" 的表达，保证视觉条件只保留可搜索的物品线索。
_PRICE_AMOUNT_RE = re.compile(
    r"(?:不超过|不高于|低于|小于|少于|最多|至多|预算|不低于|高于|大于|超过|至少|最少)?"
    r"\s*(?:[￥¥]\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*(?:元|块钱?|rmb))"
    r"\s*(?:以内|以下|以上|之内|以里|以内|内|左右|起步)?",
    re.IGNORECASE,
)
_BUDGET_RE = re.compile(
    r"(?:预算|价格|价位|售价)\s*(?:是|为|在|约|大概)?\s*"
    r"(?:\d+(?:\.\d+)?\s*(?:元|块钱?|rmb)?|不限|随意|都可以|无限制)",
    re.IGNORECASE,
)
_PRICE_EDGE_PUNCT = " \t\r\n,，、;；。.!！?？/|+-"


def strip_price_constraints(text: Any) -> str:
    """移除文本中的价格条件，只留下物品、类目或普通关键词。"""
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""

    spans: list[tuple[int, int]] = []
    spans.extend(parse_price(raw)[2])
    spans.extend((match.start(), match.end()) for match in _PRICE_AMOUNT_RE.finditer(raw))
    spans.extend((match.start(), match.end()) for match in _BUDGET_RE.finditer(raw))
    if not spans:
        return raw

    cut = [False] * len(raw)
    for start, end in spans:
        for index in range(max(0, start), min(len(raw), end)):
            cut[index] = True
    cleaned = "".join(char if not cut[index] else " " for index, char in enumerate(raw))
    parts = [
        part.strip(_PRICE_EDGE_PUNCT + "的了")
        for part in cleaned.split()
    ]
    return " ".join(part for part in parts if part)


def _dedupe(values: Iterable[Any], limit: int, *, max_chars: int = 40) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = strip_price_constraints(
            normalize_keyword(" ".join(str(value or "").split()))
        )[:max_chars]
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _match_categories(values: Iterable[Any], allowed: Iterable[str]) -> list[str]:
    categories = [str(c).strip() for c in allowed if str(c).strip()]
    matched: list[str] = []
    for raw in _dedupe(values, limit=8):
        exact = next(
            (category for category in categories if category.casefold() == raw.casefold()),
            None,
        )
        if exact:
            matched.append(exact)
            continue
        contains = [
            category for category in categories
            if raw.casefold() in category.casefold()
            or category.casefold() in raw.casefold()
        ]
        if len(contains) == 1:
            matched.append(contains[0])
    return _dedupe(matched, 3)


def normalize_analysis(
    analysis: dict[str, Any] | None,
    allowed_categories: Iterable[str],
) -> dict[str, Any]:
    """收敛桥接返回结果，避免页面展示或搜索使用无效字段。"""
    data = analysis if isinstance(analysis, dict) else {}
    objects = _dedupe(data.get("objects") or [], limit=5)
    objects = [item for item in objects if item not in _GENERIC_OBJECTS]
    colors = _dedupe(data.get("colors") or [], limit=5)
    categories = _match_categories(
        data.get("category_hints") or [], allowed_categories)
    search_query = strip_price_constraints(data.get("search_query"))[:120]
    return {
        "scene": strip_price_constraints(data.get("scene"))[:160],
        "objects": objects,
        "colors": colors,
        "category_hints": categories,
        "search_query": search_query,
    }


def compose_search_query(query: str, analysis: dict[str, Any] | None) -> str:
    """合入主类目与有效物体名，且永远不加入价格条件。

    颜色只作为界面证据展示，不默认写入查询：现有关键词匹配采用“任一命中”，
    把“白色”这类高频词带进查询会挤掉真正的物体线索。没有物体时再退回颜色。
    ``query`` 中的非价格内容可以作为补充，但视觉选品接口不传搜索框里的旧条件，
    避免上一次文字搜索残留的金额混入图片识别结果。最终长度限制为 240 字。
    """
    base = strip_price_constraints(query)
    data = normalize_analysis(analysis, (analysis or {}).get("category_hints", []))
    categories = list(data.get("category_hints") or [])[:1]
    objects = list(data.get("objects") or [])[:3]
    visual_parts = categories + objects
    if not objects:
        visual_parts += list(data.get("colors") or [])[:2]
    visual_text = " ".join(_dedupe(visual_parts, limit=10))
    if not visual_text and data.get("search_query"):
        visual_text = data["search_query"]

    combined = " ".join(part for part in (base, visual_text) if part)
    return combined[:240].strip()


def has_visual_terms(analysis: dict[str, Any] | None) -> bool:
    """视觉结果中是否至少存在一个可参与搜索的有效线索。"""
    data = analysis or {}
    return bool(
        data.get("category_hints")
        or data.get("objects")
        or data.get("colors")
        or data.get("search_query")
    )
