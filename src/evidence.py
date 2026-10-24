"""商品证据包与可核验推荐解释。

解释器只读取商品画像、硬约束命中、召回来源和达到置信度阈值的视觉标签。
没有足够事实时明确返回“信息不足”，不生成看似具体但无法核验的推荐理由。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


VISUAL_CONFIDENCE_THRESHOLD = 0.6
SOURCE_LABELS = {
    "popular": "热门召回",
    "similar": "相似扩展召回",
    "user_vec": "用户向量召回",
    "user_vector": "用户向量召回",
    "feedback": "会话反馈",
    "content": "商品画像",
    "fallback_popular": "热门兜底",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_visual_tags(tags: Iterable[Any] | None) -> list[dict[str, Any]]:
    """过滤低置信度视觉标签并保留可核验来源。"""
    out: list[dict[str, Any]] = []
    for raw in tags or []:
        if isinstance(raw, Mapping):
            value = _text(raw.get("value") or raw.get("name") or raw.get("label"))
            confidence = float(raw.get("confidence", 1.0) or 0.0)
        else:
            value = _text(raw)
            confidence = 1.0
        if value and confidence >= VISUAL_CONFIDENCE_THRESHOLD:
            out.append(
                {
                    "kind": "visual_tag",
                    "field": "vision",
                    "value": value,
                    "confidence": round(confidence, 4),
                    "source": "vision",
                }
            )
    return out


def _match_constraints(
    name: str,
    category: str,
    price: float,
    constraints: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    data = dict(constraints or {})
    evidence: list[dict[str, Any]] = []
    categories = [str(value) for value in (data.get("categories") or []) if str(value)]
    if category and category in categories:
        evidence.append(
            {
                "kind": "constraint_match",
                "field": "category",
                "value": category,
                "source": "product_profile",
                "text": f"商品类目「{category}」符合请求条件",
            }
        )
    max_price = data.get("max_price")
    min_price = data.get("min_price")
    if max_price is not None and price <= float(max_price):
        evidence.append(
            {
                "kind": "constraint_match",
                "field": "max_price",
                "value": round(price, 2),
                "limit": float(max_price),
                "source": "product_profile",
                "text": f"售价 {price:.2f} 元，不超过上限 {float(max_price):g} 元",
            }
        )
    if min_price is not None and price >= float(min_price):
        evidence.append(
            {
                "kind": "constraint_match",
                "field": "min_price",
                "value": round(price, 2),
                "limit": float(min_price),
                "source": "product_profile",
                "text": f"售价 {price:.2f} 元，不低于下限 {float(min_price):g} 元",
            }
        )
    keywords = [str(value) for value in (data.get("keywords") or []) if str(value)]
    for keyword in keywords:
        if keyword.casefold() in name.casefold():
            evidence.append(
                {
                    "kind": "constraint_match",
                    "field": "keyword",
                    "value": keyword,
                    "source": "product_profile",
                    "text": f"商品名包含请求关键词「{keyword}」",
                }
            )
    colors = [str(value) for value in (data.get("colors") or []) if str(value)]
    for color in colors:
        if color.casefold() in name.casefold():
            evidence.append(
                {
                    "kind": "constraint_match",
                    "field": "color",
                    "value": color,
                    "source": "product_profile",
                    "text": f"商品名包含请求颜色「{color}」",
                }
            )
    return evidence


@dataclass
class EvidencePack:
    """一次推荐结果的可核验事实集合。"""

    item_id: str
    name: str = ""
    category: str = ""
    price: float | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    recall_sources: list[str] = field(default_factory=list)
    visual_tags: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def has_verifiable_evidence(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "constraints": dict(self.constraints),
            "recall_sources": list(self.recall_sources),
            "visual_tags": list(self.visual_tags),
            "evidence": list(self.evidence),
            "has_verifiable_evidence": self.has_verifiable_evidence(),
        }


def build_evidence_pack(
    item: Mapping[str, Any],
    *,
    constraints: Mapping[str, Any] | None = None,
    recall_sources: Sequence[str] | None = None,
    visual_tags: Iterable[Any] | None = None,
) -> EvidencePack:
    """从统一候选字段构造 EvidencePack。"""
    item_id = _text(item.get("item_id") or item.get("id"))
    name = _text(item.get("name"))
    category = _text(item.get("category"))
    try:
        price = float(item.get("price"))
    except (TypeError, ValueError):
        price = -1.0

    sources = [str(source) for source in (recall_sources or []) if str(source)]
    visual = normalize_visual_tags(visual_tags)
    evidence: list[dict[str, Any]] = []
    if item_id:
        evidence.append(
            {
                "kind": "product_fact",
                "field": "item_id",
                "value": item_id,
                "source": "product_profile",
                "text": f"商品 ID 为 {item_id}",
            }
        )
    if name:
        evidence.append(
            {
                "kind": "product_fact",
                "field": "name",
                "value": name,
                "source": "product_profile",
                "text": f"商品画像名称为「{name}」",
            }
        )
    if category:
        evidence.append(
            {
                "kind": "product_fact",
                "field": "category",
                "value": category,
                "source": "product_profile",
                "text": f"商品画像类目为「{category}」",
            }
        )
    if price >= 0:
        evidence.append(
            {
                "kind": "product_fact",
                "field": "price",
                "value": round(price, 2),
                "source": "product_profile",
                "text": f"商品画像售价为 {price:.2f} 元",
            }
        )
    if price >= 0:
        evidence.extend(_match_constraints(name, category, price, constraints))
    for source in sources:
        evidence.append(
            {
                "kind": "retrieval",
                "field": "recall_source",
                "value": source,
                "source": "recommendation_pipeline",
                "text": f"候选由{SOURCE_LABELS.get(source, source)}产生",
            }
        )
    evidence.extend(visual)

    return EvidencePack(
        item_id=item_id,
        name=name,
        category=category,
        price=price if price >= 0 else None,
        constraints=dict(constraints or {}),
        recall_sources=sources,
        visual_tags=visual,
        evidence=evidence,
    )


def explain_item(pack: EvidencePack) -> tuple[str, str]:
    """生成简短、可核验的解释和证据摘要。"""
    if not pack.has_verifiable_evidence():
        return "信息不足，无法给出可靠推荐理由", ""

    parts: list[str] = []
    matched = [item for item in pack.evidence if item.get("kind") == "constraint_match"]
    for item in matched[:2]:
        text = _text(item.get("text"))
        if text and text not in parts:
            parts.append(text)

    source_texts: list[str] = []
    for source in pack.recall_sources:
        label = SOURCE_LABELS.get(source, source)
        if label not in source_texts:
            source_texts.append(label)
    if source_texts:
        parts.append("候选用自" + "、".join(source_texts))

    visual_values = [str(item.get("value")) for item in pack.visual_tags]
    if visual_values:
        parts.append("视觉线索为" + "、".join(visual_values[:2]))

    if not parts and pack.name:
        parts.append(f"商品「{pack.name}」来自可核验商品画像")
    reason = "；".join(parts[:3])
    details = " | ".join(
        _text(item.get("text"))
        for item in pack.evidence[:5]
        if _text(item.get("text"))
    )
    return reason or "信息不足，无法给出可靠推荐理由", details
