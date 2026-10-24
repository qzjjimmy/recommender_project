from __future__ import annotations

from src.evidence import build_evidence_pack, explain_item, normalize_visual_tags


def test_evidence_explanation_uses_product_and_constraint_facts() -> None:
    pack = build_evidence_pack(
        {
            "item_id": "X1",
            "name": "蓝色收纳盒",
            "category": "收纳用品",
            "price": 12.5,
        },
        constraints={"categories": ["收纳用品"], "max_price": 20},
        recall_sources=["user_vec"],
        visual_tags=[{"value": "蓝色", "confidence": 0.9}],
    )
    reason, detail = explain_item(pack)
    assert pack.has_verifiable_evidence()
    assert "收纳用品" in reason
    assert "售价" in detail


def test_low_confidence_visual_tag_is_removed() -> None:
    tags = normalize_visual_tags([{"value": "人物", "confidence": 0.2}])
    assert tags == []


def test_missing_evidence_does_not_fabricate_reason() -> None:
    pack = build_evidence_pack({}, recall_sources=[])
    reason, detail = explain_item(pack)
    assert "信息不足" in reason
    assert detail == ""
