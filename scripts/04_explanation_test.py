"""阶段 4：EvidencePack、事实一致性、低置信度过滤和无证据降级。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src.evidence import build_evidence_pack, explain_item, normalize_visual_tags  # noqa: E402
from src.protocol import RecommendationItem, new_trace_id, utc_now  # noqa: E402
from web.app import SEARCHER  # noqa: E402


def _write(name: str, text: str) -> Path:
    outputs = Path(PATHS["outputs_dir"])
    root_path = outputs / name
    stage_path = outputs / "stage04" / name
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    root_path.write_text(text + "\n", encoding="utf-8")
    stage_path.write_text(text + "\n", encoding="utf-8")
    return root_path


def main() -> int:
    if SEARCHER is None:
        print("[FAIL] 推荐器未加载")
        return 1
    trace_id = new_trace_id()
    search = SEARCHER.search(
        "50元以内 收纳用品",
        username=None,
        top_n=2,
        feedback_map={},
    )
    if not search.get("results"):
        print("[FAIL] 无法取得阶段4测试商品")
        return 1
    row = search["results"][0]
    constraints = {
        "categories": ["收纳用品"],
        "min_price": None,
        "max_price": 50.0,
        "keywords": [],
        "colors": [],
    }
    normal = build_evidence_pack(
        row,
        constraints=constraints,
        recall_sources=["user_vec", "popular"],
        visual_tags=[
            {"value": "蓝色", "confidence": 0.92},
            {"value": "人物", "confidence": 0.32},
        ],
    )
    normal_reason, normal_detail = explain_item(normal)
    normal_item = RecommendationItem(
        item_id=row["item_id"],
        name=row["name"],
        category=row["category"],
        price=float(row["price"]),
        score=float(row["score"]),
        reason=normal_reason,
        evidence=tuple(normal.evidence),
        recall_sources=tuple(normal.recall_sources),
        trace_id=trace_id,
    )

    insufficient = build_evidence_pack(
        {},
        constraints={},
        recall_sources=[],
    )
    insufficient_reason, _ = explain_item(insufficient)
    partial = build_evidence_pack(
        {"item_id": "ONLY-ID"},
        constraints={},
        recall_sources=["content"],
    )
    partial_reason, _ = explain_item(partial)
    low_confidence_tags = normalize_visual_tags(
        [{"value": "极低置信度物体", "confidence": 0.2}]
    )
    high_confidence_tags = [tag["value"] for tag in normal.visual_tags]

    cases = [
        {
            "id": "normal",
            "trace_id": trace_id,
            "status": "ok",
            "item": normal_item.to_dict(),
            "has_verifiable_evidence": normal.has_verifiable_evidence(),
        },
        {
            "id": "evidence-insufficient",
            "trace_id": trace_id,
            "status": "insufficient",
            "reason": insufficient_reason,
            "has_verifiable_evidence": insufficient.has_verifiable_evidence(),
        },
        {
            "id": "partial-product-fields",
            "trace_id": trace_id,
            "status": "degraded",
            "reason": partial_reason,
            "has_verifiable_evidence": partial.has_verifiable_evidence(),
        },
        {
            "id": "low-confidence-visual-tag",
            "trace_id": trace_id,
            "status": "ignored",
            "kept_tags": low_confidence_tags,
            "high_confidence_tags": high_confidence_tags,
        },
    ]
    checks = {
        "normal_evidence_present": normal.has_verifiable_evidence(),
        "normal_reason_present": bool(normal_reason),
        "price_consistent": normal.price == float(row["price"]),
        "category_consistent": normal.category == row["category"],
        "high_confidence_visual_kept": "蓝色" in high_confidence_tags,
        "low_confidence_visual_ignored": "人物" not in high_confidence_tags
        and not low_confidence_tags,
        "no_fabrication_without_evidence": "信息不足" in insufficient_reason,
        "partial_fields_degrade": "信息不足" not in partial_reason
        and partial.has_verifiable_evidence(),
    }
    passed = all(checks.values())
    jsonl = "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n"
    wrote = [_write("stage04_explanation_cases.jsonl", jsonl)]

    evidence_rows: list[dict[str, Any]] = []
    for index, item in enumerate(normal.evidence, 1):
        evidence_rows.append(
            {
                "rank": index,
                "item_id": normal.item_id,
                "kind": item.get("kind", ""),
                "field": item.get("field", ""),
                "value": item.get("value", ""),
                "source": item.get("source", ""),
                "confidence": item.get("confidence", ""),
                "text": item.get("text", ""),
            }
        )
    outputs = Path(PATHS["outputs_dir"])
    stage_dir = outputs / "stage04"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for path in (outputs / "stage04_evidence_table.csv", stage_dir / "stage04_evidence_table.csv"):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0]))
            writer.writeheader()
            writer.writerows(evidence_rows)
    wrote.append(outputs / "stage04_evidence_table.csv")

    error_text = (
        "# 阶段4 异常解释记录\n\n"
        f"- 无证据输入：{insufficient_reason}\n"
        f"- 缺失名称、类目和价格：{partial_reason}\n"
        "- 低置信度视觉标签（0.2）已丢弃，不参与解释。\n"
        f"- 校验结果：{passed}\n"
        f"- 检查项：{json.dumps(checks, ensure_ascii=False)}\n"
    )
    wrote.append(_write("stage04_error_cases.md", error_text))
    print(f"[{'PASS' if passed else 'FAIL'}] {sum(checks.values())}/{len(checks)}")
    print("证据：" + "、".join(str(path.relative_to(ROOT)) for path in wrote))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
