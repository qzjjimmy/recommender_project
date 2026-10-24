"""阶段 3：编排器正常、空结果、降级和可复现性验收。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.protocol import InteractionEvent, new_trace_id, utc_now  # noqa: E402
from src.session_context import SessionStore  # noqa: E402
from web.app import SEARCHER  # noqa: E402


def _event(text: str, *, session: str) -> InteractionEvent:
    return InteractionEvent(
        trace_id=new_trace_id(),
        user_id="stage03-user",
        session_id=session,
        timestamp=utc_now(),
        source="text",
        payload={"text": text, "reset_context": True},
    )


def _write(name: str, data: dict[str, Any]) -> Path:
    outputs = Path(PATHS["outputs_dir"])
    root_path = outputs / name
    stage_path = outputs / "stage03" / name
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    root_path.write_text(text + "\n", encoding="utf-8")
    stage_path.write_text(text + "\n", encoding="utf-8")
    return root_path


def _constraint_ok(result: dict[str, Any], category: str, max_price: float) -> bool:
    constraints = result["context"]["constraints"]
    for row in result["results"]:
        if row["category"] != category or float(row["price"]) > max_price:
            return False
    return constraints["categories"] == [category] and constraints["max_price"] == max_price


def main() -> int:
    if SEARCHER is None:
        print("[FAIL] 推荐器未加载，请先准备模型与商品画像")
        return 1

    orchestrator = Orchestrator(SEARCHER, SessionStore())
    normal_event = _event("50元以内 收纳用品", session="normal")
    normal = orchestrator.handle_event(
        normal_event, top_k=8, persistent_feedback={}
    )
    empty_event = _event("100元以内 家用电器", session="empty")
    empty = orchestrator.handle_event(empty_event, top_k=8, persistent_feedback={})
    fallback_event = _event("50元以内 收纳用品", session="fallback")
    fallback = orchestrator.handle_event(
        fallback_event,
        top_k=5,
        persistent_feedback={},
        degraded_reason="multimodal_bridge_unavailable",
    )

    replay_orchestrator = Orchestrator(SEARCHER, SessionStore())
    replay = replay_orchestrator.handle_event(
        _event("50元以内 收纳用品", session="replay"),
        top_k=8,
        persistent_feedback={},
    )
    original_ids = [item["item_id"] for item in normal["results"]]
    replay_ids = [item["item_id"] for item in replay["results"]]

    checks = {
        "normal_has_results": normal["status"] == "ok" and bool(normal["results"]),
        "normal_has_sources": all(item["recall_sources"] for item in normal["results"]),
        "normal_has_evidence": all(item["evidence"] for item in normal["results"]),
        "hard_constraints_hold": _constraint_ok(normal, "收纳用品", 50.0),
        "same_input_reproducible": original_ids == replay_ids,
        "empty_result_clear": empty["status"] == "empty" and not empty["results"],
        "empty_result_has_hint": bool(empty.get("question")),
        "fallback_runs": fallback["status"] == "degraded" and bool(fallback["results"]),
        "trace_id_consistent": all(
            result["trace"]["trace_id"] == result["trace_id"]
            for result in (normal, empty, fallback)
        ),
    }
    passed = all(checks.values())

    demo = {
        "stage": "03",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "normal": normal,
        "empty": empty,
        "fallback": fallback,
        "reproducibility": {
            "first_ids": original_ids,
            "replay_ids": replay_ids,
            "identical": original_ids == replay_ids,
        },
    }
    wrote = [
        _write("stage03_demo.json", demo),
        _write("stage03_trace_normal.json", normal["trace"]),
        _write("stage03_trace_empty.json", empty["trace"]),
        _write("stage03_trace_fallback.json", fallback["trace"]),
    ]
    print(f"[{'PASS' if passed else 'FAIL'}] {sum(checks.values())}/{len(checks)}")
    print("证据：" + "、".join(str(path.relative_to(ROOT)) for path in wrote))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
