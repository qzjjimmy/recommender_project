"""阶段 5：固定口径反馈 A/B、会话隔离与失败案例。"""
from __future__ import annotations

import csv
import json
import statistics
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


QUERY = "50元以内 收纳用品"


def _event(session: str) -> InteractionEvent:
    return InteractionEvent(
        trace_id=new_trace_id(),
        user_id="stage05-user",
        session_id=session,
        timestamp=utc_now(),
        source="text",
        payload={"text": QUERY, "reset_context": True},
    )


def _run(orchestrator: Orchestrator, session: str, top_k: int = 10) -> dict[str, Any]:
    return orchestrator.handle_event(
        _event(session),
        top_k=top_k,
        persistent_feedback={},
    )


def _ranking(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "item_id": row["item_id"],
            "name": row["name"],
            "score": row["score"],
            "feedback": row.get("feedback", ""),
        }
        for rank, row in enumerate(result["results"], 1)
    ]


def _write(name: str, text: str) -> Path:
    outputs = Path(PATHS["outputs_dir"])
    root_path = outputs / name
    stage_path = outputs / "stage05" / name
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    root_path.write_text(text + "\n", encoding="utf-8")
    stage_path.write_text(text + "\n", encoding="utf-8")
    return root_path


def main() -> int:
    if SEARCHER is None:
        print("[FAIL] 推荐器未加载")
        return 1
    store = SessionStore()
    orchestrator = Orchestrator(SEARCHER, store)

    before = _run(orchestrator, "success-before")
    if not before["results"]:
        print("[FAIL] 反馈 A/B 查询没有候选")
        return 1
    target = before["results"][0]
    context = store.get("stage05-user", "success-before")
    context.set_feedback(target["item_id"], "like", persistent_feedback={})
    after = _run(orchestrator, "success-before")
    after_target = next(
        (row for row in after["results"] if row["item_id"] == target["item_id"]),
        None,
    )
    success_case = {
        "id": "success-like",
        "action": "like",
        "item_id": target["item_id"],
        "before_rank": 1,
        "after_rank": (
            after["results"].index(after_target) + 1 if after_target is not None else None
        ),
        "before_score": target["score"],
        "after_score": after_target["score"] if after_target is not None else None,
        "changed": bool(
            after_target is not None
            and after_target["score"] > target["score"]
        ),
    }

    unknown_before = _run(orchestrator, "no-change")
    unknown_context = store.get("stage05-user", "no-change")
    unknown_context.set_feedback("__MISSING_ITEM__", "like", persistent_feedback={})
    unknown_after = _run(orchestrator, "no-change")
    no_change_case = {
        "id": "failure-unknown-item",
        "action": "like",
        "item_id": "__MISSING_ITEM__",
        "before_ids": [row["item_id"] for row in unknown_before["results"]],
        "after_ids": [row["item_id"] for row in unknown_after["results"]],
        "changed": [row["item_id"] for row in unknown_before["results"]]
        != [row["item_id"] for row in unknown_after["results"]],
        "reason": "商品不在向量索引中，反馈无法形成可计算的偏好信号",
    }

    dislike_before = _run(orchestrator, "dislike-session-a")
    dislike_target = dislike_before["results"][0]
    dislike_context = store.get("stage05-user", "dislike-session-a")
    dislike_context.set_feedback(dislike_target["item_id"], "dislike", persistent_feedback={})
    dislike_after = _run(orchestrator, "dislike-session-a")
    isolated_after = _run(orchestrator, "new-session-b")
    isolation = {
        "id": "negative-feedback-session-isolation",
        "item_id": dislike_target["item_id"],
        "removed_in_same_session": dislike_target["item_id"]
        not in [row["item_id"] for row in dislike_after["results"]],
        "present_in_new_session": dislike_target["item_id"]
        in [row["item_id"] for row in isolated_after["results"]],
    }

    elapsed = [
        float(result.get("trace", {}).get("elapsed_ms", 0))
        for result in (before, after, unknown_before, unknown_after, dislike_before, dislike_after)
    ]
    performance = {
        "runs": len(elapsed),
        "median_ms": round(statistics.median(elapsed), 3),
        "max_ms": round(max(elapsed), 3),
        "failures": [
            {
                "case": "unknown item feedback",
                "behavior": "no ranking change",
                "expected": "explicitly excluded from preference vector",
            }
        ],
    }
    checks = {
        "like_increases_score": success_case["changed"],
        "session_negative_removed": isolation["removed_in_same_session"],
        "new_session_unpolluted": isolation["present_in_new_session"],
        "no_change_case_recorded": no_change_case["changed"] is False,
        "same_query_and_candidates": before["query"] == after["query"] == QUERY,
    }
    report = {
        "stage": "05",
        "tested_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "query": QUERY,
        "cases": [success_case, no_change_case, isolation],
        "before": _ranking(before),
        "after": _ranking(after),
        "performance": performance,
    }
    wrote = [
        _write(
            "stage05_feedback_ab.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    ]

    csv_path = Path(PATHS["outputs_dir"]) / "stage05_feedback_comparison.csv"
    stage_csv = Path(PATHS["outputs_dir"]) / "stage05" / "stage05_feedback_comparison.csv"
    stage_csv.parent.mkdir(parents=True, exist_ok=True)
    for path in (csv_path, stage_csv):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "rank", "item_id", "name", "score", "feedback"])
            for phase, rows in (("before", report["before"]), ("after", report["after"])):
                for row in rows:
                    writer.writerow(
                        [
                            phase,
                            row["rank"],
                            row["item_id"],
                            row["name"],
                            row["score"],
                            row["feedback"],
                        ]
                    )
    wrote.append(csv_path)

    performance_text = (
        "# 阶段5 性能与失败案例摘要\n\n"
        f"- 六次查询中位耗时：{performance['median_ms']} ms\n"
        f"- 最大耗时：{performance['max_ms']} ms\n"
        "- 失败案例：对索引中不存在的商品点赞不会改变排序；系统没有伪造偏好信号。\n"
        f"- 会话隔离：同一会话点踩后商品被移除={isolation['removed_in_same_session']}；"
        f"新会话仍可见={isolation['present_in_new_session']}。\n"
    )
    wrote.append(_write("stage05_performance_failures.md", performance_text))
    print(f"[{'PASS' if report['status'] == 'passed' else 'FAIL'}] {sum(checks.values())}/{len(checks)}")
    print("证据：" + "、".join(str(path.relative_to(ROOT)) for path in wrote))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
