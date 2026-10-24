"""阶段 2：六类意图、槽位证据、冲突、撤销和 20 条以上组合样例。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src.protocol import InteractionEvent, new_trace_id, utc_now  # noqa: E402
from src.session_context import SessionStore  # noqa: E402
from web.app import SEARCHER  # noqa: E402


def _event(
    *,
    source: str = "text",
    text: str = "",
    payload: dict[str, Any] | None = None,
    user: str = "stage02-user",
    session: str = "stage02-session",
) -> InteractionEvent:
    data = dict(payload or {})
    if text:
        data["text"] = text
    return InteractionEvent(
        trace_id=new_trace_id(),
        user_id=user,
        session_id=session,
        timestamp=utc_now(),
        source=source,
        payload=data,
    )


def _case(
    case_id: str,
    events: list[InteractionEvent],
    *,
    intent: str,
    status: str,
    categories: list[str] | None = None,
    missing: str | None = None,
    undo: bool = False,
) -> dict[str, Any]:
    store = SessionStore()
    context = store.get(events[0].user_id, events[0].session_id)
    for event in events:
        context.apply_event(event, allowed_categories=getattr(SEARCHER, "categories", ()))
    undone = context.undo() if undo else False
    data = context.to_dict()
    checks = [data["intent"] == intent, data["status"] == status]
    if categories is not None:
        checks.append(set(categories) <= set(data["slots"]["categories"]))
    if missing is not None:
        checks.append(missing in data["missing_slots"])
    if undo:
        checks.append(undone)
    return {
        "id": case_id,
        "sources": [event.source for event in events],
        "intent": data["intent"],
        "status": data["status"],
        "missing_slots": data["missing_slots"],
        "conflicts": data["conflicts"],
        "slots": data["slots"],
        "slot_evidence_summary": {
            key: len(value) for key, value in data["slot_evidence"].items()
        },
        "expected": {
            "intent": intent,
            "status": status,
            **({"missing": missing} if missing else {}),
        },
        "passed": all(checks),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    cases = [
        _case(
            "filter-budget-category",
            [_event(text="50元以内 收纳用品")],
            intent="filter",
            status="ready",
            categories=["收纳用品"],
        ),
        _case(
            "discover-category",
            [_event(text="找点派对用品")],
            intent="discover",
            status="ready",
            categories=["派对用品"],
        ),
        _case("discover-brand", [_event(text="手机")], intent="discover", status="ready", categories=["手机"]),
        _case(
            "vision-appliance",
            [
                _event(
                    source="vision",
                    payload={
                        "vision": {
                            "scene": "室内有冰箱",
                            "objects": ["冰箱", "洗衣机"],
                            "colors": ["白色"],
                            "category_hints": ["家用电器"],
                            "confidence": 0.92,
                        }
                    },
                )
            ],
            intent="discover",
            status="ready",
            categories=["家用电器"],
        ),
        _case(
            "voice-budget",
            [_event(source="voice", text="100元以内 家用电器")],
            intent="filter",
            status="ready",
            categories=["家用电器"],
        ),
        _case(
            "gesture-like",
            [
                _event(
                    source="gesture",
                    payload={"gesture": "thumbs_up", "target_item": "PH10001", "confidence": 0.98},
                )
            ],
            intent="feedback",
            status="ready",
        ),
        _case(
            "gesture-missing-target",
            [_event(source="gesture", payload={"gesture": "thumbs_down", "confidence": 0.9})],
            intent="feedback",
            status="needs_input",
            missing="target_item",
        ),
        _case(
            "text-feedback-missing-action",
            [_event(text="我喜欢这个")],
            intent="feedback",
            status="needs_input",
            missing="target_item",
        ),
        _case(
            "compare-two-items",
            [_event(text="对比 PH10001 和 PH10002")],
            intent="compare",
            status="ready",
        ),
        _case(
            "compare-missing-item",
            [_event(text="比较一下 PH10001")],
            intent="compare",
            status="needs_input",
            missing="target_items",
        ),
        _case(
            "similar-target",
            [_event(text="找类似 PH10001 的商品")],
            intent="similar",
            status="ready",
        ),
        _case(
            "similar-missing-target",
            [_event(text="找类似的商品")],
            intent="similar",
            status="needs_input",
            missing="target_item",
        ),
        _case(
            "explain-target",
            [_event(text="为什么推荐 PH10001")],
            intent="explain",
            status="ready",
        ),
        _case(
            "low-confidence-vision-ignored",
            [
                _event(
                    source="vision",
                    payload={
                        "vision": {
                            "objects": ["冰箱"],
                            "category_hints": ["家用电器"],
                            "confidence": 0.3,
                        }
                    },
                )
            ],
            intent="discover",
            status="needs_input",
            missing="query",
        ),
        _case(
            "multi-turn-budget",
            [_event(text="收纳用品"), _event(text="预算50元以内")],
            intent="filter",
            status="ready",
            categories=["收纳用品"],
        ),
        _case(
            "category-conflict",
            [_event(text="手机"), _event(text="家用电器")],
            intent="discover",
            status="needs_clarification",
        ),
        _case(
            "price-conflict",
            [_event(text="100元以上 收纳用品"), _event(text="50元以内")],
            intent="filter",
            status="needs_clarification",
        ),
        _case(
            "undo-keeps-first-constraint",
            [_event(text="手机"), _event(text="100元以内")],
            intent="discover",
            status="ready",
            categories=["手机"],
            undo=True,
        ),
        _case(
            "visual-color",
            [
                _event(
                    source="vision",
                    payload={
                        "vision": {
                            "objects": ["收纳盒"],
                            "colors": ["蓝色"],
                            "category_hints": ["收纳用品"],
                            "confidence": 0.88,
                        }
                    },
                )
            ],
            intent="discover",
            status="ready",
            categories=["收纳用品"],
        ),
        _case(
            "gesture-dislike",
            [
                _event(
                    source="gesture",
                    payload={"gesture": "thumbs_down", "target_item": "PH10001"},
                )
            ],
            intent="feedback",
            status="ready",
        ),
        _case(
            "filter-budget-only",
            [_event(text="不超过20元")],
            intent="filter",
            status="ready",
        ),
        _case(
            "explain-after-recommendation",
            [_event(text="派对用品"), _event(text="解释一下为什么 PH10001")],
            intent="explain",
            status="ready",
        ),
        _case(
            "compare-two-visual-objects",
            [
                _event(
                    source="vision",
                    payload={
                        "vision": {
                            "objects": ["冰箱", "洗衣机"],
                            "category_hints": ["家用电器"],
                            "confidence": 0.9,
                        },
                        "text": "对比一下",
                    },
                )
            ],
            intent="compare",
            status="needs_input",
            missing="target_items",
        ),
        _case(
            "voice-feedback-target",
            [
                _event(
                    source="voice",
                    text="不喜欢 PH10001",
                    payload={"target_item": "PH10001", "action": "dislike"},
                )
            ],
            intent="feedback",
            status="ready",
        ),
    ]

    # 额外验证正负反馈互斥与清理。
    context = SessionStore().get("feedback-user", "feedback-session")
    first = context.set_feedback("PH10001", "like")
    second = context.set_feedback("PH10001", "dislike")
    mutual = first == "like" and second == "dislike" and context.session_feedback["PH10001"] == "dislike"
    context.record_browse("PH10002")
    cleared = context.clear_feedback("dislike") == 1 and "PH10001" not in context.session_feedback
    cases.extend(
        [
            {
                "id": "feedback-mutual-exclusion",
                "intent": "feedback",
                "status": "ready" if mutual else "failed",
                "passed": mutual,
                "expected": {"likes_and_dislikes_exclusive": True},
            },
            {
                "id": "session-clear-feedback",
                "intent": "feedback",
                "status": "ready" if cleared else "failed",
                "passed": cleared,
                "expected": {"session_dislikes_cleared": True},
            },
        ]
    )

    total = len(cases)
    passed = sum(bool(case["passed"]) for case in cases)
    rate = passed / total
    errors = [case for case in cases if not case["passed"]]
    output_dir = Path(PATHS["outputs_dir"])
    rows_path = output_dir / "stage02_context_cases.jsonl"
    stage_rows = output_dir / "stage02" / "stage02_context_cases.jsonl"
    _write_jsonl(rows_path, cases)
    _write_jsonl(stage_rows, cases)
    metrics = {
        "stage": "02",
        "tested_at": utc_now(),
        "status": "passed" if rate >= 0.8 else "failed",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(rate, 4),
        "threshold": 0.8,
        "intent_coverage": sorted({case["intent"] for case in cases}),
        "artifacts": [
            "outputs/stage02_context_cases.jsonl",
            "outputs/stage02_context_metrics.json",
            "outputs/stage02_error_cases.md",
        ],
    }
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    (output_dir / "stage02_context_metrics.json").write_text(metrics_text + "\n", encoding="utf-8")
    stage_dir = output_dir / "stage02"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage02_context_metrics.json").write_text(metrics_text + "\n", encoding="utf-8")

    error_lines = ["# 阶段2 错误案例分析", ""]
    if errors:
        for case in errors:
            error_lines.extend(
                [
                    f"## {case['id']}",
                    "",
                    f"- 期望：`{json.dumps(case['expected'], ensure_ascii=False)}`",
                    f"- 实际：intent=`{case.get('intent')}` status=`{case.get('status')}`",
                    "- 处理：检查意图规则、槽位证据或冲突合并逻辑。",
                    "",
                ]
            )
    else:
        error_lines.append("本批样例全部通过；低置信度视觉标签与缺槽位分支已由专门样例覆盖。")
    error_text = "\n".join(error_lines) + "\n"
    (output_dir / "stage02_error_cases.md").write_text(error_text, encoding="utf-8")
    (stage_dir / "stage02_error_cases.md").write_text(error_text, encoding="utf-8")

    print(f"[{'PASS' if metrics['status'] == 'passed' else 'FAIL'}] {passed}/{total} ({rate:.1%})")
    print(f"证据：{rows_path.relative_to(ROOT)}")
    return 0 if metrics["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
