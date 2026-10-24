"""阶段 1：统一事件、推荐请求/结果与 HTTP 错误契约验收。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src.protocol import (  # noqa: E402
    ContractError,
    InteractionEvent,
    RecommendationItem,
    RecommendationRequest,
    new_trace_id,
    utc_now,
)
from web.app import app  # noqa: E402


def _event(**overrides: Any) -> dict[str, Any]:
    data = {
        "trace_id": new_trace_id(),
        "user_id": "contract_user",
        "session_id": "contract_session",
        "timestamp": utc_now(),
        "source": "text",
        "payload": {"text": "50元以内 收纳用品"},
    }
    data.update(overrides)
    return data


def _write(name: str, data: dict[str, Any]) -> Path:
    root_path = Path(PATHS["outputs_dir"]) / name
    stage_path = Path(PATHS["outputs_dir"]) / "stage01" / name
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    root_path.write_text(text + "\n", encoding="utf-8")
    stage_path.write_text(text + "\n", encoding="utf-8")
    return root_path


def main() -> int:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    event_data = _event()
    event = InteractionEvent.from_dict(event_data)
    record(
        "valid_event_roundtrip",
        event.to_dict() == event_data,
        "合法事件可序列化并完整恢复",
    )

    for field in ("trace_id", "user_id", "session_id", "timestamp", "source"):
        invalid = dict(event_data)
        invalid.pop(field)
        try:
            InteractionEvent.from_dict(invalid)
            passed, detail = False, "缺少字段却通过校验"
        except ContractError as exc:
            passed, detail = exc.code == "MISSING_FIELD", f"{exc.code}: {exc}"
        record(f"missing_{field}", passed, detail)

    for name, overrides, expected in (
        ("invalid_source", {"source": "camera"}, "INVALID_SOURCE"),
        ("invalid_timestamp", {"timestamp": "not-a-time"}, "INVALID_TIMESTAMP"),
        ("invalid_payload", {"payload": []}, "INVALID_PAYLOAD"),
    ):
        try:
            InteractionEvent.from_dict(_event(**overrides))
            passed, detail = False, "非法输入却通过校验"
        except ContractError as exc:
            passed, detail = exc.code == expected, f"{exc.code}: {exc}"
        record(name, passed, detail)

    request = RecommendationRequest.from_event(event, text="50元以内 收纳用品", top_k=5)
    record(
        "recommendation_request_trace",
        request.trace_id == event.trace_id and request.user_id == event.user_id,
        "推荐请求继承事件 trace 与用户",
    )
    try:
        RecommendationRequest.from_event(event, top_k=0)
        record("invalid_top_k", False, "top_k=0 却通过校验")
    except ContractError as exc:
        record("invalid_top_k", exc.code == "INVALID_TOP_K", exc.message)

    try:
        RecommendationItem(
            item_id="X1",
            name="测试商品",
            category="测试",
            price=1.0,
            score=0.1,
            reason="无证据",
            evidence=(),
            recall_sources=("popular",),
            trace_id=event.trace_id,
        )
        record("missing_evidence_rejected", False, "无证据商品却通过校验")
    except ContractError as exc:
        record(
            "missing_evidence_rejected",
            exc.code == "MISSING_EVIDENCE",
            exc.message,
        )

    client = app.test_client()
    missing_auth = client.get("/api/search?q=手机")
    auth_body = missing_auth.get_json(silent=True) or {}
    record(
        "http_auth_error_shape",
        missing_auth.status_code == 401
        and auth_body.get("error_code") == "AUTH_REQUIRED"
        and bool(auth_body.get("trace_id")),
        "未登录请求返回 AUTH_REQUIRED 与 trace_id",
    )

    with client.session_transaction() as flask_session:
        flask_session["username"] = "qzj"

    missing_query = client.get(
        "/api/search", headers={"X-Trace-ID": "trace-contract-0001"}
    )
    missing_body = missing_query.get_json(silent=True) or {}
    record(
        "http_missing_query",
        missing_query.status_code == 400
        and missing_body.get("error_code") == "MISSING_QUERY"
        and missing_body.get("trace_id") == "trace-contract-0001"
        and missing_query.headers.get("X-Trace-ID") == "trace-contract-0001",
        "缺参错误含稳定错误码，请求 trace 原样返回",
    )

    ok_trace = "trace-contract-0002"
    response = client.get(
        "/api/search?q=50%E5%85%83%E4%BB%A5%E5%86%85%20%E6%94%B6%E7%BA%B3%E7%94%A8%E5%93%81&top_n=2",
        headers={"X-Trace-ID": ok_trace},
    )
    body = response.get_json(silent=True) or {}
    rows = body.get("results") or []
    record(
        "http_success_trace_and_fields",
        response.status_code == 200
        and body.get("trace_id") == ok_trace
        and all(
            {"recall_sources", "evidence", "trace_id"} <= set(row.keys())
            for row in rows
        ),
        f"HTTP 200，返回 {len(rows)} 条带来源与证据的推荐",
    )

    passed = sum(1 for check in checks if check["passed"])
    report = {
        "stage": "01",
        "tested_at": utc_now(),
        "status": "passed" if passed == len(checks) else "failed",
        "summary": {"total": len(checks), "passed": passed, "failed": len(checks) - passed},
        "checks": checks,
        "artifacts": {
            "core": ["src/protocol.py", "src/orchestrator.py", "web/app.py"],
            "architecture": "docs/architecture.md",
        },
    }
    path = _write("stage01_contract_report.json", report)
    print(f"[{'PASS' if report['status'] == 'passed' else 'FAIL'}] {passed}/{len(checks)}")
    print(f"证据：{path.relative_to(ROOT)}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
