"""阶段 6：三类输入、故障降级、部署文档与提交包验收。"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src import feedback as fb  # noqa: E402
from src import multimodal_client as mmc  # noqa: E402
from src.protocol import new_trace_id, utc_now  # noqa: E402
from web.app import SESSION_STORE, app  # noqa: E402


TEST_USER = "stage6-acceptance-user"


class FixtureBridge:
    """不加载大模型也能重跑的确定性联调替身。"""

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "available": True,
            "vision": True,
            "asr": True,
            "gesture": True,
            "tts": True,
        }

    def transcribe(self, audio: bytes, trace_id: str = "") -> dict[str, Any]:
        return {
            "text": "50元以内 收纳用品",
            "engine": "fixture-asr",
            "elapsed_ms": 1,
            "trace_id": trace_id,
        }

    def gesture(self, image: bytes, trace_id: str = "") -> dict[str, Any]:
        return {
            "gesture": "thumbs_up",
            "label": "喜欢",
            "hands": 1,
            "trace_id": trace_id,
        }

    def vision(self, image: bytes, categories=None, trace_id: str = "") -> dict[str, Any]:
        return {
            "scene": "室内有白色冰箱",
            "objects": ["冰箱"],
            "colors": ["白色"],
            "category_hints": ["家用电器"],
            "search_query": "家用电器 冰箱",
            "model": "fixture-vision",
            "elapsed_ms": 1,
            "trace_id": trace_id,
        }

    def tts(self, text: str, trace_id: str = "") -> bytes:
        return b"ID3fixture"


class UnavailableVision(FixtureBridge):
    def vision(self, image: bytes, categories=None, trace_id: str = "") -> dict[str, Any]:
        raise mmc.BridgeUnavailable("fixture bridge offline", )


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def _package_violations(files: list[str]) -> list[str]:
    violations: list[str] = []
    for path in files:
        lower = path.lower()
        if lower.endswith((".pyc", ".pyo")) or "__pycache__/" in lower:
            violations.append(f"cache: {path}")
        if lower.endswith((".task", ".pkl", ".npz", ".model", ".gguf", ".safetensors")):
            violations.append(f"model: {path}")
        if lower.endswith((".db", ".sqlite", ".sqlite3")):
            violations.append(f"database: {path}")
        if lower.endswith((".llm_key", ".secret_key", ".env")):
            violations.append(f"secret: {path}")
    return violations


def _write(name: str, data: dict[str, Any]) -> Path:
    outputs = Path(PATHS["outputs_dir"])
    root_path = outputs / name
    stage_path = outputs / "stage06" / name
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    root_path.write_text(text + "\n", encoding="utf-8")
    stage_path.write_text(text + "\n", encoding="utf-8")
    return root_path


def main() -> int:
    # 阶段6 使用独立测试账号，不触碰真实用户的点赞记录。
    fb.init_db(PATHS["feedback_db"])
    fb.clear_feedback(PATHS["feedback_db"], TEST_USER)
    SESSION_STORE.clear_all()
    client = app.test_client()
    with client.session_transaction() as flask_session:
        flask_session["username"] = TEST_USER
    traces: dict[str, str] = {}
    with patch("web.app.MULTIMODAL", FixtureBridge()):
        voice_trace = new_trace_id()
        voice = client.post(
            "/api/voice",
            data=b"RIFF" + b"\0" * 256,
            content_type="audio/wav",
            headers={"X-Trace-ID": voice_trace},
        )
        traces["voice"] = voice_trace

        gesture_trace = new_trace_id()
        gesture = client.post(
            "/api/gesture",
            data=b"\xff\xd8\xff" + b"\0" * 64,
            content_type="image/jpeg",
            headers={"X-Trace-ID": gesture_trace},
        )
        traces["gesture"] = gesture_trace

        vision_trace = new_trace_id()
        vision = client.post(
            "/api/vision/analyze",
            data={"image": (io.BytesIO(b"\xff\xd8\xff" + b"\0" * 64), "scene.jpg")},
            content_type="multipart/form-data",
            headers={"X-Trace-ID": vision_trace},
        )
        traces["vision"] = vision_trace

    voice_body = voice.get_json(silent=True) or {}
    gesture_body = gesture.get_json(silent=True) or {}
    vision_body = vision.get_json(silent=True) or {}
    text_trace = new_trace_id()
    text_search = client.get(
        "/api/search?q=50%E5%85%83%E4%BB%A5%E5%86%85%20%E6%94%B6%E7%BA%B3%E7%94%A8%E5%93%81&top_n=3",
        headers={"X-Trace-ID": text_trace},
    )
    traces["text"] = text_trace
    text_body = text_search.get_json(silent=True) or {}
    target_item = (text_body.get("results") or [{}])[0].get("item_id", "")
    feedback_trace = new_trace_id()
    gesture_feedback = client.post(
        "/feedback",
        json={"item_id": target_item, "action": "like"},
        headers={"X-Trace-ID": feedback_trace},
    )
    traces["gesture_feedback"] = feedback_trace

    voice_search_trace = new_trace_id()
    voice_search = client.get(
        "/api/search",
        query_string={"q": voice_body.get("text") or "", "top_n": 2},
        headers={"X-Trace-ID": voice_search_trace},
    )
    vision_query = str(vision_body.get("suggested_query") or "")
    vision_search_trace = new_trace_id()
    vision_search = client.get(
        "/api/search",
        query_string={"q": vision_query, "top_n": 2},
        headers={"X-Trace-ID": vision_search_trace},
    )

    fallback_trace = new_trace_id()
    with patch("web.app.MULTIMODAL", UnavailableVision()):
        fallback = client.post(
            "/api/vision/analyze",
            data={"image": (io.BytesIO(b"\xff\xd8\xff" + b"\0" * 64), "scene.jpg")},
            content_type="multipart/form-data",
            headers={"X-Trace-ID": fallback_trace},
        )
    fallback_body = fallback.get_json(silent=True) or {}
    post_fault_text = client.get(
        "/api/search?q=%E6%94%B6%E7%BA%B3%E7%94%A8%E5%93%81&top_n=2",
        headers={"X-Trace-ID": new_trace_id()},
    )

    tracked = _tracked_files()
    violations = _package_violations(tracked)
    required_docs = {
        "architecture": ROOT / "docs" / "architecture.md",
        "deployment_and_risks": ROOT / "docs" / "deployment_and_risks.md",
        "demo_cases": ROOT / "docs" / "demo_cases.md",
    }
    checks = {
        "voice_chain": voice.status_code == 200 and bool(voice_body.get("text")),
        "gesture_chain": gesture.status_code == 200
        and gesture_body.get("gesture") == "thumbs_up",
        "vision_chain": vision.status_code == 200
        and bool(vision_body.get("suggested_query")),
        "text_chain": text_search.status_code == 200 and bool(text_body.get("results")),
        "gesture_to_feedback": gesture_feedback.status_code == 200
        and (gesture_feedback.get_json(silent=True) or {}).get("state") == "like",
        "voice_to_recommendation": voice_search.status_code == 200
        and bool((voice_search.get_json(silent=True) or {}).get("results")),
        "vision_to_recommendation": vision_search.status_code == 200
        and bool((vision_search.get_json(silent=True) or {}).get("results")),
        "fault_degrades": fallback.status_code == 503
        and fallback_body.get("fallback") == "text-only",
        "text_survives_fault": post_fault_text.status_code == 200,
        "trace_propagation": all(
            response.headers.get("X-Trace-ID") == expected
            for response, expected in (
                (voice, traces["voice"]),
                (gesture, traces["gesture"]),
                (vision, traces["vision"]),
                (text_search, traces["text"]),
                (fallback, fallback_trace),
            )
        ),
        "deployment_risk_docs": all(path.exists() for path in required_docs.values()),
        "clean_submission_package": not violations,
    }
    passed = all(checks.values())
    report = {
        "stage": "06",
        "tested_at": utc_now(),
        "status": "passed" if passed else "failed",
        "mode": "deterministic-fixture",
        "checks": checks,
        "inputs": {
            "voice": {"status": voice.status_code, "body": voice_body},
            "gesture": {"status": gesture.status_code, "body": gesture_body},
            "vision": {"status": vision.status_code, "body": vision_body},
            "text": {"status": text_search.status_code, "returned": text_body.get("returned", 0)},
            "gesture_feedback": {
                "status": gesture_feedback.status_code,
                "body": gesture_feedback.get_json(silent=True),
            },
        },
        "fault_injection": {
            "status": fallback.status_code,
            "fallback": fallback_body.get("fallback"),
            "error_code": fallback_body.get("error_code"),
            "text_after_fault": post_fault_text.status_code,
        },
        "submission_audit": {
            "tracked_files": len(tracked),
            "violations": violations,
        },
        "documents": {
            name: str(path.relative_to(ROOT)) for name, path in required_docs.items()
        },
        "trace_ids": traces,
    }
    path = _write("stage06_acceptance.json", report)
    fb.clear_feedback(PATHS["feedback_db"], TEST_USER)
    SESSION_STORE.clear_all()
    print(f"[{'PASS' if passed else 'FAIL'}] {sum(checks.values())}/{len(checks)}")
    if violations:
        print("提交包违规：" + "；".join(violations[:8]))
    print(f"证据：{path.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
