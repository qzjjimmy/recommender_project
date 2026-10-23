"""桥接服务自检：验证项目二 → 项目一 的本地 HTTP 接口是否真的可用。

用法（在项目二根目录、用项目二的 conda 环境）::

    python scripts/10_test_bridge.py                 # 全部检查
    python scripts/10_test_bridge.py --skip-asr      # 跳过较慢的语音识别
    python scripts/10_test_bridge.py --skip tts,llm,vision  # 跳过指定项

脚本会**在内存里另起一个桥接服务**（随机端口，避免和正在跑的 :8100 冲突），
然后依次请求 /health、/asr、/gesture、/vision、/tts、/llm。

为什么不用外部依赖：本项目要求"不新增第三方依赖"，所以这里用 urllib 发请求。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bridge_server import BridgeHandler  # noqa: E402

OUTPUTS = PROJECT_ROOT / "outputs"
TEST_AUDIO = OUTPUTS / "test_audio.wav"
TEST_FRAME = OUTPUTS / "test_frame.jpg"
# 手势检查优先用带手的真实抓拍（``snapshots/`` 是阶段05 的相机快照），
# 没有时退回 test_frame.jpg —— 那张图里通常没有手，判空属于正常。
TEST_HAND_FRAME = OUTPUTS / "snapshots" / "snapshot_000.jpg"
TEST_VISION_IMAGE = PROJECT_ROOT.parent / "photos" / "家用电器.jpg"


def _post(base: str, path: str, body: bytes, content_type: str,
          timeout: float) -> tuple[int, bytes, str]:
    """发一个 POST，返回 ``(status, body, content_type)``。"""
    request = urllib.request.Request(
        base + path, data=body,
        headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def _get(base: str, path: str, timeout: float) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return exc.code, {}


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append((name, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}：{detail}")

    def summary(self) -> int:
        passed = sum(1 for _, ok, _ in self.rows if ok)
        total = len(self.rows)
        print("\n================ 桥接自检汇总 ================")
        for name, ok, detail in self.rows:
            print(f"  {'✓' if ok else '✗'} {name:<10} {detail}")
        print(f"  通过 {passed}/{total}")
        return 0 if passed == total else 1


def run_checks(base: str, result: Result, skip: set[str]) -> None:
    """依次执行各项检查。"""
    # ---------------------------------------------------------- /health
    started = time.time()
    status, health = _get(base, "/health", timeout=10)
    cost = int((time.time() - started) * 1000)
    ok = status == 200 and health.get("status") == "ok"
    detail = (f"{cost}ms | ollama={health.get('ollama', {}).get('available')} "
              f"model={health.get('ollama', {}).get('text_model')} | "
              f"mediapipe={health.get('mediapipe', {}).get('available')} | "
              f"vision={health.get('vision', {}).get('available')}")
    result.add("health", ok, detail)

    # ------------------------------------------------------------- /asr
    if "asr" in skip:
        print("[SKIP] asr")
    elif not TEST_AUDIO.exists():
        result.add("asr", False, f"缺少测试音频 {TEST_AUDIO}")
    else:
        started = time.time()
        status, body, _ = _post(base, "/asr", TEST_AUDIO.read_bytes(),
                                "audio/wav", timeout=180)
        cost = int((time.time() - started) * 1000)
        data = json.loads(body.decode("utf-8")) if status == 200 else {}
        text = data.get("text", "")
        result.add("asr", status == 200 and "error" not in data,
                   f"{cost}ms | 识别结果：{text!r}")

    # --------------------------------------------------------- /gesture
    if "gesture" in skip:
        print("[SKIP] gesture")
    elif not (TEST_HAND_FRAME.exists() or TEST_FRAME.exists()):
        result.add("gesture", False, f"缺少测试图片 {TEST_HAND_FRAME} / {TEST_FRAME}")
    else:
        frame_path = TEST_HAND_FRAME if TEST_HAND_FRAME.exists() else TEST_FRAME
        started = time.time()
        status, body, _ = _post(base, "/gesture", frame_path.read_bytes(),
                                "image/jpeg", timeout=60)
        cost = int((time.time() - started) * 1000)
        data = json.loads(body.decode("utf-8")) if status == 200 else {}
        result.add("gesture", status == 200 and "gesture" in data,
                   f"{cost}ms | {frame_path.name} → 单帧判定={data.get('gesture')} "
                   f"检测到手={data.get('hands')}")

    # ------------------------------------------------------------ /vision
    if "vision" in skip:
        print("[SKIP] vision")
    elif not TEST_VISION_IMAGE.exists():
        result.add("vision", False, f"缺少测试图片 {TEST_VISION_IMAGE}")
    else:
        categories = [
            "家用电器", "家具", "家居装饰", "宿舍用品", "智能设备",
            "手机", "笔记本电脑", "平板电脑", "耳机音响", "数码配件",
        ]
        payload = json.dumps(
            {
                "image_b64": base64.b64encode(
                    TEST_VISION_IMAGE.read_bytes()).decode("ascii"),
                "categories": categories,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        started = time.time()
        status, body, _ = _post(
            base, "/vision", payload, "application/json", timeout=180)
        cost = int((time.time() - started) * 1000)
        data = json.loads(body.decode("utf-8")) if status == 200 else {}
        required = {"scene", "objects", "colors", "category_hints", "search_query"}
        ok = (
            status == 200
            and required.issubset(data)
            and isinstance(data.get("category_hints"), list)
            and any(item in categories for item in data.get("category_hints", []))
        )
        result.add(
            "vision",
            ok,
            f"{cost}ms | 类目={data.get('category_hints')} "
            f"查询={data.get('search_query')!r}",
        )

    # ------------------------------------------------------------- /tts
    if "tts" in skip:
        print("[SKIP] tts")
    else:
        payload = json.dumps(
            {"text": "你好，这是项目二桥接服务的语音合成自检。"},
            ensure_ascii=False).encode("utf-8")
        started = time.time()
        status, body, content_type = _post(base, "/tts", payload,
                                           "application/json", timeout=60)
        cost = int((time.time() - started) * 1000)
        ok = status == 200 and len(body) > 1000 and "audio" in content_type
        if ok:
            out = OUTPUTS / "bridge_selftest.mp3"
            out.write_bytes(body)
            detail = f"{cost}ms | {len(body)} 字节，已保存 {out.name}"
        else:
            detail = f"{cost}ms | 返回 {status} {body[:120]!r}"
        result.add("tts", ok, detail)

    # -------------------------------------------------------------- /llm
    if "llm" in skip:
        print("[SKIP] llm")
    else:
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "你是简洁的助手，只回答一个词。"},
                {"role": "user", "content": "中国的首都是哪里？"},
            ],
            "max_tokens": 64,
        }, ensure_ascii=False).encode("utf-8")
        started = time.time()
        status, body, _ = _post(base, "/llm", payload, "application/json",
                                timeout=180)
        cost = int((time.time() - started) * 1000)
        data = json.loads(body.decode("utf-8")) if status == 200 else {}
        content = data.get("content", "")
        result.add("llm", status == 200 and bool(content.strip()),
                   f"{cost}ms | model={data.get('model')} | 回答：{content.strip()[:40]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="桥接服务自检")
    parser.add_argument("--port", type=int, default=0,
                        help="自检用的端口（0 = 随机空闲端口，默认）")
    parser.add_argument("--skip", default="",
                        help="跳过某些项，逗号分隔，如 asr,tts")
    parser.add_argument("--skip-asr", action="store_true", help="等价于 --skip asr")
    args = parser.parse_args(argv)

    skip = {item.strip().lower() for item in args.skip.split(",") if item.strip()}
    if args.skip_asr:
        skip.add("asr")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), BridgeHandler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    print(f"桥接自检服务已启动：{base}（跳过：{sorted(skip) or '无'}）\n")

    result = Result()
    try:
        run_checks(base, result, skip)
    finally:
        server.shutdown()
        server.server_close()
    return result.summary()


if __name__ == "__main__":
    raise SystemExit(main())
