"""项目二多模态桥接服务的客户端。

项目一（本仓库）与项目二（``multimodal_ai_project1``）跑在不同的 conda 环境里，
两边都有一个顶层 ``config`` 包，直接互相 import 会撞名。所以项目二把能力
包成了一个本地 HTTP 服务（``bridge_server.py``，默认 127.0.0.1:8100），
本模块就是项目一这侧的薄客户端。

四条工程约束（与智能推荐页其它能力一致）
------------------------------------------
1. **可插拔**：``search.multimodal.enabled=False`` 或服务没启动时，所有方法都会
   明确失败（抛 :class:`BridgeUnavailable`），由上层降级到项目一原有实现；
2. **探测有缓存**：健康探测结果缓存 ``health_ttl`` 秒，避免逐帧请求都去探一次；
3. **超时分级**：健康探测必须快（默认 1.5s），ASR / 视觉 / 大模型可以慢；
4. **失败不致命**：任何网络异常都转成异常对象，绝不让页面 500。
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any

from config.settings import HYPERPARAMS


class BridgeUnavailable(RuntimeError):
    """桥接服务不可达 / 未启用（调用方应降级到内置实现）。"""


class BridgeError(RuntimeError):
    """桥接服务可达，但本次请求失败（返回了错误 JSON 或非预期状态码）。"""

    def __init__(self, status: int, message: str, trace_id: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.trace_id = trace_id


def default_config() -> dict[str, Any]:
    """从 ``config/settings.py`` 取默认配置（唯一配置来源）。"""
    return dict(HYPERPARAMS.get("search", {}).get("multimodal", {}) or {})


class MultimodalClient:
    """桥接服务的 HTTP 客户端（标准库 urllib，不依赖 requests）。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = default_config() if config is None else dict(config)
        self.enabled = bool(cfg.get("enabled", True))
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:8100")).rstrip("/")
        timeouts = dict(cfg.get("timeout", {}) or {})
        self.timeout_health = float(timeouts.get("health", 1.5))
        self.timeout_asr = float(timeouts.get("asr", 60))
        self.timeout_gesture = float(timeouts.get("gesture", 6))
        self.timeout_vision = float(timeouts.get("vision", 180))
        self.timeout_tts = float(timeouts.get("tts", 45))
        self.timeout_llm = float(timeouts.get("llm", 180))
        self.health_ttl = float(cfg.get("health_ttl", 5.0))
        self.gesture_fps = int(cfg.get("gesture_fps", 5))
        self.asr_sample_rate = int(cfg.get("asr_sample_rate", 16000))

        self._health_cache: tuple[float, dict[str, Any] | None] = (0.0, None)

    # ---------------------------------------------------------------- HTTP
    def _request(self, path: str, *, data: bytes | None = None,
                 content_type: str = "application/json",
                 timeout: float = 10.0, method: str | None = None,
                 trace_id: str | None = None):
        """发一个请求；网络层异常统一转成 :class:`BridgeUnavailable`。"""
        headers = {"Content-Type": content_type} if data is not None else {}
        if trace_id:
            headers["X-Trace-ID"] = str(trace_id)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method or ("POST" if data is not None else "GET"),
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # 服务可达但拒绝了这次请求：把桥接返回的 {"error": ...} 原样带出去
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:  # noqa: BLE001
                detail = ""
            raise BridgeError(
                exc.code, detail or f"桥接服务返回 {exc.code}", str(trace_id or "")
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BridgeUnavailable(
                f"无法连接多模态桥接服务 {self.base_url}（{exc}）"
            ) from exc

    def _request_json(self, path: str, payload: dict | None = None,
                      timeout: float = 10.0, method: str | None = None,
                      trace_id: str | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._request(
            path, data=body, timeout=timeout, method=method, trace_id=trace_id
        ) as response:
            raw = response.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError(502, f"桥接服务返回的不是 JSON：{raw[:120]!r}") from exc
        if not isinstance(data, dict):
            raise BridgeError(502, "桥接服务返回的 JSON 不是对象")
        return data

    # --------------------------------------------------------------- 状态
    def health(self, force: bool = False) -> dict[str, Any] | None:
        """探测桥接服务；不可用时返回 ``None``（结果按 ``health_ttl`` 缓存）。"""
        if not self.enabled:
            return None
        now = time.time()
        cached_at, cached = self._health_cache
        if cached is not None and not force and (now - cached_at) < self.health_ttl:
            return cached
        try:
            data = self._request_json("/health", timeout=self.timeout_health)
        except (BridgeUnavailable, BridgeError):
            data = None
        self._health_cache = (now, data)
        return data

    def available(self) -> bool:
        """桥接服务是否可用。"""
        return self.health() is not None

    def status(self) -> dict[str, Any]:
        """给页面/接口用的能力摘要（永不抛异常）。"""
        if not self.enabled:
            return {"enabled": False, "available": False,
                    "reason": "config/settings.py 里关闭了 search.multimodal.enabled"}
        health = self.health()
        if health is None:
            return {
                "enabled": True,
                "available": False,
                "base_url": self.base_url,
                "reason": (
                    f"多模态桥接服务未启动（{self.base_url}）。"
                    "启动方式：cd multimodal_ai_project1 && python bridge_server.py"
                ),
            }
        ollama = health.get("ollama", {}) or {}
        vision = health.get("vision", {}) or {}
        return {
            "enabled": True,
            "available": True,
            "base_url": self.base_url,
            "asr": bool((health.get("asr") or {}).get("available")),
            "gesture": bool((health.get("mediapipe") or {}).get("available")),
            "vision": bool(vision.get("available")),
            "vision_model": vision.get("model", ""),
            "tts": bool((health.get("tts") or {}).get("available")),
            "llm": bool(ollama.get("available")),
            "text_model": ollama.get("text_model", ""),
            "ollama_base_url": ollama.get("base_url", ""),
            "gesture_fps": self.gesture_fps,
            "asr_sample_rate": self.asr_sample_rate,
        }

    def describe(self) -> str:
        """一句话说明桥接服务当前状态（页面与日志用）。"""
        status = self.status()
        if status.get("available"):
            return (f"已连接：{status.get('base_url')}"
                    f"（本地模型 {status.get('text_model') or '未知'}）")
        return str(status.get("reason", "不可用"))

    # ----------------------------------------------------------- 各能力
    def transcribe(self, wav_bytes: bytes, trace_id: str | None = None) -> dict[str, Any]:
        """语音识别：WAV 字节 → ``{"text", "elapsed_ms", ...}``。

        语种由桥接服务侧的 ``ASR_CONFIG["language"]`` 决定（默认中文），
        前端不传语种参数，避免每条请求都要多带一份 JSON。
        """
        if not wav_bytes:
            raise BridgeError(400, "音频内容为空")
        with self._request("/asr", data=wav_bytes, content_type="audio/wav",
                           timeout=self.timeout_asr, trace_id=trace_id) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise BridgeError(502, "ASR 返回格式异常")
        return data

    def gesture(self, image_bytes: bytes, trace_id: str | None = None) -> dict[str, Any]:
        """单帧手势识别：JPEG 字节 → ``{"gesture", "label", "hands"}``。"""
        if not image_bytes:
            raise BridgeError(400, "图像内容为空")
        with self._request("/gesture", data=image_bytes, content_type="image/jpeg",
                           timeout=self.timeout_gesture, trace_id=trace_id) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise BridgeError(502, "手势接口返回格式异常")
        return data

    def vision(self, image_bytes: bytes,
               categories: list[str] | None = None,
               trace_id: str | None = None) -> dict[str, Any]:
        """图片推荐线索：JPEG/PNG 字节 → 结构化场景、物体与类目。"""
        if not image_bytes:
            raise BridgeError(400, "图片内容为空")
        payload = {
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "categories": [str(c) for c in (categories or [])],
        }
        data = self._request_json(
            "/vision", payload, timeout=self.timeout_vision, trace_id=trace_id)
        if not isinstance(data.get("category_hints", []), list):
            raise BridgeError(502, "视觉接口返回的 category_hints 格式异常")
        return data

    def tts(self, text: str, trace_id: str | None = None) -> bytes:
        """语音合成：文本 → MP3 字节。"""
        text = (text or "").strip()
        if not text:
            raise BridgeError(400, "text 不能为空")
        body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        with self._request(
            "/tts", data=body, timeout=self.timeout_tts, trace_id=trace_id
        ) as response:
            return response.read()

    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int | None = None,
             trace_id: str | None = None) -> dict[str, Any]:
        """本地大模型对话：``{"content", "model", "elapsed_ms"}``。"""
        if not messages:
            raise BridgeError(400, "messages 不能为空")
        payload: dict[str, Any] = {"messages": messages, "temperature": temperature}
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        return self._request_json(
            "/llm", payload, timeout=self.timeout_llm, trace_id=trace_id
        )


_DEFAULT_CLIENT: MultimodalClient | None = None


def get_client() -> MultimodalClient:
    """返回进程内共享的客户端（共享健康探测缓存）。"""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = MultimodalClient()
    return _DEFAULT_CLIENT
