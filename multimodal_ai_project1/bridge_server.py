"""本地桥接服务：把项目二的多模态能力开放给项目一（智能推荐）使用。

为什么要有这个文件
------------------
项目一（``recommender_project``）与项目二（``multimodal_ai_project1``）跑在**两个不同的
conda 环境**里，而且两边都有一个顶层 ``config`` 包——直接互相 ``import`` 会撞名，
把 torch / funasr / mediapipe 装进项目一的环境也很重。所以这里把项目二的能力
包成一个**只依赖标准库**的本地 HTTP 服务：

    ┌─────────────────────┐        HTTP (127.0.0.1:8100)        ┌──────────────────────┐
    │ 项目一 Flask (5000) │ ──────────────────────────────────► │ 本项目 bridge_server │
    │  智能推荐页面        │  /asr /gesture /vision /tts /llm   │  ASR / MediaPipe /   │
    └─────────────────────┘                                     │  VL / Ollama / TTS   │
                                                                └──────────────────────┘

它**不是**新的第三方依赖：只用了 ``http.server``、``json``、``base64``、``tempfile``
等标准库，模型全部懒加载，缺失或推理失败都返回结构化错误而不是让进程崩掉。

接口契约（全部 JSON；错误统一 ``{"error": ...}``）
--------------------------------------------------
* ``GET  /health``   服务与各项能力的可用性摘要
* ``POST /asr``      ``audio/wav`` 原始字节（或 ``{"audio_b64":...}``）→ 转写文本
* ``POST /gesture``  ``image/jpeg`` 原始字节 → 单帧手势（**不做冷却抑制**）
* ``POST /vision``   ``{"image_b64":..., "categories":[...]}`` → 结构化视觉线索
* ``POST /tts``      ``{"text": ...}`` → ``audio/mpeg`` 字节流
* ``POST /llm``      ``{"messages":[...]}`` → Ollama 回答文本

启动（必须在项目二根目录、用项目二的 conda 环境）::

    conda activate multimodal_ai
    python bridge_server.py            # 默认 127.0.0.1:8100
    python bridge_server.py --port 8100 --verbose

自检::

    python scripts/10_test_bridge.py
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    ASR_CONFIG,
    MODEL_ASSETS,
    MODEL_PATHS,
    OLLAMA_CONFIG,
    PATHS,
    TTS_CONFIG,
    resolve_vl_model,
)

MAX_BODY_BYTES = 24 * 1024 * 1024  # 单次请求体上限（24MB：十几帧 JPEG 也够）
MAX_IMAGE_BYTES = 8 * 1024 * 1024
TEMP_DIR = Path(PATHS["temp"])
TEMP_DIR.mkdir(parents=True, exist_ok=True)
START_TIME = time.time()

# 模型推理使用的单把锁：MediaPipe / FunASR / torch 的推理都不是线程安全的，
# 而且同时跑两份会让内存峰值翻倍。HTTP 层是多线程的，但推理在这里串行化。
_INFER_LOCK = threading.Lock()

# 手势默认识别"只跑 Hands"。MediaPipe 的 Pose / Hands / Face 三个模型一起跑时
# 单帧要 0.8~0.9 秒（实测），而手势判定只需要手部关键点；前端要按 5fps 轮询，
# 三个模型一起上根本喂不动。用 ``--full-detection`` 可以切回全量检测。
HANDS_ONLY = True

VISION_SYSTEM_PROMPT = """你是电商推荐系统的视觉线索提取器。
只描述图片中明确可见的内容，不识别或推断人物身份、年龄、性别、民族、健康与情绪。
不猜测品牌、价格、材质、尺寸等画面无法确认的属性。输出必须严格遵循用户给出的 JSON 结构。"""

VISION_PROMPT_TEMPLATE = """请分析这张图片，为电商商品推荐提供可核验的线索。
只输出一个 JSON 对象，不要 Markdown 代码块，不要解释。字段必须为：
{{
  "scene": "简短环境描述，最多 60 字",
  "objects": ["画面中可见、可购买或可联想到商品的具体物品名，最多 5 个，不要含价格"],
  "colors": ["物体或环境的主要颜色，最多 5 个"],
  "category_hints": ["从下方可选类目中选 1~3 个，必须原样使用类目名"],
  "search_query": "一句只含可见品类、物体和颜色的商品搜索词"
}}
可选商品类目：{categories}

约束：
1. 看不清的物体不要写；不要写人物身份或外貌评价。
2. category_hints 只能从可选类目中选择，没有合适类目时返回空数组。
3. objects 和 search_query 都只写物品、类目和颜色，不要包含价格、金额、币种、预算、品牌或主观评价。"""

_VISION_STOP_OBJECTS = {
    "人", "人物", "男生", "女生", "男人", "女人", "房间", "室内", "背景", "环境",
}


class _Models:
    """模型注册表：全部懒加载，加载失败会被记住并如实上报（不会反复重试刷屏）。"""

    def __init__(self) -> None:
        self.asr = None
        self.asr_error: str | None = None
        self.detector = None
        self.detector_error: str | None = None
        self.gesture = None
        self.backend = None
        self.tts = None
        self.tts_error: str | None = None
        self.vision = None
        self.vision_error: str | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- ASR
    def get_asr(self):
        """懒加载语音识别引擎（SenseVoiceSmall）。失败返回 ``None``。"""
        with self._lock:
            if self.asr is not None or self.asr_error:
                return self.asr
            try:
                from src.asr import ASREngine

                engine = ASREngine(model_path=str(MODEL_PATHS["asr"]))
                engine.load()
                self.asr = engine
                print(f"[桥接] ASR 已加载：{MODEL_PATHS['asr']}")
            except Exception as exc:  # noqa: BLE001 - 缺模型/缺依赖都要如实降级
                self.asr_error = f"{type(exc).__name__}: {exc}"
                print(f"[桥接] ASR 加载失败：{self.asr_error}")
            return self.asr

    # ------------------------------------------------------------ 手势
    def get_detector(self):
        """懒加载 MediaPipe 人体/手部检测器。失败返回 ``None``。"""
        with self._lock:
            if self.detector is not None or self.detector_error:
                return self.detector
            try:
                from src.gesture import GestureRecognizer
                from src.human_detector import HumanDetector

                self.detector = HumanDetector()
                # 冷却时间给一个极小值：``GestureRecognizer.__init__`` 里 ``0`` 会被
                # 当成"没传"而回落到配置值（2 秒），会让逐帧判定失效。
                # 冷却与稳定保持交给前端做（那里能按时间窗口统计，更准）。
                self.gesture = GestureRecognizer(cooldown_seconds=0.001)
                print("[桥接] MediaPipe 手势模型已加载")
            except Exception as exc:  # noqa: BLE001
                self.detector_error = f"{type(exc).__name__}: {exc}"
                print(f"[桥接] MediaPipe 加载失败：{self.detector_error}")
            return self.detector

    # -------------------------------------------------------------- LLM
    def get_backend(self):
        """懒加载 Ollama 客户端（不联网，构造很便宜）。"""
        with self._lock:
            if self.backend is None:
                from src.ollama_backend import OllamaBackend

                self.backend = OllamaBackend()
            return self.backend

    # -------------------------------------------------------------- TTS
    def get_tts(self):
        """懒加载语音合成引擎（edge-tts，联网）。"""
        with self._lock:
            if self.tts is not None or self.tts_error:
                return self.tts
            try:
                from src.tts import TTSEngine

                self.tts = TTSEngine(output_dir=str(TEMP_DIR))
            except Exception as exc:  # noqa: BLE001
                self.tts_error = f"{type(exc).__name__}: {exc}"
                print(f"[桥接] TTS 初始化失败：{self.tts_error}")
            return self.tts

    # ------------------------------------------------------------ 视觉
    def get_vision(self):
        """懒加载视觉语言模型适配器（不预热大模型权重）。"""
        with self._lock:
            if self.vision is not None or self.vision_error:
                return self.vision
            try:
                from src.vl_model import VLModel

                self.vision = VLModel()
                print(f"[桥接] 视觉模型已就绪：{self.vision.model_path}")
            except Exception as exc:  # noqa: BLE001
                self.vision_error = f"{type(exc).__name__}: {exc}"
                print(f"[桥接] 视觉模型加载失败：{self.vision_error}")
            return self.vision


MODELS = _Models()


# --------------------------------------------------------------------- 能力实现
def _clean_terms(value, limit: int) -> list[str]:
    """把模型返回的字符串或对象数组规范成去重后的字符串列表。"""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("object") or "").strip()
            color = str(item.get("color") or item.get("colour") or "").strip()
            text = " ".join(part for part in (name, color) if part)
        else:
            text = str(item or "").strip()
        text = " ".join(text.split())[:40]
        if not text or text in _VISION_STOP_OBJECTS:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _extract_json_object(text: str) -> dict | None:
    """从普通 JSON、Markdown 代码块或前后带解释的文本中提取第一个 JSON 对象。"""
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].lstrip()
            if part.startswith("{"):
                candidates.append(part)
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _dedupe_text(values: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _match_categories(values, categories: list[str]) -> list[str]:
    """将模型类目映射到项目一真实类目，优先精确匹配，再做唯一子串匹配。"""
    allowed = [str(c) for c in categories if str(c).strip()]
    raw = _clean_terms(values, limit=8)
    matched: list[str] = []
    for hint in raw:
        exact = next((c for c in allowed if c.casefold() == hint.casefold()), None)
        if exact:
            matched.append(exact)
            continue
        contains = [
            c for c in allowed
            if hint.casefold() in c.casefold() or c.casefold() in hint.casefold()
        ]
        if len(contains) == 1:
            matched.append(contains[0])
    return _dedupe_text(matched, 3)


def normalize_vision_result(raw_text: str, categories: list[str]) -> dict:
    """把视觉模型输出规范化为稳定的推荐线索结构。"""
    parsed = _extract_json_object(raw_text)
    if parsed:
        scene = str(parsed.get("scene") or "").strip()
        objects = _clean_terms(parsed.get("objects"), 5)
        colors = _clean_terms(parsed.get("colors"), 5)
        hints = _match_categories(parsed.get("category_hints"), categories)
        search_query = str(parsed.get("search_query") or "").strip()
    else:
        # 模型没有给出结构化结果时保留可核验的简短描述，不臆造列表字段。
        scene = " ".join((raw_text or "").strip().split())
        objects, colors, hints, search_query = [], [], [], ""

    scene = scene[:160]
    fallback_query = " ".join(
        _dedupe_text(hints + objects[:2] + colors[:2], limit=6)
    )
    search_query = " ".join(search_query.split())[:120] or fallback_query
    return {
        "scene": scene,
        "objects": objects,
        "colors": colors,
        "category_hints": hints,
        "search_query": search_query,
    }


def run_asr(wav_bytes: bytes, language: str = "auto") -> dict:
    """把 WAV 字节落临时文件后交给 SenseVoiceSmall 识别，返回 ``{"text", "elapsed_ms"}``。"""
    engine = MODELS.get_asr()
    if engine is None:
        raise RuntimeError(MODELS.asr_error or "ASR 引擎不可用")
    if not wav_bytes:
        raise ValueError("音频内容为空")

    suffix = ".wav"
    handle = tempfile.NamedTemporaryFile(prefix="bridge_asr_", suffix=suffix,
                                         dir=TEMP_DIR, delete=False)
    tmp_path = Path(handle.name)
    try:
        handle.write(wav_bytes)
        handle.close()
        started = time.time()
        # ASR_CONFIG["language"] 默认 "zh"：短句用 auto 容易在中英之间乱跳，
        # 但这里尊重调用方传来的值，前端不传时用配置。
        lang = language or ASR_CONFIG.get("language", "auto")
        with _INFER_LOCK:
            text = engine.transcribe(str(tmp_path), language=lang)
        return {"text": text or "", "elapsed_ms": int((time.time() - started) * 1000)}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_gesture(image_bytes: bytes) -> dict:
    """单帧手势识别，返回 ``{"gesture", "label", "hands"}``。

    只做**单帧判定**，不在这里做"稳定保持/冷却"——那是前端的职责（前端有 0.9s
    时间窗口的统计，比后端凭时间戳更贴近用户感受）。

    默认走"只用 Hands"的快速通道：复用 ``HumanDetector`` 已经建好的 HandLandmarker
    实例与 ``primary_hand`` 逻辑，跳过 Pose / Face 两个模型（手势用不到它们）。
    ``HANDS_ONLY=False`` 时回退到 ``process_frame`` 的全量检测。
    """
    if not image_bytes:
        raise ValueError("图像内容为空")
    detector = MODELS.get_detector()
    if detector is None:
        raise RuntimeError(MODELS.detector_error or "MediaPipe 不可用")

    import cv2
    import numpy as np

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码图像，请确认上传的是 JPEG/PNG 字节")

    with _INFER_LOCK:
        if HANDS_ONLY:
            import mediapipe as mp

            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hands_result = detector.hands.detect(mp_image)
            # 复用项目二写好的"主手"挑选逻辑（优先右手，否则第一只手）
            hand = detector.primary_hand({"hands": hands_result})
        else:
            results = detector.process_frame(frame)
            hands_result = results.get("hands")
            hand = detector.primary_hand(results)
        hands = len(getattr(hands_result, "hand_landmarks", None) or [])
        gesture = MODELS.gesture.recognize(hand) if hand is not None else None

    label = MODELS.gesture.label_of(gesture) if MODELS.gesture else None
    return {"gesture": gesture, "label": label, "hands": hands}


def run_vision(image_bytes: bytes, categories: list[str] | None = None) -> dict:
    """单张图片 → 可供推荐系统使用的结构化视觉线索。"""
    if not image_bytes:
        raise ValueError("图片内容为空")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"图片过大（上限 {MAX_IMAGE_BYTES // 1024 // 1024}MB）")

    model = MODELS.get_vision()
    if model is None:
        raise RuntimeError(MODELS.vision_error or "视觉模型不可用")

    allowed_categories = _dedupe_text(
        [
            str(c).strip() for c in (categories or [])
            if str(c).strip() and str(c).strip() not in ("其他", "未知类目")
        ],
        limit=60,
    )
    prompt = VISION_PROMPT_TEMPLATE.format(
        categories=json.dumps(allowed_categories, ensure_ascii=False))
    started = time.time()
    with _INFER_LOCK:
        raw = model.generate_response_from_image_bytes(
            image_bytes,
            prompt,
            system=VISION_SYSTEM_PROMPT,
            max_side=768,
        )
    result = normalize_vision_result(raw, allowed_categories)
    result.update(
        {
            "model": str(model.model_path),
            "elapsed_ms": int((time.time() - started) * 1000),
        }
    )
    return result


def run_tts(text: str, voice: str | None = None) -> bytes:
    """文本转语音，返回 MP3 字节。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("text 不能为空")
    if len(text) > 2000:
        # 一次读几十条商品名会很长，截断而不是拒绝，保证演示不中断。
        text = text[:2000]

    engine = MODELS.get_tts()
    if engine is None:
        raise RuntimeError(MODELS.tts_error or "TTS 引擎不可用")

    out_path = TEMP_DIR / f"bridge_tts_{int(time.time() * 1000)}.mp3"
    try:
        with _INFER_LOCK:
            engine.synthesize(text, output_path=str(out_path), voice=voice)
        data = out_path.read_bytes()
        if not data:
            raise RuntimeError("TTS 返回了空音频")
        return data
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_llm(messages: list[dict], model: str | None = None,
            temperature: float = 0.2, max_tokens: int | None = None) -> dict:
    """转发一次对话请求给 Ollama，返回 ``{"content", "model", "elapsed_ms"}``。"""
    if not messages:
        raise ValueError("messages 不能为空")

    from src.ollama_backend import resolve_text_model

    backend = MODELS.get_backend()
    chosen = model or resolve_text_model()
    started = time.time()
    with _INFER_LOCK:
        content = backend.chat(messages, model=chosen, temperature=temperature,
                               max_tokens=max_tokens)
    return {
        "content": content or "",
        "model": chosen,
        "elapsed_ms": int((time.time() - started) * 1000),
    }


# ------------------------------------------------------------------- HTTP 层
class BridgeHandler(BaseHTTPRequestHandler):
    """极简 HTTP 处理器：路由 + JSON 编解码 + 统一错误格式。"""

    server_version = "MultimodalBridge/1.0"
    protocol_version = "HTTP/1.1"

    # -------------------------------------------------------- 基础工具
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if getattr(self.server, "verbose", False):
            trace_id = self._trace_id()
            prefix = f" trace={trace_id}" if trace_id else ""
            sys.stderr.write(
                "[桥接] %s -%s %s\n" % (self.address_string(), prefix, fmt % args)
            )

    def _trace_id(self) -> str:
        """读取跨服务 trace；没有时返回空串，兼容旧调用。"""
        return str(self.headers.get("X-Trace-ID") or "").strip()[:128]

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        trace_id = self._trace_id()
        if trace_id:
            self.send_header("X-Trace-ID", trace_id)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body_data = dict(payload)
        trace_id = self._trace_id()
        if trace_id:
            body_data.setdefault("trace_id", trace_id)
        body = json.dumps(body_data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_body(self) -> bytes:
        """读取请求体。长度非法或超限时抛 ``ValueError``。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            raise ValueError("Content-Length 非法")
        if length < 0:
            raise ValueError("Content-Length 非法")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"请求体过大（上限 {MAX_BODY_BYTES // 1024 // 1024}MB）")
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _fail(
        self, message: str, status: int = 400, *, error_code: str | None = None
    ) -> None:
        self.close_connection = True
        self._send_json(
            {
                "error": message,
                "message": message,
                "error_code": error_code or f"BRIDGE_{status}",
            },
            status,
        )

    # ---------------------------------------------------------- 健康检查
    def _health(self) -> dict:
        from src import ollama_backend

        models: list[str] = []
        try:
            models = ollama_backend.list_models()
        except Exception:  # noqa: BLE001 - Ollama 没起也要能回 health
            models = []
        text_model = ""
        try:
            text_model = ollama_backend.resolve_text_model()
        except Exception:  # noqa: BLE001
            text_model = ""
        vision_model = str(OLLAMA_CONFIG.get("vl_model", ""))
        local_vision = bool(resolve_vl_model())
        ollama_vision = any(
            str(name).split(":", 1)[0] == vision_model.split(":", 1)[0]
            for name in models
        )

        return {
            "status": "ok",
            "service": "multimodal-bridge",
            "uptime_s": int(time.time() - START_TIME),
            "ollama": {
                "available": bool(models),
                "base_url": OLLAMA_CONFIG.get("base_url", ""),
                "models": models,
                "text_model": text_model,
            },
            "asr": {
                "available": MODELS.asr is not None or MODELS.asr_error is None,
                "loaded": MODELS.asr is not None,
                "model_path": str(MODEL_PATHS["asr"]),
                "error": MODELS.asr_error,
            },
            "mediapipe": {
                "available": all(Path(p).exists() for p in MODEL_ASSETS.values()),
                "loaded": MODELS.detector is not None,
                "error": MODELS.detector_error,
            },
            "tts": {
                "available": MODELS.tts_error is None,
                "voice": TTS_CONFIG.get("zh_voice", ""),
                "error": MODELS.tts_error,
            },
            "vision": {
                "available": bool(local_vision or ollama_vision or MODELS.vision),
                "initialized": MODELS.vision is not None,
                "model": "local-directory" if local_vision else vision_model,
                "error": MODELS.vision_error,
            },
        }

    # -------------------------------------------------------------- 路由
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path in ("/", "/health"):
                self._send_json(self._health())
            else:
                self._fail(f"未知路径：{path}", 404)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/asr":
                self._post_asr()
            elif path == "/gesture":
                self._post_gesture()
            elif path == "/vision":
                self._post_vision()
            elif path == "/tts":
                self._post_tts()
            elif path == "/llm":
                self._post_llm()
            else:
                self._fail(f"未知路径：{path}", 404)
        except ValueError as exc:
            self._fail(str(exc), 400, error_code="INVALID_INPUT")
        except Exception as exc:  # noqa: BLE001 - 任何异常都转成 JSON，不让进程崩
            traceback.print_exc()
            self._fail(
                f"{type(exc).__name__}: {exc}",
                500,
                error_code="BRIDGE_INTERNAL_ERROR",
            )

    # ------------------------------------------------------- 各能力处理器
    def _post_asr(self) -> None:
        content_type = (self.headers.get("Content-Type") or "").lower()
        language = ASR_CONFIG.get("language", "auto")
        if "application/json" in content_type:
            payload = self._read_json()
            b64 = str(payload.get("audio_b64") or "")
            if not b64:
                return self._fail(
                    "JSON 模式下必须提供 audio_b64", 400, error_code="MISSING_AUDIO"
                )
            try:
                wav = base64.b64decode(b64)
            except Exception as exc:  # noqa: BLE001
                return self._fail(
                    f"audio_b64 解码失败：{exc}", 400, error_code="INVALID_AUDIO"
                )
            language = str(payload.get("language") or language)
        else:
            wav = self._read_body()
        result = run_asr(wav, language=language)
        result["engine"] = "sensevoice-small"
        self._send_json(result)

    def _post_gesture(self) -> None:
        data = self._read_body()
        result = run_gesture(data)
        result["engine"] = "mediapipe"
        self._send_json(result)

    def _post_vision(self) -> None:
        content_type = (self.headers.get("Content-Type") or "").lower()
        if "application/json" not in content_type:
                return self._fail(
                "视觉接口需要 application/json，字段为 image_b64 和可选 categories",
                400,
                    error_code="INVALID_CONTENT_TYPE",
            )
        payload = self._read_json()
        b64 = str(payload.get("image_b64") or "")
        if not b64:
            return self._fail(
                "缺少 image_b64", 400, error_code="MISSING_IMAGE"
            )
        try:
            image = base64.b64decode(b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            return self._fail(
                f"image_b64 解码失败：{exc}", 400, error_code="INVALID_IMAGE"
            )

        raw_categories = payload.get("categories") or []
        if not isinstance(raw_categories, list):
            return self._fail(
                "categories 必须是数组", 400, error_code="INVALID_CATEGORIES"
            )
        categories = [
            str(value).strip()[:80]
            for value in raw_categories[:60]
            if str(value).strip()
        ]
        result = run_vision(image, categories)
        result["engine"] = "qwen-vl"
        self._send_json(result)

    def _post_tts(self) -> None:
        payload = self._read_json()
        text = str(payload.get("text") or "")
        voice = payload.get("voice") or None
        audio = run_tts(text, voice=str(voice) if voice else None)
        self._send_bytes(audio, "audio/mpeg")

    def _post_llm(self) -> None:
        payload = self._read_json()
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return self._fail(
                "messages 必须是数组", 400, error_code="INVALID_MESSAGES"
            )
        model = payload.get("model") or None
        try:
            temperature = float(payload.get("temperature", 0.2))
        except (TypeError, ValueError):
            temperature = 0.2
        max_tokens = payload.get("max_tokens")
        try:
            max_tokens = int(max_tokens) if max_tokens else None
        except (TypeError, ValueError):
            max_tokens = None
        result = run_llm(messages, model=str(model) if model else None,
                         temperature=temperature, max_tokens=max_tokens)
        self._send_json(result)


def serve(host: str = "127.0.0.1", port: int = 8100, verbose: bool = False,
          hands_only: bool = True) -> None:
    """启动桥接服务（阻塞）。"""
    global HANDS_ONLY
    HANDS_ONLY = bool(hands_only)
    server = ThreadingHTTPServer((host, port), BridgeHandler)
    server.daemon_threads = True
    server.verbose = verbose  # type: ignore[attr-defined]
    print(f"[桥接] 项目二多模态桥接服务已启动： http://{host}:{port}")
    print("[桥接] 接口：GET /health | POST /asr /gesture /vision /tts /llm")
    print(f"[桥接] 手势检测模式：{'仅 Hands（快）' if HANDS_ONLY else 'Pose+Hands+Face（全量）'}")
    print("[桥接] 项目一（Flask :5000）会自动探测本服务；停止本服务只影响多模态通道。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[桥接] 收到中断信号，正在退出……")
    finally:
        server.shutdown()
        server.server_close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="项目二 → 项目一 多模态桥接服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8100, help="监听端口（默认 8100）")
    parser.add_argument("--verbose", action="store_true", help="打印每个请求的访问日志")
    parser.add_argument("--full-detection", action="store_true",
                        help="手势检测跑 Pose+Hands+Face 全量模型（更慢，默认只用 Hands）")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    serve(host=args.host, port=args.port, verbose=args.verbose,
          hands_only=not args.full_detection)
