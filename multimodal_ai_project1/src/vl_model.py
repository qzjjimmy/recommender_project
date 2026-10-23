"""阶段04 — 视觉语言模型模块。

基线：Qwen2.5-VL-3B-Instruct（bfloat16）。

两个必须注意的点：
1. 加载类不是固定的。Qwen2-VL 用 `Qwen2VLForConditionalGeneration`，
   Qwen2.5-VL 用 `Qwen2_5_VLForConditionalGeneration`，写死会直接报错。
   按模型目录 config.json 的 `model_type` 自动判别最稳：
       "qwen2_5_vl" → Qwen2_5_VLForConditionalGeneration
       "qwen2_vl"   → Qwen2VLForConditionalGeneration

2. 必须用 bfloat16。CPU 上曾用 float32 加载 + int8 动态量化，但量化过程需
   同时持有 fp32(8.8GB) 与 int8(2.2GB) 两份模型，叠加 ASR + 文本LLM 后
   峰值 23.4 GB，在 32 GB 机器上直接 std::bad_alloc。改 bf16 后实测：
       内存峰值 23.4 GB → 6.2 GB
       加载耗时  26.0 s → 1.1 s
       推理耗时   7.5 s → 8.2 s（基本持平，未牺牲速度）
   另注：**光加 low_cpu_mem_usage=True 解决不了**（实测峰值反而略升）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.settings import (DEVICE, OLLAMA_CONFIG, PATHS, VL_CONFIG,
                             resolve_vl_model)
from src.ollama_backend import OllamaBackend, OllamaError


def detect_model_class(model_path: str):
    """按模型目录的 config.json 识别模型家族。

    这样同一份代码既能加载 Qwen2.5-VL，也能加载 Qwen2-VL，
    换模型时无需改代码。

    Args:
        model_path: 模型目录路径。

    Returns:
        (模型类, 家族名)。识别失败时默认回退到 Qwen2-VL。
    """
    cfg_file = Path(model_path) / "config.json"
    model_type = ""
    if cfg_file.exists():
        try:
            model_type = json.loads(
                cfg_file.read_text(encoding="utf-8")
            ).get("model_type", "")
        except Exception:
            pass

    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration, "Qwen2.5-VL"
    return Qwen2VLForConditionalGeneration, "Qwen2-VL"


class VLModel:
    """视觉语言模型。

    支持多帧图像的视觉问答（"看场景"模式）。
    """

    SYSTEM_PROMPT = """你是多模态AI助手的视觉模块。
你能看到摄像头的实时画面。请基于画面内容回答用户的问题。
回答要准确、简洁。如果画面中看不清，请说明。"""

    def __init__(self, model_path: str | None = None,
                 device: str | None = None):
        """初始化 VL 模型。

        Args:
            model_path: 模型**目录**路径。留空则由 resolve_vl_model() 自动挑选。
            device: 推理设备。
        """
        # 后端选择策略：
        #   显式传了本地目录 → transformers；
        #   否则本地 models/ 下有 VL 目录 → transformers；
        #   否则如果 Ollama 里已有视觉模型 → ollama（本项目默认走这条）。
        self.backend = "transformers"
        self.backend_client: OllamaBackend | None = None
        if model_path is None:
            model_path = resolve_vl_model()
        if not model_path:
            backend = OllamaBackend()
            if backend.is_available() and backend.has_model(OLLAMA_CONFIG["vl_model"]):
                self.backend = "ollama"
                self.backend_client = backend
                model_path = OLLAMA_CONFIG["vl_model"]
            else:
                raise FileNotFoundError(
                    "未找到可用的视觉语言模型。\n"
                    f"  方式一（推荐）：安装并启动 Ollama 后执行 ollama pull "
                    f"{OLLAMA_CONFIG['vl_model']}\n"
                    "  方式二：把 Qwen2.5-VL-3B / Qwen2-VL-2B 放到 models/ 下，"
                    "并检查 config/settings.py 的 MODEL_PATHS。"
                )

        self.model_path = model_path
        self.device = device or DEVICE
        self.dtype_name = VL_CONFIG["dtype"]  # "bfloat16"

        self.processor = None  # TODO: AutoProcessor
        self.model = None      # TODO: Qwen2_5_VLForConditionalGeneration
        self.family = ""

    @staticmethod
    def is_available() -> bool:
        """是否有可用的 VL 后端（本地模型目录 或 Ollama 视觉模型）。"""
        if resolve_vl_model():
            return True
        try:
            backend = OllamaBackend()
            return backend.is_available() and backend.has_model(
                OLLAMA_CONFIG["vl_model"])
        except OllamaError:
            return False

    @staticmethod
    def availability_note() -> str:
        """返回一句话说明当前 VL 的来源，供脚本/界面显示。"""
        local = resolve_vl_model()
        if local:
            return f"本地模型目录：{local}"
        try:
            backend = OllamaBackend()
            if backend.is_available() and backend.has_model(OLLAMA_CONFIG["vl_model"]):
                return f"Ollama 模型：{OLLAMA_CONFIG['vl_model']}"
            if backend.is_available():
                return (f"Ollama 已启动，但缺少 {OLLAMA_CONFIG['vl_model']}"
                        f"（请执行 ollama pull {OLLAMA_CONFIG['vl_model']}）")
            return "Ollama 未启动，且本地没有 VL 模型目录"
        except OllamaError as exc:
            return f"Ollama 不可用：{exc}"

    def load(self):
        """加载模型和处理器。

        注意：VL 模型体积大（7.2 GB），建议**懒加载**——
        首次点击"看场景"时再加载，并用信号回报"加载中",
        否则界面看起来像卡死（见 VL_CONFIG["lazy_load"]）。
        """
        if self.backend == "ollama":
            # Ollama 侧模型常驻由服务管理，这里只做可用性确认
            if self.backend_client is None:
                self.backend_client = OllamaBackend()
            if not self.backend_client.is_available():
                raise OllamaError("Ollama 服务未启动，无法使用视觉能力。")
            if not self.backend_client.has_model(self.model_path):
                raise OllamaError(
                    f"Ollama 中缺少视觉模型 {self.model_path}，"
                    f"请执行：ollama pull {self.model_path}")
            self.family = "Ollama"
            return self

        if self.model is not None:
            return self
        import torch
        from transformers import AutoProcessor

        model_cls, self.family = detect_model_class(self.model_path)
        dtype = getattr(torch, self.dtype_name, torch.bfloat16)
        try:
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=dtype,          # 必须 bfloat16（见文件头说明）
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True)
        except TypeError:
            self.model = model_cls.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True)
        except Exception:
            # 少数平台不支持 bfloat16，退回 float32 保证能跑起来
            self.model = model_cls.from_pretrained(
                self.model_path, torch_dtype=torch.float32,
                low_cpu_mem_usage=True)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, use_fast=True)
        return self

    def _ensure_loaded(self):
        """懒加载入口（VL 模型很大，不要一启动就加载）。"""
        if self.backend == "ollama":
            if self.backend_client is None:
                self.load()
            return
        if self.model is None:
            self.load()

    def process_frames(self, frames: list[np.ndarray],
                       max_frames: int | None = None) -> list[str]:
        """处理视频帧，保存为临时图片并返回路径列表。

        Args:
            frames: 视频帧列表 (numpy BGR 数组)。
            max_frames: 最多使用的帧数，默认取 VL_CONFIG["num_frames"]（2）。
                帧数越多 prefill 越慢，2 帧通常已能看清主体。

        Returns:
            图片文件路径列表。
        """
        import cv2

        max_frames = max_frames or VL_CONFIG["num_frames"]
        if not frames:
            return []
        selected = frames[-max_frames:] if len(frames) > max_frames else list(frames)
        # 缩到最大边长以内再送模型：视觉 token 数与像素面积近似成正比，
        # 缩小后 4 帧也能稳稳落在上下文窗口内。
        max_side = VL_CONFIG.get("max_side", 0)
        temp_dir = Path(PATHS["temp"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, frame in enumerate(selected):
            if max_side:
                h, w = frame.shape[:2]
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA)
            path = temp_dir / f"vl_frame_{i:02d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(str(path))
        return paths

    @staticmethod
    def prepare_image_bytes(image_bytes: bytes,
                            max_side: int = 768) -> tuple[bytes, int, int]:
        """解码并重新编码上传图片。

        重新编码有两个目的：把任意尺寸图片缩到适合本地 VL 推理的大小，
        同时去掉 EXIF、GPS 等原始元数据。整个过程只在内存中完成，不落盘。

        Args:
            image_bytes: 原始 JPEG/PNG 字节。
            max_side: 输出图片最长边像素数。

        Returns:
            ``(jpeg_bytes, width, height)``。
        """
        if not image_bytes:
            raise ValueError("图片内容为空")

        import cv2

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码图片，请确认上传的是 JPEG 或 PNG")

        h, w = frame.shape[:2]
        if max_side > 0:
            scale = min(1.0, float(max_side) / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("图片重新编码失败")
        out_h, out_w = frame.shape[:2]
        return encoded.tobytes(), out_w, out_h

    def generate_response_from_image_bytes(
        self,
        image_bytes: bytes,
        user_text: str,
        *,
        system: str | None = None,
        max_side: int = 768,
    ) -> str:
        """用单张图片字节执行一次视觉问答。

        这是网页桥接使用的入口。Ollama 后端直接接收内存中的 JPEG 字节；
        transformers 后端复用 ``generate_response``，并由它的清理逻辑删除临时帧。
        """
        prepared, _, _ = self.prepare_image_bytes(image_bytes, max_side=max_side)
        self._ensure_loaded()

        if self.backend == "ollama":
            return self.backend_client.chat_with_images(
                user_text,
                [prepared],
                system=system or self.SYSTEM_PROMPT,
                model=self.model_path,
                max_tokens=VL_CONFIG["max_new_tokens"],
            )

        import cv2

        array = np.frombuffer(prepared, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("预处理后的图片无法解码")
        return self.generate_response([frame], user_text, num_frames=1)

    def create_messages(self, frame_paths: list[str],
                        user_text: str) -> list[dict]:
        """构建多模态 messages。

        Args:
            frame_paths: 图片路径列表。
            user_text: 用户问题。

        Returns:
            Qwen-VL 格式的 messages。
        """
        return [{
            "role": "user",
            "content": [
                {"type": "video",
                 "video": frame_paths,
                 "fps": 1.0,
                 "max_pixels": VL_CONFIG["max_pixels"],  # 限制视觉 token 数
                 },
                {"type": "text", "text": user_text},
            ],
        }]

    def generate_response(self, frames: list[np.ndarray],
                          user_text: str,
                          num_frames: int | None = None) -> str:
        """基于视频帧和用户文本生成回答。

        Args:
            frames: 视频帧列表。
            user_text: 用户问题。
            num_frames: 使用的帧数。

        Returns:
            模型生成的回答文本。
        """
        if not frames:
            return "（当前没有可用的画面，请检查摄像头是否正常工作。）"

        paths = self.process_frames(frames, num_frames)
        try:
            if self.backend == "ollama":
                self._ensure_loaded()
                return self.backend_client.chat_with_images(
                    user_text, paths, system=self.SYSTEM_PROMPT,
                    model=self.model_path,
                    max_tokens=VL_CONFIG["max_new_tokens"])

            self._ensure_loaded()
            import torch
            from qwen_vl_utils import process_vision_info

            messages = self.create_messages(paths, user_text)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=VL_CONFIG["max_new_tokens"])
            gen_ids = [g[len(i):] for i, g in
                       zip(inputs["input_ids"], gen_ids)]
            return self.processor.batch_decode(
                gen_ids, skip_special_tokens=True)[0].strip()
        finally:
            # 临时图片用完即删，避免 outputs/temp 越积越多
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
