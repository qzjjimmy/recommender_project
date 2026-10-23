"""阶段02 — 语音识别 (ASR) 模块。

基于 FunASR / SenseVoiceSmall 实现语音到文本的转换。
"""

from __future__ import annotations

import re

from config.settings import ASR_CONFIG, DEVICE


class ASREngine:
    """语音识别引擎。

    封装 SenseVoiceSmall 模型，提供 transcribe 接口。
    """

    def __init__(self, model_path: str, device: str | None = None):
        """初始化 ASR 引擎。

        Args:
            model_path: SenseVoiceSmall 模型目录路径。
            device: 推理设备 ("cuda" / "cpu")。
        """
        self.model_path = model_path
        self.device = device or DEVICE
        self.model = None  # TODO: 使用 funasr.AutoModel 加载

    def load(self):
        """加载 SenseVoiceSmall 模型（首次调用较慢）。

        单独抽成方法，便于脚本里预热一次，避免第一句识别特别慢。
        """
        if self.model is not None:
            return self
        from funasr import AutoModel

        self.model = AutoModel(
            model=self.model_path,
            device=self.device,
            disable_update=True,   # 关掉联网检查更新，保证离线也能跑
            trust_remote_code=False,
        )
        return self

    def _ensure_loaded(self):
        """确保模型已加载（懒加载）。"""
        if self.model is None:
            self.load()

    def transcribe(self, audio_path: str, language: str = "auto") -> str:
        """将音频文件转写为文本。

        Args:
            audio_path: WAV 音频文件路径（16kHz, 单声道, 16bit PCM）。
            language: 语言 ("auto" / "zh" / "en")。

        Returns:
            转写后的纯文本。
        """
        try:
            self._ensure_loaded()
            result = self.model.generate(
                input=audio_path,
                cache={},
                language=language,
                use_itn=ASR_CONFIG["use_itn"],
                batch_size_s=60,
            )
        except Exception as exc:
            # 模型缺失 / 音频格式不对等：返回空串，由上层提示"未识别到语音"
            print(f"[ASR] 识别失败：{exc}")
            return ""

        if not result:
            return ""
        first = result[0]
        raw = first.get("text", "") if isinstance(first, dict) else str(first)
        return self._clean_result(raw)

    def _clean_result(self, raw_text: str) -> str:
        """清洗 ASR 原始输出。

        去除语言标记、情感标记等前缀，提取纯文本。
        """
        if not raw_text:
            return ""
        # SenseVoice 形如 "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好，多模态AI助手"
        # 标记都在 ">" 之前，取最后一段即是纯文本
        text = raw_text.split(">")[-1]
        # 兜底：清掉可能残留的 <|...|> 标记
        text = re.sub(r"<\|[^|]*\|>", "", text)
        text = re.sub(r"<[^>]{1,20}>", "", text)
        return text.strip()
