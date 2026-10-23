"""阶段04 — 语音合成 (TTS) 模块。

基于 edge-tts 实现文本到语音的合成，支持中英文。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import langid

from config.settings import TTS_CONFIG


class TTSEngine:
    """语音合成引擎。

    使用 edge-tts (Microsoft Azure TTS) 进行在线语音合成。
    """

    def __init__(self, output_dir: str | None = None):
        """初始化 TTS 引擎。

        Args:
            output_dir: 音频文件输出目录。
        """
        self.output_dir = Path(output_dir or TTS_CONFIG["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.zh_voice = TTS_CONFIG["zh_voice"]
        self.en_voice = TTS_CONFIG["en_voice"]
        self._counter = 0   # 自动生成输出文件名时用

    def detect_language(self, text: str) -> str:
        """检测文本的主要语言。

        Args:
            text: 输入文本。

        Returns:
            "zh" 或 "en"。
        """
        try:
            lang, _ = langid.classify(text)
            return "zh" if lang.startswith("zh") else "en"
        except Exception:
            # 兜底：出现中文字符即判为中文
            return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

    async def _synthesize_async(self, text: str, voice: str,
                                output_path: str):
        """异步合成语音（内部方法）。"""
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def synthesize(self, text: str, output_path: str | None = None,
                   voice: str | None = None) -> str:
        """合成语音（同步接口）。

        Args:
            text: 要合成的文本。
            output_path: 输出 MP3 文件路径（可选，默认自动生成）。
            voice: 音色名称（可选，默认自动选择）。

        Returns:
            输出 MP3 文件路径。
        """
        if voice is None:
            voice = self.zh_voice if self.detect_language(text) == "zh" else self.en_voice
        if output_path is None:
            self._counter += 1
            output_path = str(self.output_dir / f"tts_{self._counter:04d}.mp3")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # 独立事件循环：本方法通常被工作线程调用，不能依赖外部 loop
        asyncio.run(self._synthesize_async(text, voice, output_path))
        return output_path

    def split_sentences(self, text: str) -> list[str]:
        """按标点将文本切分为句子。

        Args:
            text: 原始文本。

        Returns:
            句子列表（保留标点）。
        """
        # 用后向断言切分，标点会保留在前一句末尾
        newline = chr(10)
        pattern = "(?<=[。！？!?；;])|(?<=" + newline + ")"
        parts = re.split(pattern, text or "")
        return [p.strip() for p in parts if p and p.strip()]
