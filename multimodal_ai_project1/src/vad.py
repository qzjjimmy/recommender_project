"""阶段02 — 语音活动检测 (VAD) 模块。

基于 WebRTC VAD 实现实时语音段检测和智能断句。
"""

from __future__ import annotations

from collections import deque

import webrtcvad
from config.settings import VAD_CONFIG, AUDIO_CONFIG


class VADProcessor:
    """语音活动检测处理器。

    维护音频缓冲区，持续判断语音/静音帧，在连续静音超过阈值时触发断句。
    """

    FRAME_MS = 30  # VAD 每帧时长 (ms)

    def __init__(self,
                 sample_rate: int | None = None,
                 silence_duration_ms: int | None = None,
                 min_speech_duration_ms: int | None = None,
                 aggressiveness: int | None = None):
        """初始化 VAD 处理器。

        Args:
            sample_rate: 音频采样率。
            silence_duration_ms: 连续静音触发断句的时长 (ms)。
            min_speech_duration_ms: 最短有效语音时长 (ms)。
            aggressiveness: VAD 灵敏度 (0-3)。
        """
        self.sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
        self.frame_size = int(self.sample_rate * self.FRAME_MS / 1000)

        silence_ms = silence_duration_ms or VAD_CONFIG["silence_duration_ms"]
        self.silence_threshold = silence_ms // self.FRAME_MS

        min_speech_ms = min_speech_duration_ms or VAD_CONFIG["min_speech_duration_ms"]
        self.min_speech_frames = min_speech_ms // self.FRAME_MS

        mode = aggressiveness or VAD_CONFIG["aggressiveness"]
        self.vad = webrtcvad.Vad(mode)

        # 内部状态
        self.audio_buffer: list[bytes] = []
        self.silence_count: int = 0
        self.speech_frame_count: int = 0

        # 前置缓冲：VAD 判定有一定延迟，保存语音开始前的一小段，避免开头被切掉
        pre_roll_ms = VAD_CONFIG.get("pre_roll_ms", 0)
        self._pre_roll_frames = max(1, pre_roll_ms // self.FRAME_MS)
        self._pre_roll: deque[bytes] = deque(maxlen=self._pre_roll_frames)

        # 尾部补齐：句子结尾多留一点静音，避免尾音被截断
        self._tail_padding_bytes = int(
            self.sample_rate * VAD_CONFIG.get("tail_padding_ms", 0) / 1000) * 2

    def process_frame(self, frame: bytes) -> bool:
        """处理一帧音频数据。

        Args:
            frame: 一帧 PCM 数据，长度必须等于 frame_size * 2 (16bit)。

        Returns:
            是否触发断句（连续静音超过阈值）。
        """
        if len(frame) != self.frame_size * 2:
            raise ValueError(
                f"VAD 帧长度必须是 {self.frame_size * 2} 字节"
                f"（{self.FRAME_MS}ms @ {self.sample_rate}Hz, 16bit），"
                f"实际收到 {len(frame)} 字节")

        is_speech = self.vad.is_speech(frame, self.sample_rate)

        if is_speech:
            if not self.audio_buffer:
                # 语音刚开始：把前置缓冲里的静音补进来，保住句首
                self.audio_buffer.extend(self._pre_roll)
                self._pre_roll.clear()
            self.audio_buffer.append(frame)
            self.speech_frame_count += 1
            self.silence_count = 0
            return False

        # ---- 静音帧 ----
        if not self.audio_buffer:
            # 还没开始说话：只滚动前置缓冲，不消耗计数
            self._pre_roll.append(frame)
            return False

        self.audio_buffer.append(frame)   # 句尾静音也保留，作为"尾巴"
        self.silence_count += 1
        if self.silence_count < self.silence_threshold:
            return False

        # 连续静音够长 → 判定一句结束
        if self.speech_frame_count >= self.min_speech_frames:
            return True

        # 语音太短（咳嗽、敲键盘之类）：丢弃，不触发识别
        self.discard_segment()
        return False

    def get_segment(self) -> bytes:
        """取出当前语音段的完整音频数据并重置状态。

        Returns:
            拼接后的 PCM 字节数据。
        """
        data = b"".join(self.audio_buffer)
        # 保证句尾至少有 tail_padding 的静音，避免尾音缺失
        if self._tail_padding_bytes:
            tail_silence = self._tail_padding_bytes
            # 粗略估算尾部已有的静音字节数（每个静音帧 frame_size*2 字节）
            already = min(self.silence_count, self.silence_threshold) * self.frame_size * 2
            if already < tail_silence:
                data += b"\x00" * (tail_silence - already)
        self.reset()
        return data

    def discard_segment(self):
        """丢弃当前语音段（用于最短语音过滤）。"""
        self.audio_buffer.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self._pre_roll.clear()

    def reset(self):
        """完全重置 VAD 状态。"""
        self.audio_buffer.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self._pre_roll.clear()
