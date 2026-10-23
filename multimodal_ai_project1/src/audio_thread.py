"""阶段02 — 音频采集线程。

将 AudioCapture + VAD + ASR 串联为独立线程中的完整流水线。
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from collections.abc import Callable

from config.settings import PATHS, AUDIO_CONFIG
from src.audio_capture import AudioCapture
from src.vad import VADProcessor
from src.asr import ASREngine


class AudioStreamThread(threading.Thread):
    """实时音频处理线程。

    主循环: 音频采集 → VAD断句 → 保存WAV → ASR识别 → 回调通知。
    """

    def __init__(self,
                 audio_capture: AudioCapture,
                 vad_processor: VADProcessor,
                 asr_engine: ASREngine,
                 on_result: Callable[[str], None] | None = None):
        """初始化音频线程。

        Args:
            audio_capture: 音频采集器实例。
            vad_processor: VAD 处理器实例。
            asr_engine: ASR 引擎实例。
            on_result: ASR 识别结果回调函数。
        """
        super().__init__(daemon=True)
        self.capture = audio_capture
        self.vad = vad_processor
        self.asr = asr_engine
        self.on_result = on_result

        self._stop_event = threading.Event()
        self._frame_size = self.vad.frame_size
        self._temp_dir = Path(PATHS["temp"])

    def run(self):
        """线程主循环。"""
        try:
            self.capture.start()
        except Exception as exc:
            print(f"[音频线程] 启动失败，已退出：{exc}")
            return

        frame_bytes = self._frame_size * 2   # 16bit = 2 字节/采样点
        buffer = bytearray()
        index = 0
        print("[音频线程] 已启动，请开始说话…")

        while not self._stop_event.is_set():
            try:
                chunk = self.capture.read_chunk()
            except Exception as exc:
                print(f"[音频线程] 采集失败，已退出：{exc}")
                break

            buffer.extend(chunk)

            # 按 30ms 一帧喂给 VAD；不足一帧的留在缓冲区
            while len(buffer) >= frame_bytes:
                frame = bytes(buffer[:frame_bytes])
                del buffer[:frame_bytes]
                try:
                    segmented = self.vad.process_frame(frame)
                except Exception as exc:
                    print(f"[音频线程] VAD 处理异常，已重置：{exc}")
                    self.vad.reset()
                    continue
                if not segmented:
                    continue

                # 最短语音过滤（VAD 内部已判一次，这里再兜一层）
                if self.vad.speech_frame_count < self.vad.min_speech_frames:
                    self.vad.discard_segment()
                    continue

                segment = self.vad.get_segment()
                if not segment:
                    continue

                wav_path = self._temp_dir / f"asr_segment_{index:04d}.wav"
                index += 1
                try:
                    self._save_wav(segment, wav_path)
                    text = self.asr.transcribe(
                        str(wav_path), language=ASR_CFG_LANGUAGE())
                except Exception as exc:
                    print(f"[音频线程] 识别失败：{exc}")
                    continue

                if text and self.on_result:
                    self.on_result(text)

        print("[音频线程] 已停止。")

    def stop(self):
        """安全停止线程。"""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=3)
        try:
            self.capture.stop()
        except Exception:
            pass

    def _save_wav(self, audio_data: bytes, filepath: Path):
        """将 PCM 数据保存为 WAV 文件。"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(AUDIO_CONFIG["channels"])
            wf.setsampwidth(AUDIO_CONFIG["sample_width"])
            wf.setframerate(AUDIO_CONFIG["sample_rate"])
            wf.writeframes(audio_data)


def ASR_CFG_LANGUAGE() -> str:
    """取配置里固定的识别语种。

    单独抽成函数是为了让 run() 里的调用点保持简洁；
    ASR_CONFIG["language"] 固定为 "zh"——短句上传 "auto" 会来回误判语种。
    """
    from config.settings import ASR_CONFIG

    return ASR_CONFIG["language"]
