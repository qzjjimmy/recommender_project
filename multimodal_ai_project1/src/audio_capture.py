"""阶段01 — 麦克风音频采集模块。

使用 PyAudio 从系统麦克风实时采集 PCM 音频数据。
"""

from __future__ import annotations

import pyaudio
from config.settings import AUDIO_CONFIG


class AudioCapture:
    """实时音频采集器。

    封装 PyAudio，提供统一的 start/read_chunk/stop 接口。
    """

    def __init__(self, rate: int | None = None,
                 chunk: int | None = None,
                 channels: int | None = None):
        """初始化音频采集器。

        Args:
            rate: 采样率 (Hz)。默认使用 AUDIO_CONFIG。
            chunk: 每次读取的采样点数。默认使用 AUDIO_CONFIG。
            channels: 声道数。默认使用 AUDIO_CONFIG。
        """
        self.rate = rate or AUDIO_CONFIG["sample_rate"]
        self.chunk = chunk or AUDIO_CONFIG["chunk_size"]
        self.channels = channels or AUDIO_CONFIG["channels"]

        self.pa: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None

    def start(self):
        """打开麦克风并启动音频流。"""
        if self.stream is not None:
            return
        try:
            self.pa = pyaudio.PyAudio()
            if not self.list_input_devices():
                raise RuntimeError(
                    "没有检测到麦克风输入设备。请插入麦克风后重试；"
                    "本项不影响文本对话 / 知识库 / 视觉等其它链路。")
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except Exception as exc:
            # 打开失败时把自己清理干净，避免留下半开状态
            self.stop()
            raise RuntimeError(f"打开麦克风失败：{exc}") from exc

    def read_chunk(self) -> bytes:
        """读取一块音频数据。

        Returns:
            PCM 原始字节数据，每帧 2 字节 (int16)。
        """
        if self.stream is None:
            raise RuntimeError("音频流尚未打开，请先调用 start()。")
        # exception_on_overflow=False：缓冲区溢出时丢数据而不是抛异常，
        # 否则实时采集偶尔卡顿就会把整条链路打断。
        return self.stream.read(self.chunk, exception_on_overflow=False)

    def stop(self):
        """停止音频流并释放资源。"""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception:
                pass
            finally:
                self.pa = None

    # ------------------------------------------------------------------
    # 辅助方法（在原有签名基础上新增，便于自检与错误提示）
    # ------------------------------------------------------------------
    @staticmethod
    def list_input_devices() -> list[str]:
        """列出可用的输入设备名；pyaudio 不可用时返回空列表。"""
        try:
            pa = pyaudio.PyAudio()
        except Exception:
            return []
        names: list[str] = []
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    names.append(str(info.get("name", f"device-{i}")))
        except Exception:
            pass
        finally:
            try:
                pa.terminate()
            except Exception:
                pass
        return names

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
