"""阶段06 — 多模态AI助手统一启动入口。

本文件负责组装各模块、创建 GUI，并在退出时统一释放资源。
学生完成各阶段 src/ 中的 TODO 后，在项目根目录运行：
    python main.py
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication

from config.settings import OLLAMA_CONFIG, PATHS, set_seed
from src.gui import MainWindow


class VideoThread(threading.Thread):
    """视频采集线程：持续读帧、保留最近若干帧、把帧回调给界面。

    视觉链路需要"最近 2 帧"送给 VL 模型，所以维护一个环形缓冲；
    帧通过回调（由 main.py 接到 Bridge 信号）投递到主线程绘制。
    """

    def __init__(self, capture, target_fps: int = 30, buffer_size: int = 30):
        super().__init__(daemon=True, name="VideoThread")
        self.capture = capture
        self.frames: deque = deque(maxlen=buffer_size)
        self._interval = 1.0 / max(1, target_fps)
        self._stop_event = threading.Event()
        self._callback = None
        self._lock = threading.Lock()

    def set_callback(self, callback):
        """设置帧回调（一般是 window.bridge.emit_frame）。"""
        self._callback = callback

    def run(self):
        """主循环：读帧 → 入缓冲 → 回调。"""
        try:
            self.capture.start()
        except Exception as exc:
            print(f"[视频线程] 启动失败，已退出：{exc}")
            return
        print("[视频线程] 已启动。")
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                frame = self.capture.read_frame()
            except Exception as exc:
                print(f"[视频线程] 读帧失败，已退出：{exc}")
                break
            if frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frames.append(frame)
            if self._callback is not None:
                try:
                    self._callback(frame)
                except Exception:
                    pass
            cost = time.time() - t0
            if cost < self._interval:
                time.sleep(self._interval - cost)
        print("[视频线程] 已停止。")

    def latest_frame(self):
        """取最近一帧。"""
        with self._lock:
            return self.frames[-1] if self.frames else None

    def latest_frames(self, n: int) -> list:
        """取最近 n 帧（旧的在前）。"""
        with self._lock:
            frames = list(self.frames)
        return frames[-n:] if n > 0 else frames

    def stop(self, timeout: float = 3.0):
        """停止线程并释放摄像头。"""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        try:
            self.capture.stop()
        except Exception:
            pass


def build_application() -> MainWindow:
    """按依赖顺序创建系统组件并返回主窗口。

    设计原则：配置先加载，模型与硬件组件后初始化；子线程通过 Bridge
    信号与界面通信，禁止在子线程中直接更新 UI。
    任何一条链路不可用（没麦克风 / 没摄像头 / Ollama 没启动）都只打印提示，
    其余功能照常可用——绝不能因为一个模块异常就让整个程序起不来。
    """
    set_seed()

    # ---- 阶段02：音频链路（麦克风 → VAD → ASR → 回调）----
    audio_thread = None
    try:
        from src.asr import ASREngine
        from src.audio_capture import AudioCapture
        from src.audio_thread import AudioStreamThread
        from src.vad import VADProcessor

        audio_capture = AudioCapture()
        vad = VADProcessor()
        asr = ASREngine(str(Path(PROJECT_ROOT) / "models" / "SenseVoiceSmall"))
        audio_thread = AudioStreamThread(audio_capture, vad, asr)
        # on_result 由主窗口创建后再接：这里先不设，避免拿不到 bridge
    except Exception as exc:
        print(f"[启动] 音频链路不可用：{exc}")

    # ---- 阶段03：文本 LLM + 知识库 RAG ----
    # 用 create_text_llm() 工厂自动选后端（Ollama / GGUF / transformers）
    rag_pipeline = None
    try:
        from src.knowledge_base import VectorStore
        from src.rag import RAGPipeline
        from src.text_llm import create_text_llm

        text_llm = create_text_llm()
        # 模型加载放在后台：Ollama 首次加载也可能要几秒，别卡住窗口
        try:
            text_llm.load()
            print("[启动] 文本大模型已就绪。")
        except Exception as exc:
            print(f"[启动] 文本大模型加载失败：{exc}")
        rag_pipeline = RAGPipeline(VectorStore(), text_llm)
    except Exception as exc:
        print(f"[启动] 文本对话不可用：{exc}\n"
              f"      提示：安装并启动 Ollama 后执行 "
              f"ollama pull {OLLAMA_CONFIG['text_model']}")

    # ---- 阶段04：TTS（VL 模型懒加载，不在这里创建）----
    tts_player = None
    try:
        from src.tts import TTSEngine
        from src.tts_player import StreamingTTSPlayer

        tts_engine = TTSEngine()
        tts_player = StreamingTTSPlayer(tts_engine)
        tts_player.start()
        print("[启动] 语音合成已就绪。")
    except Exception as exc:
        print(f"[启动] 语音合成不可用：{exc}")

    # ---- 阶段05：摄像头 + 检测/手势 ----
    video_thread = None
    try:
        from src.video_capture import VideoCapture

        video_thread = VideoThread(VideoCapture())
    except Exception as exc:
        print(f"[启动] 摄像头不可用：{exc}")

    human_detector = None
    try:
        from src.human_detector import HumanDetector

        human_detector = HumanDetector()
    except Exception as exc:
        print(f"[启动] 人体/手势检测不可用：{exc}")

    gesture_recognizer = None
    try:
        from src.gesture import GestureRecognizer

        gesture_recognizer = GestureRecognizer()
    except Exception as exc:
        print(f"[启动] 手势识别不可用：{exc}")

    # ---- 组装主窗口 ----
    window = MainWindow(
        audio_thread=audio_thread,
        video_thread=video_thread,
        human_detector=human_detector,
        gesture_recognizer=gesture_recognizer,
        rag_pipeline=rag_pipeline,
        tts_player=tts_player,
    )

    # ---- 把各线程的结果接到主窗口的信号桥上 ----
    if audio_thread is not None:
        audio_thread.on_result = lambda text: window.bridge.emit_asr_text(text)
    if video_thread is not None:
        video_thread.set_callback(window.bridge.emit_frame)
        video_thread.start()
        window.bridge.emit_status("摄像头已启动")
    else:
        window.video_label.setText("摄像头不可用（可继续用文本/知识库问答）")

    window.bridge.emit_log("[启动] 系统就绪。输入问题或点击「开始聆听」。")
    return window


def main() -> int:
    """启动 Qt 应用事件循环。"""
    app = QApplication(sys.argv)
    window = build_application()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
