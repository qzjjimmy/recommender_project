"""阶段05 — 跨线程信号桥接模块。

使用 PyQt5 信号机制实现子线程到主线程 (GUI) 的安全通信。
"""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class Bridge(QObject):
    """跨线程信号桥。

    子线程通过发射信号，主线程在槽函数中更新 UI。
    绝不能直接在子线程中操作 UI 组件。
    """

    # 日志消息
    log_signal = pyqtSignal(str)

    # ASR 识别结果
    asr_text_signal = pyqtSignal(str)

    # LLM 流式 token
    llm_chunk_signal = pyqtSignal(str)

    # LLM 生成完毕
    llm_done_signal = pyqtSignal()

    # TTS 播放状态
    tts_status_signal = pyqtSignal(str)  # "playing" / "done" / "stopped"

    # 手势检测结果
    gesture_signal = pyqtSignal(str)

    # 知识库更新通知
    kb_updated_signal = pyqtSignal(int)  # chunk 数量

    # 视频帧就绪
    frame_ready_signal = pyqtSignal(object)  # np.ndarray

    # 系统状态
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._alive = True

    def emit_log(self, message: str):
        """线程安全地发射日志信号。"""
        if self._alive:
            self.log_signal.emit(message)

    def emit_asr_text(self, text: str):
        """线程安全地发射 ASR 结果信号。"""
        if self._alive:
            self.asr_text_signal.emit(text)

    def emit_llm_chunk(self, token: str):
        """线程安全地发射 LLM token 信号。"""
        if self._alive:
            self.llm_chunk_signal.emit(token)

    def emit_gesture(self, gesture: str):
        """线程安全地发射手势信号。"""
        if self._alive:
            self.gesture_signal.emit(gesture)

    def emit_llm_done(self):
        """线程安全地发射"LLM 生成完毕"信号。"""
        if self._alive:
            self.llm_done_signal.emit()

    def emit_tts_status(self, status: str):
        """线程安全地发射 TTS 状态（"playing" / "done" / "stopped"）。"""
        if self._alive:
            self.tts_status_signal.emit(status)

    def emit_kb_updated(self, chunk_count: int):
        """线程安全地发射知识库更新通知。"""
        if self._alive:
            self.kb_updated_signal.emit(chunk_count)

    def emit_frame(self, frame):
        """线程安全地发射视频帧。"""
        if self._alive:
            self.frame_ready_signal.emit(frame)

    def emit_status(self, message: str):
        """线程安全地发射系统状态。"""
        if self._alive:
            self.status_signal.emit(message)

    def shutdown(self):
        """标记桥接器为不可用，防止销毁后误访问。"""
        self._alive = False
