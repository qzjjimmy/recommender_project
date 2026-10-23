"""阶段05 — PyQt5 图形用户界面。

多模态AI助手的主窗口，集成视频预览、对话日志和控制面板。

线程与界面分工（重要）：
* 所有耗时逻辑（ASR、LLM、VL、TTS）都跑在后台线程里；
* 后台线程**绝不直接碰控件**，只通过 :class:`~src.bridge.Bridge` 的
  ``pyqtSignal`` 把结果投递回主线程；
* 界面定时器只负责"取最新帧 → 画出来"，保证拖动窗口/点击按钮不卡。
"""

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QLabel,
                              QTextEdit, QPushButton, QStatusBar, QLineEdit,
                              QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from config.settings import PATHS, VL_CONFIG
from src.bridge import Bridge
from src.human_detector import HumanDetector
from src.gesture import GestureRecognizer
from src.gesture_actions import GestureActionMapper
from src.modal_router import ModalRouter

# 句子结束标点：LLM 流式输出时按此切句，做到"边生成边说话"
_SENTENCE_END = "。！？!?；;\n"


class MainWindow(QMainWindow):
    """多模态AI助手主窗口。

    布局：左侧视频+字幕，右侧对话日志，底部控制栏。
    """

    WINDOW_TITLE = "多模态AI助手"
    DEFAULT_SIZE = (1280, 720)

    def __init__(self,
                 audio_thread=None,
                 video_thread=None,
                 human_detector: HumanDetector | None = None,
                 gesture_recognizer: GestureRecognizer | None = None,
                 rag_pipeline=None,
                 tts_player=None):
        """初始化主窗口。

        Args:
            audio_thread: 音频采集线程实例。
            video_thread: 视频采集线程实例。
            human_detector: 人体检测器实例。
            gesture_recognizer: 手势识别器实例。
            rag_pipeline: RAG 管线实例。
            tts_player: 流式 TTS 播放器实例。
        """
        super().__init__()

        # 保存组件引用
        self.audio_thread = audio_thread
        self.video_thread = video_thread
        self.human_detector = human_detector
        self.gesture_recognizer = gesture_recognizer
        self.rag_pipeline = rag_pipeline
        self.tts_player = tts_player

        # 手势动作映射
        self.action_mapper = GestureActionMapper()

        # 信号桥接
        self.bridge = Bridge()
        self._connect_signals()

        # 视频帧缓冲
        self.current_frame: np.ndarray | None = None
        self._overlay_frame: np.ndarray | None = None

        # 模态路由 + VL 懒加载 + 查询串行锁
        self.router = ModalRouter()
        self._vl_model = None
        self._query_lock = threading.Lock()
        self._llm_buffer = ""

        # 界面初始化
        self._setup_ui()

        # 视频刷新定时器 (约 30fps)
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_frame)
        self.frame_timer.start(33)

        # 人体/手势检测定时器（每 100ms，避免拖慢界面）
        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self._run_detection)
        self.detect_timer.start(100)

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _setup_ui(self):
        """搭建界面布局。"""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(*self.DEFAULT_SIZE)

        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        # 左侧：视频预览 + 字幕
        left = QVBoxLayout()
        self.video_label = QLabel("摄像头未开启")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background:#111;color:#888;border-radius:8px;")
        left.addWidget(self.video_label, stretch=1)
        self.subtitle_label = QLabel("字幕：—")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(
            "color:#333;background:#f2f4f8;border-radius:6px;padding:8px;")
        left.addWidget(self.subtitle_label)

        # 右侧：对话日志
        right = QVBoxLayout()
        right.addWidget(QLabel("对话 / 日志"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        right.addWidget(self.log_view, stretch=1)

        top.addLayout(left, stretch=3)
        top.addLayout(right, stretch=2)
        root.addLayout(top, stretch=1)

        # 底部：输入框 + 控制按钮
        bottom = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText(
            "输入问题后回车（含“看/画面/摄像头”自动切视觉模式；/text、/vision 可强制切换）")
        self.input_box.returnPressed.connect(self._on_submit)
        bottom.addWidget(self.input_box, stretch=1)

        self.btn_listen = QPushButton("🎤 开始聆听")
        self.btn_listen.clicked.connect(self._toggle_listen)
        self.btn_text = QPushButton("文本模式")
        self.btn_text.clicked.connect(lambda: self._set_mode("text"))
        self.btn_vision = QPushButton("视觉模式")
        self.btn_vision.clicked.connect(lambda: self._set_mode("vision"))
        self.btn_import = QPushButton("📄 导入文档")
        self.btn_import.clicked.connect(self._on_import_doc)
        self.btn_snapshot = QPushButton("📷 拍照")
        self.btn_snapshot.clicked.connect(self._take_snapshot)
        self.btn_clear = QPushButton("清空对话")
        self.btn_clear.clicked.connect(self._clear_chat)
        for btn in (self.btn_listen, self.btn_text, self.btn_vision,
                    self.btn_import, self.btn_snapshot, self.btn_clear):
            bottom.addWidget(btn)
        root.addLayout(bottom)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")
        self._set_mode("text")

    def _connect_signals(self):
        """连接所有信号槽。"""
        self.bridge.asr_text_signal.connect(self._on_asr_text)
        self.bridge.llm_chunk_signal.connect(self._on_llm_chunk)
        self.bridge.llm_done_signal.connect(self._on_llm_done)
        self.bridge.gesture_signal.connect(self._on_gesture)
        self.bridge.frame_ready_signal.connect(self._on_frame_ready)
        self.bridge.log_signal.connect(self._log)
        self.bridge.status_signal.connect(self.statusBar().showMessage)
        self.bridge.kb_updated_signal.connect(self._on_kb_updated)
        self.bridge.tts_status_signal.connect(
            lambda s: self.statusBar().showMessage(f"TTS: {s}"))

    # ------------------------------------------------------------------
    # 画面
    # ------------------------------------------------------------------
    def _run_detection(self):
        """在最近一帧上做人体/手势检测，结果画好备用。"""
        if self.human_detector is None or self.current_frame is None:
            return
        try:
            results = self.human_detector.process_frame(self.current_frame)
        except Exception as exc:
            self.human_detector = None      # 出错就停用，避免每周期都报错
            self.bridge.emit_log(f"检测模块已停用：{exc}")
            return
        self._overlay_frame = self.human_detector.draw_landmarks(
            self.current_frame, results)
        if self.gesture_recognizer is not None:
            hand = self.human_detector.primary_hand(results)
            if hand is not None:
                gesture = self.gesture_recognizer.recognize(hand)
                if gesture:
                    self.bridge.emit_gesture(gesture)

    def _update_frame(self):
        """定时刷新视频画面。"""
        frame = self._overlay_frame if self._overlay_frame is not None \
            else self.current_frame
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

    def _on_frame_ready(self, frame: np.ndarray):
        """视频帧就绪回调。"""
        self.current_frame = frame
        self._overlay_frame = None   # 新帧到了，旧标注作废

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    def _on_submit(self):
        """输入框回车。"""
        text = self.input_box.text().strip()
        if not text:
            return
        self.input_box.clear()
        self._append(f"\n[我] {text}\n")
        self._append(f"[提示] 输入: {text}\n")
        self._submit_query(text)

    def _on_asr_text(self, text: str):
        """ASR 识别结果回调。"""
        self.subtitle_label.setText(f"字幕：{text}")
        self._append(f"\n[用户] {text}\n")
        self._submit_query(text)

    def _submit_query(self, text: str):
        """把一次提问丢到后台线程处理（避免阻塞界面）。"""
        threading.Thread(target=self._handle_query, args=(text,),
                         daemon=True, name="QueryWorker").start()

    def _handle_query(self, text: str):
        """后台处理一次提问：路由 → 视觉或文本链路。"""
        if not text.strip() or not self._query_lock.acquire(blocking=False):
            return
        try:
            clean = self.router.strip_command(text)
            mode = self.router.route(text)
            self.bridge.emit_log(f"[路由] {mode} 模式：{clean}")
            if mode == "vision":
                self._handle_vision(clean)
            else:
                self._handle_text(clean)
        except Exception as exc:
            self.bridge.emit_log(f"[错误] 处理失败：{exc}")
        finally:
            self._query_lock.release()

    def _handle_text(self, query: str):
        """文本链路：检索知识库 → 流式生成 → 分句送 TTS。"""
        if self.rag_pipeline is None:
            self.bridge.emit_log("[提示] 文本模型未就绪，无法回答。")
            return
        context = None
        store = getattr(self.rag_pipeline, "vector_store", None)
        if store is not None and getattr(store, "documents", None):
            hits = store.search(query)
            context = self.rag_pipeline.build_context(hits) if hits else None
            if hits:
                self.bridge.emit_log(
                    f"[知识库] 命中 {len(hits)} 块，来源："
                    + "、".join(dict.fromkeys(h.get("source", "?") for h in hits)))
        self._append("[助手] ")
        self._llm_buffer = ""
        try:
            for chunk in self.rag_pipeline.text_llm.stream_generate(
                    query, context=context):
                self.bridge.emit_llm_chunk(chunk)
        finally:
            self.bridge.emit_llm_done()

    def _handle_vision(self, query: str):
        """视觉链路：取最近若干帧 → VL 模型回答（首次会懒加载模型）。"""
        frames = []
        if self.video_thread is not None and hasattr(self.video_thread, "latest_frames"):
            frames = self.video_thread.latest_frames(VL_CONFIG["num_frames"])
        if not frames:
            self.bridge.emit_llm_chunk("（没有可用的摄像头画面，请先开启摄像头。）")
            self.bridge.emit_llm_done()
            return
        try:
            if self._vl_model is None:
                self.bridge.emit_log("[视觉] 首次使用，正在加载 VL 模型…")
                from src.vl_model import VLModel

                self._vl_model = VLModel()
                self._vl_model.load()
            self._append("[视觉] ")
            answer = self._vl_model.generate_response(frames, query)
            self.bridge.emit_llm_chunk(answer)
            if self.tts_player is not None:
                self.tts_player.enqueue(answer)
        except Exception as exc:
            self.bridge.emit_llm_chunk(f"（视觉理解不可用：{exc}）")
        finally:
            self.bridge.emit_llm_done()

    def _on_llm_chunk(self, token: str):
        """LLM 流式 token 回调。"""
        self._append(token)
        self.subtitle_label.setText(f"字幕：{token}")
        # 累积到完整句子就送 TTS，实现"边生成边说话"
        self._llm_buffer += token
        if any(ch in self._llm_buffer for ch in _SENTENCE_END):
            if self.tts_player is not None:
                self.tts_player.enqueue(self._llm_buffer)
            self._llm_buffer = ""

    def _on_llm_done(self):
        """LLM 生成结束。"""
        if self._llm_buffer.strip() and self.tts_player is not None:
            self.tts_player.enqueue(self._llm_buffer)
        self._llm_buffer = ""
        self._append("\n")
        self.statusBar().showMessage("回答完成")

    # ------------------------------------------------------------------
    # 手势与动作
    # ------------------------------------------------------------------
    def _on_gesture(self, gesture: str):
        """手势检测回调。"""
        action = self.action_mapper.map(gesture)
        label = GestureRecognizer.label_of(gesture)
        self.statusBar().showMessage(f"手势：{label} → {self.action_mapper.describe(action)}")
        result = self.action_mapper.execute(action, handlers=self._action_handlers())
        self._append(f"[手势] {label} → {result}\n")

    def _action_handlers(self) -> dict:
        """把手势动作接到界面能力上（音量/停止/拍照/聆听）。"""
        return {
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "stop": self._stop_speaking,
            "start_listening": self._toggle_listen,
            "mute": self._stop_speaking,
            "take_snapshot": self._take_snapshot,
        }

    def _volume_up(self, **_) -> str:
        """调大系统音量（macOS 用 osascript，其它平台仅提示）。"""
        return self._set_volume(+10)

    def _volume_down(self, **_) -> str:
        """调小系统音量。"""
        return self._set_volume(-10)

    def _set_volume(self, delta: int) -> str:
        if sys.platform != "darwin":
            return f"已请求音量 {'+' if delta > 0 else ''}{delta}（当前平台需手动调整）"
        import subprocess

        try:
            cur = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=3)
            level = int(cur.stdout.strip() or 50)
        except Exception:
            level = 50
        level = max(0, min(100, level + delta))
        try:
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"],
                           capture_output=True, timeout=3)
        except Exception:
            pass
        return f"系统音量已设为 {level}%"

    def _stop_speaking(self, **_) -> str:
        """停止当前语音播放。"""
        if self.tts_player is not None:
            self.tts_player.clear()
            self.bridge.emit_tts_status("stopped")
        return "已停止播报"

    def _take_snapshot(self, **_) -> str:
        """保存当前画面到 outputs/snapshots/。"""
        if self.current_frame is None:
            return "暂时没有画面，无法拍照"
        out_dir = Path(PATHS["snapshots"])
        out_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(out_dir.glob("snapshot_*.jpg")))
        path = out_dir / f"snapshot_{index:03d}.jpg"
        cv2.imwrite(str(path), self.current_frame)
        return f"已保存画面：{path.name}"

    # ------------------------------------------------------------------
    # 控制按钮
    # ------------------------------------------------------------------
    def _toggle_listen(self):
        """开始 / 停止聆听。"""
        if self.audio_thread is None:
            self.bridge.emit_log("[提示] 音频线程未就绪（可能没有麦克风）。")
            return
        try:
            if self.audio_thread.is_alive():
                self.audio_thread.stop()
                self.btn_listen.setText("🎤 开始聆听")
                self.bridge.emit_status("已停止聆听")
            else:
                self.audio_thread.start()
                self.btn_listen.setText("⏹ 停止聆听")
                self.bridge.emit_status("正在聆听…")
        except Exception as exc:
            self.bridge.emit_log(f"[错误] 聆听切换失败：{exc}")

    def _set_mode(self, mode: str):
        """手动切换模态。"""
        self.router.set_mode(mode)
        active = "background:#2d7ff9;color:#fff;"
        self.btn_text.setStyleSheet(active if mode == "text" else "")
        self.btn_vision.setStyleSheet(active if mode == "vision" else "")
        self.bridge.emit_status(f"当前模式：{'文本' if mode == 'text' else '视觉'}")

    def _on_import_doc(self):
        """选择文档并导入知识库（后台执行，避免卡界面）。"""
        if self.rag_pipeline is None:
            self.bridge.emit_log("[提示] 知识库未就绪，无法导入文档。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择知识库文档", str(PATHS["data"]),
            "文档 (*.pdf *.txt *.md);;所有文件 (*)")
        if not path:
            return

        def worker() -> None:
            try:
                added = self.rag_pipeline.import_document(path)
                self.bridge.emit_kb_updated(
                    len(self.rag_pipeline.vector_store.documents))
                self.bridge.emit_log(
                    f"[知识库] 已导入 {Path(path).name}：新增 {added} 块")
            except Exception as exc:
                self.bridge.emit_log(f"[错误] 导入失败：{exc}")

        threading.Thread(target=worker, daemon=True, name="ImportDoc").start()

    def _on_kb_updated(self, chunk_count: int):
        """知识库更新通知。"""
        self.statusBar().showMessage(f"知识库已更新，共 {chunk_count} 块")

    def _clear_chat(self):
        """清空对话区。"""
        self.log_view.clear()
        self.subtitle_label.setText("字幕：—")
        self.bridge.emit_log("[提示] 对话已清空")

    # ------------------------------------------------------------------
    def _append(self, text: str):
        """往日志区追加文本（自动滚到底）。"""
        self.log_view.moveCursor(self.log_view.textCursor().End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(self.log_view.textCursor().End)

    def _log(self, message: str):
        """桥接过来的日志消息。"""
        self._append(f"{message}\n")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """窗口关闭事件 — 安全停止所有线程并释放资源。"""
        # 1. 停止定时器，避免关闭过程中还在刷新
        try:
            self.frame_timer.stop()
            self.detect_timer.stop()
        except Exception:
            pass
        # 2. 停止音频线程（其 stop() 内部会 join 并释放麦克风）
        if self.audio_thread is not None:
            try:
                self.audio_thread.stop()
            except Exception:
                pass
        # 3. 停止视频线程
        if self.video_thread is not None:
            try:
                self.video_thread.stop()
            except Exception:
                pass
        # 4. 停止 TTS 播放线程
        if self.tts_player is not None:
            try:
                self.tts_player.stop()
            except Exception:
                pass
        # 5. 关闭 pygame.mixer
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass
        # 6. 清理临时文件（保留 outputs 下的正式产物）
        try:
            shutil.rmtree(Path(PATHS["temp"]), ignore_errors=True)
        except Exception:
            pass
        # 7. 释放检测器
        if self.human_detector is not None:
            try:
                self.human_detector.close()
            except Exception:
                pass
        # 8. 关闭信号桥
        self.bridge.shutdown()
        event.accept()
