"""阶段04 — 流式 TTS 播放器。

实现分句合成 + 逐句播放，降低首句响应延迟。
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import pygame

from src.tts import TTSEngine

# 队列结束标记：stop() 时投一个，播放线程收到就退出
_SENTINEL = object()


class StreamingTTSPlayer:
    """流式语音播放器。

    将文本分句后异步合成并逐句播放，避免长文本"等待不出声"。
    """

    def __init__(self, tts_engine: TTSEngine):
        """初始化播放器。

        Args:
            tts_engine: TTSEngine 实例。
        """
        self.tts = tts_engine
        self.sentence_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._player_thread: threading.Thread | None = None

    def start(self):
        """启动后台播放线程。"""
        if self._player_thread and self._player_thread.is_alive():
            return
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000)
        self._stop_event.clear()
        self._player_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="TTSPlayer")
        self._player_thread.start()

    def enqueue(self, text: str):
        """将文本分句后加入播放队列。

        Args:
            text: 要播放的文本（会自动分句）。
        """
        for sentence in self.tts.split_sentences(text):
            self.sentence_queue.put(sentence)

    def stop(self):
        """停止播放并清空队列。"""
        self._stop_event.set()
        # 清空未播放的句子
        while True:
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except queue.Empty:
                break
        self.sentence_queue.put(_SENTINEL)
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=3)
        self._player_thread = None

    def _play_loop(self):
        """播放线程主循环。

        从队列取句子 → TTS合成 → pygame播放 → 取下一句。
        """
        while not self._stop_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if sentence is _SENTINEL:
                break

            filepath = None
            try:
                filepath = self.tts.synthesize(sentence)
                # 等上一句播完再播下一句，避免两句话叠在一起
                while (pygame.mixer.music.get_busy()
                       and not self._stop_event.is_set()):
                    pygame.time.wait(50)
                self._play_audio_file(filepath)
            except Exception as exc:
                print(f"[TTS] 播放失败：{exc}")
            finally:
                if filepath:
                    try:
                        Path(filepath).unlink(missing_ok=True)  # 临时文件及时清理
                    except OSError:
                        pass
                self.sentence_queue.task_done()

    def _play_audio_file(self, filepath: str):
        """播放单个音频文件。"""
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000)
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
            pygame.time.wait(50)

    # ------------------------------------------------------------------
    @property
    def is_playing(self) -> bool:
        """当前是否正在播放（供界面显示状态）。"""
        try:
            return bool(pygame.mixer.get_init() and pygame.mixer.music.get_busy())
        except Exception:
            return False

    def clear(self):
        """只清空待播放队列，不停止播放线程。"""
        while True:
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except queue.Empty:
                break
