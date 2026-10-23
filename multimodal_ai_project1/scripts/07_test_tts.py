"""阶段04 — TTS 语音合成测试脚本。

测试 edge-tts 的中英文合成和播放效果。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402


def main():
    """测试 TTS 合成与播放。"""
    print("=" * 50)
    print("TTS 语音合成测试")
    print("=" * 50)

    from src.tts import TTSEngine
    from src.tts_player import StreamingTTSPlayer

    engine = TTSEngine()
    out_dir = Path(PATHS["outputs"])

    # ---- 单句合成（中 / 英）----
    for name, text in (("tts_test_zh.mp3", "你好，我是多模态AI助手。"),
                       ("tts_test_en.mp3", "Hello, I am your AI assistant.")):
        lang = engine.detect_language(text)
        voice = engine.zh_voice if lang == "zh" else engine.en_voice
        print(f"[合成] {name}\n  文本：{text}\n  语种：{lang}  音色：{voice}")
        t0 = time.time()
        try:
            path = engine.synthesize(text, str(out_dir / name))
        except Exception as exc:
            print(f"  ✗ 合成失败（edge-tts 需要联网）：{exc}")
            return
        size_kb = Path(path).stat().st_size / 1024
        print(f"  ✓ 已保存 {path}（{size_kb:.0f} KB，{time.time() - t0:.2f}s）")

    # ---- 分句 + 流式播放：验证"首句延迟" ----
    long_text = ("第一句，先给结论。第二句，补充原因；第三句，给出建议！"
                 "This is the fourth sentence.")
    sentences = engine.split_sentences(long_text)
    print(f"\n[分句] 共 {len(sentences)} 句：")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")

    player = StreamingTTSPlayer(engine)
    player.start()
    print("\n[流式播放] 开始播报长文本（首句合成完成即出声）…")
    t0 = time.time()
    player.enqueue(long_text)
    first_sentence_cost = time.time() - t0
    print(f"  入队耗时 {first_sentence_cost:.3f}s（分句本身几乎不耗时，"
          "真正延迟在首句合成）")
    deadline = time.time() + 30
    while not player.sentence_queue.empty() and time.time() < deadline:
        time.sleep(0.2)
    player.stop()
    print("  ✓ 播放结束，播放线程已停止")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
