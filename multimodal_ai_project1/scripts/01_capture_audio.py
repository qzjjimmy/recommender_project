"""阶段01 — 音频采集测试脚本。

验证 AudioCapture 模块：采集 5 秒音频并保存为 WAV 文件。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS, AUDIO_CONFIG
from src.audio_capture import AudioCapture


def main():
    """采集 5 秒音频并保存。"""
    print("测试音频采集...")

    output_path = Path(PATHS["outputs"]) / "test_audio.wav"
    duration = 5  # 秒
    total_chunks = int(duration * AUDIO_CONFIG["sample_rate"]
                       / AUDIO_CONFIG["chunk_size"])

    capture = AudioCapture()
    try:
        capture.start()
    except Exception as exc:
        print(f"✗ 打开麦克风失败：{exc}")
        print("  请插入麦克风后重试；本项不影响文本/知识库/视觉链路。")
        return

    print(f"开始采集 {duration} 秒，请对着麦克风说话…")
    chunks: list[bytes] = []
    try:
        for _ in range(total_chunks):
            chunks.append(capture.read_chunk())
    finally:
        capture.stop()

    audio_data = b"".join(chunks)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(AUDIO_CONFIG["channels"])
        wf.setsampwidth(AUDIO_CONFIG["sample_width"])
        wf.setframerate(AUDIO_CONFIG["sample_rate"])
        wf.writeframes(audio_data)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ 采集完成，写入 {len(audio_data)} 字节（{size_kb:.0f} KB）")
    print(f"音频已保存到: {output_path}")
    print(f"采样率: {AUDIO_CONFIG['sample_rate']}Hz, "
          f"时长: {duration}s, "
          f"总采样点: {duration * AUDIO_CONFIG['sample_rate']}")


if __name__ == "__main__":
    main()
