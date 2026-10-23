"""阶段02 — ASR 端到端测试脚本。

测试语音采集→VAD断句→ASR识别的完整流水线。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "SenseVoiceSmall"


def main():
    """测试语音识别链路。"""
    print("=" * 50)
    print("ASR 端到端测试")
    print("=" * 50)

    from src.asr import ASREngine
    from src.audio_capture import AudioCapture
    from src.audio_thread import AudioStreamThread
    from src.vad import VADProcessor

    if not MODEL_DIR.exists():
        print(f"✗ 未找到 ASR 模型：{MODEL_DIR}")
        print("  请把 SenseVoiceSmall 下载到 models/SenseVoiceSmall/ 后重试。")
        return

    print(f"加载 SenseVoiceSmall（设备 {ASREngine(str(MODEL_DIR)).device}）…")
    t0 = time.time()
    asr = ASREngine(str(MODEL_DIR)).load()
    print(f"✓ 模型加载完成，用时 {time.time() - t0:.1f}s")

    # ---- 离线测试：对 outputs 下已有的 wav 逐条识别，得到准确耗时 ----
    wavs = sorted(Path(PATHS["outputs"]).glob("*.wav"))
    if wavs:
        print("\n--- 离线文件识别 ---")
        for wav in wavs:
            start = time.time()
            text = asr.transcribe(str(wav))
            print(f"  {wav.name:26s} {time.time() - start:5.2f}s  "
                  f"{text or '（未识别到内容）'}")
    else:
        print("\n（outputs/ 下暂无 wav，跳过离线测试；"
              "可先运行 scripts/01_capture_audio.py）")

    # ---- 在线测试：麦克风 → VAD 断句 → ASR ----
    if not AudioCapture.list_input_devices():
        print("\n✗ 没有检测到麦克风，跳过实时链路测试。")
        print("\n测试完成。")
        return

    print("\n--- 实时链路测试（说几句话，30 秒后自动结束）---")
    results: list[tuple[str, float]] = []
    last_at = {"t": time.time()}

    def on_result(text: str) -> None:
        now = time.time()
        gap = now - last_at["t"]
        last_at["t"] = now
        results.append((text, gap))
        print(f"  [识别] {text}   （距上一句 {gap:.2f}s）")

    capture = AudioCapture()
    vad = VADProcessor()
    thread = AudioStreamThread(capture, vad, asr, on_result=on_result)
    thread.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and len(results) < 5:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n（手动中断）")
    finally:
        thread.stop()

    print(f"\n共识别 {len(results)} 条：")
    for text, gap in results:
        print(f"  · {text}（{gap:.2f}s）")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
