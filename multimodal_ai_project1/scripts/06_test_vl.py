"""阶段04 — 视觉语言模型测试脚本。

测试 Qwen2.5-VL-3B-Instruct（bfloat16）的视觉理解能力。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS, VL_CONFIG, resolve_vl_model  # noqa: E402


def main():
    """测试 VL 模型。"""
    print("=" * 50)
    print("视觉语言模型测试")
    print("=" * 50)

    from src.vl_model import VLModel

    if not VLModel.is_available():
        print("✗ 未找到可用的视觉语言模型。")
        print(f"  当前状态：{VLModel.availability_note()}")
        print("  请安装并启动 Ollama 后执行：ollama pull qwen2.5vl:3b")
        return
    print(f"模型来源: {VLModel.availability_note()}")
    print(f"帧数: {VL_CONFIG['num_frames']} | "
          f"dtype: {VL_CONFIG['dtype']} | "
          f"max_pixels: {VL_CONFIG['max_pixels']}")
    print("-" * 50)

    import csv

    from src.video_capture import VideoCapture

    # ---- 采集画面：优先摄像头，其次 outputs/ 里已有的测试图 ----
    frames: list = []
    source = ""
    if VideoCapture.enumerate_cameras():
        cap = VideoCapture()
        try:
            cap.start()
            source = "摄像头"
            for _ in range(8):
                frame = cap.read_frame()
                if frame is not None:
                    frames.append(frame)
        except Exception as exc:
            print(f"⚠ 摄像头不可用：{exc}")
        finally:
            cap.stop()
    if not frames:
        import cv2
        for img in sorted(Path(PATHS["outputs"]).glob("*.jpg")):
            data = cv2.imread(str(img))
            if data is not None:
                frames = [data] * 4
                source = f"图片 {img.name}"
                break
    if not frames:
        print("✗ 没有可用的画面输入（摄像头不可用且 outputs/ 下没有图片）。")
        return
    print(f"输入来源：{source}，共 {len(frames)} 帧")

    vl = VLModel()
    vl.load()
    rows: list[dict] = []
    for num_frames in (2, 4):
        question = "画面中有什么？简短描述一下。"
        import time
        t0 = time.time()
        answer = vl.generate_response(frames, question, num_frames=num_frames)
        cost = time.time() - t0
        print(f"\n[帧数 {num_frames}] {question}\n  → {answer}\n  耗时 {cost:.2f}s")
        rows.append({"frames": num_frames, "seconds": round(cost, 2),
                     "answer_chars": len(answer)})

    out_csv = Path(PATHS["outputs"]) / "vl_benchmark.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frames", "seconds", "answer_chars"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ 基准数据已写入 {out_csv}")

    leftovers = list(Path(PATHS["temp"]).glob("vl_frame_*.jpg"))
    print(f"✓ 临时图片残留 {len(leftovers)} 张（应为 0）")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
