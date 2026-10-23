"""阶段01 — 视频采集测试脚本。

验证 VideoCapture 模块：采集 100 帧并保存首帧为 JPG 图片。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS, VIDEO_CONFIG
from src.video_capture import VideoCapture


def main():
    """采集 100 帧视频并保存首帧。"""
    print("测试视频采集...")

    output_path = Path(PATHS["outputs"]) / "test_frame.jpg"

    capture = VideoCapture()
    try:
        capture.start()
    except Exception as exc:
        print(f"✗ 打开摄像头失败：{exc}")
        print("  请检查系统摄像头授权；本项不影响语音/文本/知识库链路。")
        return

    first_frame = None
    count = 0
    try:
        for _ in range(100):
            frame = capture.read_frame()
            if frame is None:
                continue
            if first_frame is None:
                first_frame = frame
            count += 1
    finally:
        capture.stop()

    if first_frame is None:
        print("✗ 没有采集到有效画面。")
        return

    cv2.imwrite(str(output_path), first_frame)
    h, w = first_frame.shape[:2]
    print(f"✓ 共采集 {count} 帧，实际分辨率 {w}×{h}")
    print(f"首帧已保存到: {output_path}")
    print(f"分辨率: {VIDEO_CONFIG['width']}×{VIDEO_CONFIG['height']}")


if __name__ == "__main__":
    main()
