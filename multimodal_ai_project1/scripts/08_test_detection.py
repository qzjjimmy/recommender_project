"""阶段05 — 人体检测与手势识别测试脚本。

测试 MediaPipe 骨架检测和 6 种手势识别。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PATHS  # noqa: E402


class _LM(NamedTuple):
    """极简关键点：手势识别只用到 x / y（与 MediaPipe 的字段同名）。"""

    x: float
    y: float
    z: float = 0.0


def make_hand(extended: set[str], thumb: str) -> list[_LM]:
    """构造一只手的 21 个关键点（用于不依赖摄像头的单元验证）。

    Args:
        extended: 伸直的手指名集合（index/middle/ring/pinky 的子集）。
        thumb: 拇指状态：``up`` 伸直朝上 / ``down`` 伸直朝下 / ``curl`` 收拢在掌心。
    """
    xs = {"index": 0.42, "middle": 0.50, "ring": 0.58, "pinky": 0.66}
    pts: list[_LM] = [_LM(0.5, 0.85)]  # 0 手腕

    # 拇指 1-4：伸出去时指尖离食指根远，收拢时贴近食指根 (0.42, 0.62)
    if thumb == "up":
        thumb_pts = [(0.32, 0.70), (0.30, 0.58), (0.28, 0.46), (0.26, 0.34)]
    elif thumb == "down":
        thumb_pts = [(0.32, 0.70), (0.30, 0.76), (0.28, 0.82), (0.26, 0.88)]
    else:
        thumb_pts = [(0.34, 0.70), (0.38, 0.68), (0.41, 0.65), (0.43, 0.62)]
    pts += [_LM(x, y) for x, y in thumb_pts]

    # 其余四指：MCP / PIP / DIP / TIP
    for name in ("index", "middle", "ring", "pinky"):
        x = xs[name]
        if name in extended:
            ys = [0.62, 0.50, 0.42, 0.34]   # 指尖高于 PIP → 伸直
        else:
            ys = [0.62, 0.55, 0.62, 0.70]   # 指尖低于 PIP → 弯曲
        pts += [_LM(x, y) for y in ys]
    return pts


def _index_down_hand() -> list[_LM]:
    """单独构造"食指向下"：指尖低于 PIP，其余三指弯曲、拇指收拢。"""
    pts = make_hand({"index"}, "curl")
    pts[5 + 0] = _LM(0.42, 0.62)   # MCP
    pts[5 + 1] = _LM(0.42, 0.55)   # PIP
    pts[5 + 2] = _LM(0.42, 0.62)   # DIP
    pts[5 + 3] = _LM(0.42, 0.72)   # TIP 低于 PIP
    return pts


GESTURE_CASES = {
    "thumbs_up": make_hand(set(), "up"),
    "thumbs_down": make_hand(set(), "down"),
    "open_palm": make_hand({"index", "middle", "ring", "pinky"}, "up"),
    "index_up": make_hand({"index"}, "curl"),
    "index_down": _index_down_hand(),
    "victory": make_hand({"index", "middle"}, "curl"),
}


def unit_test() -> float:
    """用合成关键点验证 6 种手势，返回准确率。"""
    from src.gesture import GestureRecognizer

    recognizer = GestureRecognizer(cooldown_seconds=0.0)
    print("\n--- 合成关键点单元验证 ---")
    correct = 0
    for expected, pts in GESTURE_CASES.items():
        recognizer.reset()
        got = recognizer.recognize(pts)
        ok = got == expected
        correct += int(ok)
        print(f"[{'✔' if ok else '✖'}] 期望 {expected:12s} 实际 {got}")
    acc = correct / len(GESTURE_CASES)
    print(f"合成手势准确率：{correct}/{len(GESTURE_CASES)} = {acc:.0%}")
    return acc


def live_test(max_frames: int = 400) -> int:
    """打开摄像头实时显示骨架与手势识别结果（按 q 退出）。"""
    import cv2

    from src.gesture import GestureRecognizer
    from src.gesture_actions import GestureActionMapper
    from src.human_detector import HumanDetector
    from src.video_capture import VideoCapture

    if not HumanDetector.is_available():
        print("⚠ MediaPipe 模型包缺失，跳过实时检测（models/mediapipe/*.task）")
        return 0
    if not VideoCapture.enumerate_cameras():
        print("⚠ 没有检测到摄像头，跳过实时检测。")
        return 0

    detector = HumanDetector()
    recognizer = GestureRecognizer()
    mapper = GestureActionMapper()
    cap = VideoCapture()
    try:
        cap.start()
    except Exception as exc:
        print(f"⚠ 无法打开摄像头：{exc}")
        detector.close()
        return 0

    print("\n--- 实时检测（做出 6 种手势验证，按 q 退出）---")
    count = 0
    saved: set[str] = set()
    try:
        while count < max_frames:
            frame = cap.read_frame()
            if frame is None:
                continue
            results = detector.process_frame(frame)
            overlay = detector.draw_landmarks(frame, results)
            hand = detector.primary_hand(results)
            if hand is not None:
                gesture = recognizer.recognize(hand)
                if gesture:
                    label = GestureRecognizer.label_of(gesture)
                    print(f"  检测到 {label} → {mapper.describe(mapper.map(gesture))}")
                    if gesture not in saved:
                        out = Path(PATHS["outputs"]) / f"gesture_test_{gesture}.jpg"
                        cv2.imwrite(str(out), overlay)
                        saved.add(gesture)
            cv2.imshow("detection (press q to quit)", overlay)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.stop()
        detector.close()
        cv2.destroyAllWindows()
    print(f"共处理 {count} 帧，保存了 {len(saved)} 种手势截图到 outputs/")
    return count


def main():
    """测试人体检测与手势识别。"""
    print("=" * 50)
    print("人体检测与手势识别测试")
    print("=" * 50)

    unit_test()

    # 冷却机制验证：同一手势连续两帧只应触发一次
    from src.gesture import GestureRecognizer

    recognizer = GestureRecognizer(cooldown_seconds=2.0)
    first = recognizer.recognize(GESTURE_CASES["thumbs_up"])
    second = recognizer.recognize(GESTURE_CASES["thumbs_up"])
    print(f"\n--- 冷却机制 ---\n  第一次：{first}\n  冷却期内第二次：{second}"
          f"  {'✓ 正常' if first and second is None else '✖ 异常'}")

    live_test()

    print("\n测试完成。")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # MediaPipe 的遥测线程是非守护线程，失败重试会把退出拖住几分钟；
    # 检测已经跑完，这里直接结束进程。
    os._exit(0)
