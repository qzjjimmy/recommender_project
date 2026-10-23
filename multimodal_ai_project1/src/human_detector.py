"""阶段05 — 人体姿态检测模块。

基于 MediaPipe 实现人体关键点检测、骨架绘制和姿态稳定性判断。

**接口说明（重要）**
课程脚手架原本按 ``mp.solutions.pose/hands/face_detection`` 书写，但 Apple
Silicon 上只能安装到 ``mediapipe>=0.10.30``，而这些版本**已经整体移除了
Solutions 接口**，只保留 Tasks 接口——照原样写会「import 即崩」。
因此这里改用官方现行的 Tasks 接口
（``mediapipe.tasks.python.vision`` 的 PoseLandmarker / HandLandmarker /
FaceLandmarker）实现**完全相同的能力**：

* Pose 33 个关键点、Hands 21 个关键点、人脸检测；
* 关键点绘制（Tasks 版不再提供 drawing_utils，这里用 OpenCV 手动画）；
* ``is_full_body_visible`` 用肩/髋/膝的 visibility 判断是否全身入镜；
* ``is_pose_stable`` 用最近若干帧关键点的平均差异判断姿态是否稳定。

类名与所有方法签名与脚手架保持一致，上层代码无需改动。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# MediaPipe 退出时会尝试上传遥测（clearcut），失败后每 60 秒重试一次，
# 会把脚本退出拖住好几分钟；这里只保留 error 级日志，减少无关输出。
os.environ.setdefault("GLOG_minloglevel", "2")

import mediapipe as mp

from config.settings import GESTURE_CONFIG, MODEL_ASSETS

# Pose 骨架连线（33 点）
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Hands 连线（21 点）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HumanDetector:
    """人体姿态检测器。

    集成 MediaPipe Pose、Hands、Face 三个模型（Tasks 接口）。
    """

    # 全身完整性检查的关键点：左/右肩(11,12)、左/右髋(23,24)、左/右膝(25,26)
    FULL_BODY_LANDMARKS = [11, 12, 23, 24, 25, 26]

    def __init__(self,
                 pose_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """初始化人体检测器。

        Args:
            pose_complexity: Pose 模型复杂度 (0/1/2)。Tasks 版官方只提供
                full 模型，此参数保留以兼容原签名，不再影响加载。
            min_detection_confidence: 检测置信度阈值。
            min_tracking_confidence: 追踪置信度阈值。
        """
        missing = [k for k, p in MODEL_ASSETS.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "缺少 MediaPipe 模型包：" + "、".join(missing)
                + "。请把这些 .task 文件放到 models/mediapipe/ 下"
                  "（README 里有下载地址）。")

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp_python = mp_python
        self._vision = vision
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_ASSETS["pose"])),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_ASSETS["hands"])),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ))
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_ASSETS["face"])),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=min_detection_confidence,
            ))

        self.history: list = []  # 姿态历史帧
        self.MAX_HISTORY = GESTURE_CONFIG.get("history_size", 10)

    # ------------------------------------------------------------------
    @staticmethod
    def is_available() -> bool:
        """三个 MediaPipe 模型包是否都已就位。"""
        return all(Path(p).exists() for p in MODEL_ASSETS.values())

    def process_frame(self, bgr_image: np.ndarray) -> dict:
        """处理一帧图像，返回所有检测结果。

        Args:
            bgr_image: BGR 格式图像 (H, W, 3)。

        Returns:
            dict: {"pose": ..., "hands": ..., "face": ...}
        """
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return {
            "pose": self.pose.detect(mp_image),
            "hands": self.hands.detect(mp_image),
            "face": self.face.detect(mp_image),
        }

    def primary_hand(self, results: dict):
        """取"主手"关键点列表：优先右手（多数人惯用手），否则取第一只手。"""
        hands = results.get("hands")
        landmarks = getattr(hands, "hand_landmarks", None) or []
        if not landmarks:
            return None
        handedness = getattr(hands, "handedness", None) or []
        for i, cats in enumerate(handedness):
            if cats and getattr(cats[0], "category_name", "") == "Right":
                return landmarks[i]
        return landmarks[0]

    def draw_landmarks(self, image: np.ndarray,
                       results: dict) -> np.ndarray:
        """在图像上绘制检测关键点和连线。

        Args:
            image: 原始图像。
            results: process_frame 的返回结果。

        Returns:
            叠加了标注的图像（新数组，不修改入参）。
        """
        import cv2

        canvas = image.copy()
        h, w = canvas.shape[:2]

        pose = results.get("pose")
        for landmarks in (getattr(pose, "pose_landmarks", None) or []):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
            for p in pts:
                cv2.circle(canvas, p, 3, (0, 120, 255), -1)

        hands = results.get("hands")
        for landmarks in (getattr(hands, "hand_landmarks", None) or []):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(canvas, pts[a], pts[b], (255, 80, 80), 2)
            for p in pts:
                cv2.circle(canvas, p, 3, (255, 255, 255), -1)

        face = results.get("face")
        for landmarks in (getattr(face, "face_landmarks", None) or []):
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            cv2.rectangle(canvas, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), (0, 215, 255), 2)
        return canvas

    def is_full_body_visible(self, pose_landmarks) -> bool:
        """判断用户是否全身完整入镜。

        Args:
            pose_landmarks: Pose 关键点列表。

        Returns:
            6 个关键点 visibility 均 ≥ 0.5 → True。
        """
        if not pose_landmarks:
            return False
        try:
            return all(
                float(getattr(pose_landmarks[i], "visibility", 1.0) or 0.0) >= 0.5
                for i in self.FULL_BODY_LANDMARKS)
        except (IndexError, TypeError):
            return False

    def is_pose_stable(self, pose_landmarks) -> bool:
        """判断用户姿态是否稳定。

        计算当前帧关键点与历史平均的差异。

        Args:
            pose_landmarks: 当前帧 Pose 关键点。

        Returns:
            差异 < 阈值 → True。
        """
        if not pose_landmarks:
            return False
        current = np.array([(lm.x, lm.y) for lm in pose_landmarks], dtype="float32")
        threshold = GESTURE_CONFIG.get("stability_threshold", 0.1)
        stable = False
        if len(self.history) >= 3:
            mean = np.mean(np.stack(self.history[-self.MAX_HISTORY:]), axis=0)
            diff = float(np.mean(np.abs(current - mean)))
            stable = diff < threshold
        self.history.append(current)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
        return stable

    def close(self):
        """释放 MediaPipe 资源。"""
        for model in (self.pose, self.hands, self.face):
            try:
                model.close()
            except Exception:
                pass
