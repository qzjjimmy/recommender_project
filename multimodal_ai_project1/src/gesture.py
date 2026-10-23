"""阶段05 — 手势识别模块。

基于 MediaPipe Hands 关键点的几何关系判断 6 种手势。
"""

from __future__ import annotations

import time

from config.settings import GESTURE_CONFIG


class GestureRecognizer:
    """手势识别器。

    基于手部 21 个关键点的 y 坐标关系判断手指伸直/弯曲。
    """

    # 手指尖和 PIP 关节的索引 (MediaPipe Hands)
    FINGER_TIPS = [4, 8, 12, 16, 20]    # 拇指、食、中、无名、小指尖
    FINGER_PIPS = [2, 6, 10, 14, 18]    # 对应的 PIP 关节

    GESTURE_NAMES = [
        "thumbs_up",    # 点赞 👍
        "thumbs_down",  # 倒赞 👎
        "open_palm",    # 手掌张开 ✋
        "index_up",     # 食指向上 ☝️
        "index_down",   # 食指向下 👇
        "victory",      # 胜利 ✌️
    ]

    def __init__(self, cooldown_seconds: float | None = None):
        """初始化手势识别器。

        Args:
            cooldown_seconds: 同一手势的冷却时间 (秒)。
        """
        self.cooldown = cooldown_seconds or GESTURE_CONFIG["cooldown_seconds"]
        self.last_gesture: str | None = None
        self.last_trigger_time: float = 0.0

    def recognize(self, hand_landmarks) -> str | None:
        """识别手势。

        Args:
            hand_landmarks: MediaPipe Hands 检测结果 (.landmark 列表)。

        Returns:
            手势名称 (str) 或 None (未识别/冷却中)。
        """
        if hand_landmarks is None:
            return None
        # 兼容两种入参：MediaPipe 结果对象（有 .landmark）或关键点列表
        landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
        if landmarks is None or len(landmarks) < 21:
            return None

        detected: str | None = None
        # 顺序有讲究：先判"两指伸直"的手势，再判更宽松的手掌，
        # 否则 victory 会被 open_palm 抢走。
        if self._is_victory(landmarks):
            detected = "victory"
        elif self._is_open_palm(landmarks):
            detected = "open_palm"
        elif self._is_thumb_up(landmarks):
            detected = "thumbs_up"
        elif self._is_thumb_down(landmarks):
            detected = "thumbs_down"
        elif self._is_index_up(landmarks):
            detected = "index_up"
        elif self._is_index_down(landmarks):
            detected = "index_down"

        if detected is None:
            return None

        # 冷却：同一手势在冷却期内只触发一次，避免一帧一次刷屏
        if detected == self.last_gesture and self._is_within_cooldown():
            return None
        self.last_gesture = detected
        self.last_trigger_time = time.time()
        return detected

    def _is_finger_extended(self, landmarks, tip_idx: int,
                            pip_idx: int) -> bool:
        """判断单根手指是否伸直。

        tip.y < pip.y 表示伸直 (手朝上，指尖在上)。
        """
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    # ------------------------------------------------------------------
    # 辅助：拇指的"伸展 / 收拢"判别
    # ------------------------------------------------------------------
    # 为什么不能只看 y：拇指竖起来和拇指收拢在掌心时，指尖的 y 都可能高于
    # IP 关节，单看 y 无法区分——结果就是"👇 食指向下"被误判成"👎 倒赞"。
    # 真正的区别是**指尖离食指根（关键点5）的距离**：伸出去时离得远，
    # 收拢时会贴近食指根。
    THUMB_EXTENDED_MIN = 0.18
    THUMB_FOLDED_MAX = 0.16

    @staticmethod
    def _thumb_index_distance(landmarks) -> float:
        """拇指尖(4) 到食指根(5) 的归一化欧氏距离。"""
        tip = landmarks[4]
        mcp = landmarks[5]
        return ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5

    def _is_thumb_extended(self, landmarks) -> bool:
        """拇指是否向外伸展。"""
        return self._thumb_index_distance(landmarks) >= self.THUMB_EXTENDED_MIN

    def _is_thumb_folded(self, landmarks) -> bool:
        """拇指是否收拢在掌心。"""
        return self._thumb_index_distance(landmarks) <= self.THUMB_FOLDED_MAX

    def _four_fingers_curled(self, landmarks) -> bool:
        """食指/中指/无名指/小指是否全部弯曲。"""
        return all(
            not self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]))

    def _is_thumb_up(self, landmarks) -> bool:
        """判断大拇指向上。"""
        # 拇指伸展且朝上（指尖高于 IP 关节），其余四指弯曲
        return (self._is_thumb_extended(landmarks)
                and landmarks[4].y < landmarks[3].y
                and self._four_fingers_curled(landmarks))

    def _is_thumb_down(self, landmarks) -> bool:
        """判断大拇指向下。"""
        return (self._is_thumb_extended(landmarks)
                and landmarks[4].y > landmarks[3].y
                and self._four_fingers_curled(landmarks))

    def _is_open_palm(self, landmarks) -> bool:
        """判断手掌张开。"""
        # 四指伸直 + 拇指不在掌心（伸展开）
        if not all(self._is_finger_extended(landmarks, tip, pip)
                   for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:])):
            return False
        return not self._is_thumb_folded(landmarks)

    def _is_index_up(self, landmarks) -> bool:
        """判断食指向上。"""
        if not self._is_finger_extended(landmarks, self.FINGER_TIPS[1],
                                        self.FINGER_PIPS[1]):
            return False
        others_curled = all(
            not self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS[2:], self.FINGER_PIPS[2:]))
        return others_curled and self._is_thumb_folded(landmarks)

    def _is_index_down(self, landmarks) -> bool:
        """判断食指向下。"""
        if self._is_finger_extended(landmarks, self.FINGER_TIPS[1],
                                    self.FINGER_PIPS[1]):
            return False
        others_curled = all(
            not self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS[2:], self.FINGER_PIPS[2:]))
        return others_curled and self._is_thumb_folded(landmarks)

    def _is_victory(self, landmarks) -> bool:
        """判断胜利手势 (✌️)。"""
        index_ok = self._is_finger_extended(landmarks, self.FINGER_TIPS[1],
                                            self.FINGER_PIPS[1])
        middle_ok = self._is_finger_extended(landmarks, self.FINGER_TIPS[2],
                                             self.FINGER_PIPS[2])
        if not (index_ok and middle_ok):
            return False
        others_curled = all(
            not self._is_finger_extended(landmarks, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS[3:], self.FINGER_PIPS[3:]))
        return others_curled

    def _is_within_cooldown(self) -> bool:
        """检查是否在冷却时间内。"""
        return (time.time() - self.last_trigger_time) < self.cooldown

    def reset(self):
        """重置冷却状态（测试与手动打断时用）。"""
        self.last_gesture = None
        self.last_trigger_time = 0.0

    @classmethod
    def label_of(cls, gesture: str | None) -> str:
        """把英文手势名转成带 emoji 的中文标签。"""
        labels = {
            "thumbs_up": "👍 点赞",
            "thumbs_down": "👎 倒赞",
            "open_palm": "✋ 手掌",
            "index_up": "☝ 食指上",
            "index_down": "👇 食指向下",
            "victory": "✌ 胜利",
        }
        return labels.get(gesture or "", gesture or "未识别")
