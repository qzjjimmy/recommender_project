"""阶段04 — 模态路由器。

根据用户输入自动或手动切换文本模式和视觉模式。
"""

from __future__ import annotations


class ModalRouter:
    """模态路由器。

    决定用户查询应由文本 LLM 还是视觉 VL 模型处理。
    """

    VISION_KEYWORDS = [
        "看", "画面", "摄像头", "屏幕", "拍", "镜头",
        "图像", "照片", "图片", "视频",
        "look", "see", "camera", "view", "what", "who",
    ]

    def __init__(self):
        self.current_mode: str = "text"
        self.auto_mode: bool = True

    def route(self, user_input: str) -> str:
        """根据用户输入判断处理模式。

        Args:
            user_input: 用户输入的文本。

        Returns:
            "text" 或 "vision"。
        """
        if not self.auto_mode:
            return self.current_mode

        text = (user_input or "").strip()
        # 支持 /text、/vision 手工指令：既切模式也把指令本身吃掉
        lowered = text.lower()
        if lowered.startswith("/vision"):
            self.set_mode("vision")
            return "vision"
        if lowered.startswith("/text"):
            self.set_mode("text")
            return "text"

        hit = any(kw in text or kw in lowered for kw in self.VISION_KEYWORDS)
        self.current_mode = "vision" if hit else "text"
        return self.current_mode

    def set_mode(self, mode: str):
        """手动设置模式。

        Args:
            mode: "text" 或 "vision"。
        """
        if mode not in ("text", "vision"):
            raise ValueError("mode 只能是 'text' 或 'vision'")
        self.current_mode = mode
        self.auto_mode = False

    def strip_command(self, user_input: str) -> str:
        """去掉 /text、/vision 前缀，返回真正要问的内容。"""
        text = (user_input or "").strip()
        for cmd in ("/vision", "/text"):
            if text.lower().startswith(cmd):
                return text[len(cmd):].strip()
        return text

    def set_auto(self, auto: bool = True):
        """设置自动模式开关。

        Args:
            auto: True 启用自动路由，False 手动模式。
        """
        self.auto_mode = auto
