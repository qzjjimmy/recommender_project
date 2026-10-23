"""阶段05 — 手势动作映射模块。

将识别到的手势映射为具体的系统控制指令。
"""

from __future__ import annotations


class GestureActionMapper:
    """手势动作映射器。

    维护手势→动作的映射表，提供映射查询和执行接口。
    """

    # 默认手势→动作映射
    DEFAULT_ACTION_MAP: dict[str, str] = {
        "thumbs_up":    "volume_up",
        "thumbs_down":  "volume_down",
        "open_palm":    "stop",
        "index_up":     "start_listening",
        "index_down":   "mute",
        "victory":      "take_snapshot",
    }

    def __init__(self, action_map: dict[str, str] | None = None):
        """初始化映射器。

        Args:
            action_map: 自定义手势→动作映射表 (默认使用 DEFAULT_ACTION_MAP)。
        """
        self.action_map = action_map or self.DEFAULT_ACTION_MAP.copy()

    def map(self, gesture: str) -> str:
        """将手势名称映射为动作名称。

        Args:
            gesture: 手势名称 (如 "thumbs_up")。

        Returns:
            动作名称 (如 "volume_up")，未匹配返回 "unknown"。
        """
        return self.action_map.get(gesture, "unknown")

    def set_mapping(self, gesture: str, action: str):
        """设置自定义手势映射。

        Args:
            gesture: 手势名称。
            action: 动作名称。
        """
        self.action_map[gesture] = action

    def reset_mapping(self):
        """重置为默认映射表。"""
        self.action_map = self.DEFAULT_ACTION_MAP.copy()

    def get_all_mappings(self) -> dict[str, str]:
        """获取当前全部手势映射。"""
        return self.action_map.copy()

    # ------------------------------------------------------------------
    ACTION_LABELS: dict[str, str] = {
        "volume_up":       "增大音量",
        "volume_down":     "减小音量",
        "stop":            "停止当前播放/生成",
        "start_listening": "开始录音/提问",
        "mute":            "暂停/静音",
        "take_snapshot":   "拍照保存当前画面",
        "unknown":         "未定义动作",
    }

    def describe(self, action: str) -> str:
        """动作的中文说明（界面显示用）。"""
        return self.ACTION_LABELS.get(action, action)

    def execute(self, action: str, handlers: dict | None = None, **kwargs) -> str:
        """执行动作。

        真正的执行依赖界面（音量、拍照、停止播放都是界面/系统层面的操作），
        所以这里**由调用方传入 handlers**（一个 {动作名: 可调用} 的字典），
        传了就跑，没传就只返回一句说明——这样本模块不依赖 PyQt，
        也便于单元测试。

        Args:
            action: 动作名（如 "take_snapshot"）。
            handlers: {动作名: 回调}；回调可以接收 **kwargs。
            **kwargs: 透传给回调的上下文（如 frame=当前帧）。

        Returns:
            一句可显示的结果说明。
        """
        if action == "unknown":
            return "未定义的手势动作"
        if handlers and action in handlers:
            try:
                result = handlers[action](**kwargs)
                return result if isinstance(result, str) else self.describe(action)
            except Exception as exc:
                return f"执行「{self.describe(action)}」失败：{exc}"
        return f"已请求：{self.describe(action)}"
