"""阶段01 — 摄像头视频采集模块。

使用 OpenCV 从系统摄像头实时采集视频帧。
"""

from __future__ import annotations

import cv2
import numpy as np
from config.settings import VIDEO_CONFIG


class VideoCapture:
    """实时视频采集器。

    封装 OpenCV VideoCapture，提供统一的 start/read_frame/stop 接口。
    """

    def __init__(self, camera_id: int | None = None,
                 width: int | None = None,
                 height: int | None = None):
        """初始化视频采集器。

        Args:
            camera_id: 摄像头设备ID。默认使用 VIDEO_CONFIG。
            width: 画面宽度。默认使用 VIDEO_CONFIG。
            height: 画面高度。默认使用 VIDEO_CONFIG。
        """
        self.camera_id = camera_id or VIDEO_CONFIG["camera_id"]
        self.width = width or VIDEO_CONFIG["width"]
        self.height = height or VIDEO_CONFIG["height"]

        self.cap: cv2.VideoCapture | None = None

    def start(self):
        """打开摄像头设备。"""
        if self.cap is not None:
            return
        # macOS 首次访问摄像头会弹系统授权框；未授权时 cv2.VideoCapture
        # 可能一直阻塞，所以放到子线程里计时，超时就给出明确提示。
        result: dict = {}

        def _open() -> None:
            cap = cv2.VideoCapture(self.camera_id)
            result["cap"] = cap
            if cap is None or not cap.isOpened():
                result["error"] = (
                    f"无法打开摄像头 {self.camera_id}。请确认：① 摄像头已连接；"
                    "② 未被其它程序占用；③ 已授权（macOS："
                    "设置 → 隐私与安全性 → 摄像头）。")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, _ = cap.read()
            if not ok:
                result["error"] = f"摄像头 {self.camera_id} 打开了但读不到画面。"

        import threading

        worker = threading.Thread(target=_open, daemon=True)
        worker.start()
        worker.join(10.0)
        if worker.is_alive():
            raise RuntimeError(
                f"打开摄像头 {self.camera_id} 超时（10s）。通常是系统尚未授权，"
                "请在「设置 → 隐私与安全性 → 摄像头」允许终端 / PyCharm 访问。")
        if "error" in result:
            cap = result.get("cap")
            if cap is not None:
                cap.release()
            raise RuntimeError(result["error"])
        self.cap = result["cap"]

    def read_frame(self) -> np.ndarray | None:
        """读取一帧画面。

        Returns:
            BGR 格式的 numpy 数组 (H, W, 3)；若读取失败返回 None。
        """
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        """释放摄像头设备。"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def enumerate_cameras(max_probe: int = 3, timeout: float = 4.0) -> list[int]:
        """探测可用的摄像头编号（带超时，避免未授权时卡死）。"""
        import threading

        found: list[int] = []
        for cid in range(max_probe):
            res: dict = {}

            def _probe(cid: int = cid) -> None:
                cap = cv2.VideoCapture(cid)
                try:
                    if cap is None or not cap.isOpened():
                        res["ok"] = False
                        return
                    ok, _ = cap.read()
                    res["ok"] = bool(ok)
                except Exception:
                    res["ok"] = False
                finally:
                    if cap is not None:
                        cap.release()

            t = threading.Thread(target=_probe, daemon=True)
            t.start()
            t.join(timeout)
            if res.get("ok"):
                found.append(cid)
        return found

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
