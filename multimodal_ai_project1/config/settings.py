"""全局配置文件。

集中管理项目所有路径、超参、设备和参数。
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 随机种子（贯穿全项目，确保可复现）
# ============================================================
SEED = 42


def set_seed(seed: int = SEED):
    """设置全局随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 设备
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 路径常量
# ============================================================
PATHS = {
    "data":       PROJECT_ROOT / "data",
    "models":     PROJECT_ROOT / "models",
    "outputs":    PROJECT_ROOT / "outputs",
    "temp":       PROJECT_ROOT / "outputs" / "temp",
    "snapshots":  PROJECT_ROOT / "outputs" / "snapshots",
    "scripts":    PROJECT_ROOT / "scripts",
}

# ============================================================
# 模型路径 — 请根据实际下载位置修改
# ============================================================
# 自适应定位：优先使用项目内 models/ 下的模型（整目录拷贝到新电脑时无需改配置），
# 项目内不存在时由 resolve_* 函数按优先级回退。
#
# 当前模型基线（2026-09 更新）：
#   语音识别  SenseVoiceSmall            ~0.9 GB
#   文本大模型 Qwen3-4B-Instruct-2507     GGUF int4 ~2.4 GB（llama.cpp 推理）
#   视觉理解  Qwen2.5-VL-3B-Instruct     bfloat16 ~7.2 GB
#   知识库嵌入 all-MiniLM-L6-v2          ~0.09 GB
#
# 注意：文本/视觉各保留一个"备选模型"键，主模型缺失时自动回退，
# 便于在显存/内存不足的机器上降级运行。
_EMBEDDING_LOCAL = PATHS["models"] / "all-MiniLM-L6-v2"

MODEL_PATHS = {
    # ---- 语音识别 (ASR) ----
    "asr": PATHS["models"] / "SenseVoiceSmall",

    # ---- 文本大模型 ----
    # 优先：GGUF 量化版（llama.cpp，model_path 是 .gguf **文件**，不是目录）
    "text_gguf": PATHS["models"] / "gguf" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    # 回退：transformers 原生版（model_path 是**目录**）
    "text": PATHS["models"] / "Qwen2-1.5B-Instruct",

    # ---- 视觉语言模型 ----
    # 优先：Qwen2.5-VL-3B
    "vl": PATHS["models"] / "Qwen2.5-VL-3B-Instruct",
    # 回退：Qwen2-VL-2B（显存/内存不足时可用）
    "vl_legacy": PATHS["models"] / "Qwen2-VL-2B-Instruct",

    # ---- 知识库嵌入模型 ----
    # 本地目录存在则用本地，否则用模型名让 sentence-transformers 自动下载
    "embedding": str(_EMBEDDING_LOCAL) if _EMBEDDING_LOCAL.exists()
                 else "all-MiniLM-L6-v2",
}


def resolve_text_model() -> tuple[str, str]:
    """按优先级挑选可用的文本大模型。

    Returns:
        (模型路径, 后端类型)。后端为 "gguf" / "transformers" / "none"。
        GGUF 路径是**文件**，transformers 路径是**目录**——两者语义不同，
        这是本模块最容易踩的坑。
    """
    if MODEL_PATHS["text_gguf"].exists():
        return str(MODEL_PATHS["text_gguf"]), "gguf"
    if MODEL_PATHS["text"].exists():
        return str(MODEL_PATHS["text"]), "transformers"
    return "", "none"


def resolve_vl_model() -> str:
    """按优先级挑选可用的视觉语言模型目录。

    Returns:
        模型目录路径；都缺失时返回空字符串（调用方应据此跳过 VL 功能）。
    """
    for key in ("vl", "vl_legacy"):
        if MODEL_PATHS[key].exists():
            return str(MODEL_PATHS[key])
    return ""

# ============================================================
# Ollama 配置（本项目默认的文本 / 视觉模型来源）
# ------------------------------------------------------------
# 为什么用 Ollama：课程要求"本地大模型"，但 Apple Silicon 上从零编译
# llama-cpp-python / 下载 7GB 的 VL 权重都很折腾。Ollama 用一条
# `ollama pull` 就能拿到量化好的 Qwen 模型，并对外提供 OpenAI 兼容接口，
# 代码里只需发 HTTP 请求即可，不必关心量化与显存。
#
# 需要你先做的事：
#     ollama pull qwen3:4b          # 文本对话
#     ollama pull qwen2.5vl:3b      # 视觉理解（支持图片输入）
#
# 若 Ollama 未启动，程序不会崩溃：相关链路会给出明确提示并降级。
# ============================================================
OLLAMA_CONFIG = {
    # 服务地址；若设置了环境变量 OLLAMA_HOST（Ollama 官方约定）则优先用它
    "base_url": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "text_model": "qwen3:4b",      # 文本对话模型
    # 文本模型候选（按优先级自动挑第一个已 pull 的）。
    # 为什么把 qwen2.5:3b 放前面：qwen3 是**推理模型**，实测在知识库问答场景
    # 会先"思考"4500+ 字，把 token 预算吃光（content 为空、单轮耗时约 100s）；
    # qwen2.5:3b 是非推理模型，回答干净且快得多。
    # 你只需 `ollama pull qwen2.5:3b`，程序会自动改用它，无需改代码。
    "text_model_candidates": ["qwen2.5:3b", "qwen3:4b"],
    "vl_model":   "qwen2.5vl:3b",  # 视觉理解模型
    "timeout":    180,             # 单次请求超时（秒）；4B 模型 CPU 推理较慢
    "keep_alive": "10m",           # 模型在显存/内存里的驻留时间
    # 推理模型（qwen3 等）的思考开关。**必须保持 True**，实测结论：
    #   ① think=True  → 思考内容走 message.thinking（我们不展示），
    #                   message.content 是干净的最终回答（实测 49 字、19 秒）；
    #   ② think=False → 该模型的模板仍强制思考，推理过程会直接混进 content，
    #                   输出是"首先，用户要求我…"这类推演，不可用。
    # 关掉思考要想真的生效，得换成非推理模型（如 qwen2.5:3b）。
    "enable_thinking": True,
    # 推理会额外消耗 token。预算太小（如默认 512）会导致思考没结束、
    # content 为空；这里给推理模型兜一个下限。
    "thinking_max_tokens": 2048,
    # 单次请求的上下文窗口。Ollama 默认只有 4096，多帧图片很容易撑爆
    # （实测 4 帧 640×480 图片约 8400 tokens，会报 exceed_context_size_error）。
    "num_ctx": 8192,
}

# ============================================================
# MediaPipe 模型包（.task 文件）
# ------------------------------------------------------------
# 注意：Apple Silicon 上只能装 mediapipe>=0.10.30，而这些版本**已经移除了
# 旧的 mp.solutions 接口**，只保留 Tasks 接口。Tasks 接口需要独立的 .task
# 模型文件，故单独下载到这里（Pose/Hands/Face 各一个）。
# ============================================================
MODEL_ASSETS = {
    "pose":  PATHS["models"] / "mediapipe" / "pose_landmarker_full.task",
    "hands": PATHS["models"] / "mediapipe" / "hand_landmarker.task",
    "face":  PATHS["models"] / "mediapipe" / "face_landmarker.task",
}

# 官方下载地址（Google 存储，实测可达）
LANDMARKER_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hands": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
}

# ============================================================
# 音频配置
# ============================================================
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size":  1024,
    "channels":    1,
    "sample_width": 2,  # 16-bit = 2 bytes
}

# ============================================================
# 视频配置
# ============================================================
VIDEO_CONFIG = {
    "camera_id": 0,
    "width":     640,
    "height":    480,
    "fps":       30,
}

# ============================================================
# VAD (语音活动检测) 配置
# ============================================================
VAD_CONFIG = {
    "aggressiveness":      2,     # 0-3, 3 最不敏感；2 为平衡值，避免说话中停顿被误判为句尾
    "frame_duration_ms":   30,    # VAD 帧时长（必须是 10/20/30）
    "silence_duration_ms": 600,   # 连续静音触发断句
    "min_speech_duration_ms": 500,  # 最短有效语音（过滤环境噪音误触发）
    "pre_roll_ms":         240,   # 语音段前置缓冲，弥补 VAD 检测延迟，防止开头截断
    "tail_padding_ms":     120,   # 语音段尾部保留静音，避免尾音丢失
}

# ============================================================
# ASR (语音识别) 配置
# ============================================================
ASR_CONFIG = {
    "language": "zh",     # 固定中文；"auto" 在短句上会误判语种（如"OK""好的"来回切）
    "use_itn":  True,     # 逆文本正则化："二零二六年"→"2026年"，下游 LLM 更易理解数字
}

# ============================================================
# LLM (文本大模型) 配置
# ============================================================
LLM_CONFIG = {
    "max_new_tokens": 512,
    "temperature":    0.7,
    "top_p":          0.9,
    "do_sample":      True,
}

# GGUF 后端专用（llama.cpp）——仅当 resolve_text_model() 返回 "gguf" 时生效
LLM_GGUF_CONFIG = {
    "n_ctx":     4096,   # 上下文窗口
    "n_threads": None,   # None = 自动使用全部 CPU 核心
    "verbose":   False,
}

# ============================================================
# VL (视觉语言模型) 配置
# ============================================================
VL_CONFIG = {
    # 必须用 bfloat16。实测 fp32 + int8 动态量化时，量化过程需同时持有
    # fp32(8.8GB) 与 int8(2.2GB) 两份模型，叠加 ASR+LLM 后峰值 23.4 GB，
    # 在 32 GB 机器上直接触发 std::bad_alloc。改 bf16 后峰值降至 6.2 GB，
    # 加载 26.0s → 1.1s，推理耗时基本持平。
    "dtype":         "bfloat16",
    "num_frames":    2,               # 单次问答使用的帧数（2 帧即可看清主体，prefill 减半）
    "max_pixels":    256 * 28 * 28,   # 每帧最大像素数，限制视觉 token 数加速 prefill
    "max_new_tokens": 128,            # 视觉回答限 50 字左右，128 足够
    "lazy_load":     True,            # 懒加载：首次"看场景"时才加载，避免拖慢冷启动
    # 送进模型前把帧缩到这个最大边长，显著减少视觉 token
    # （缩小后 4 帧也能稳稳落在上下文窗口内）
    "max_side":      512,
}

# ============================================================
# TTS (语音合成) 配置
# ============================================================
TTS_CONFIG = {
    "zh_voice": "zh-CN-XiaoxiaoNeural",
    "en_voice": "en-US-JennyNeural",
    "output_dir": str(PATHS["temp"]),
}

# ============================================================
# 手势识别配置
# ============================================================
GESTURE_CONFIG = {
    "cooldown_seconds": 2.0,  # 同一手势冷却时间
    "stability_threshold": 0.1,  # 姿态稳定性阈值
    "history_size": 10,  # 姿态历史帧数
}

# ============================================================
# 知识库配置
# ============================================================
KB_CONFIG = {
    "chunk_size":    500,   # 文本分块大小
    "chunk_overlap": 50,    # 块间重叠
    "search_top_k":  3,     # 检索返回数
}

# ============================================================
# 启动时自动创建所需目录
# ============================================================
for _key, _path in PATHS.items():
    _path.mkdir(parents=True, exist_ok=True)
