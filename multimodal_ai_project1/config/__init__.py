"""全局配置模块。

所有模块从此读取配置，禁止在模块中硬编码路径/超参。
"""

from .settings import (
    PROJECT_ROOT,
    DEVICE,
    SEED,
    PATHS,
    MODEL_PATHS,
    MODEL_ASSETS,
    LANDMARKER_URLS,
    OLLAMA_CONFIG,
    AUDIO_CONFIG,
    VIDEO_CONFIG,
    VAD_CONFIG,
    ASR_CONFIG,
    LLM_CONFIG,
    LLM_GGUF_CONFIG,
    VL_CONFIG,
    TTS_CONFIG,
    GESTURE_CONFIG,
    KB_CONFIG,
    resolve_text_model,
    resolve_vl_model,
    set_seed,
)
