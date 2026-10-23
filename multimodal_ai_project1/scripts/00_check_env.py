"""阶段01 — 环境自检脚本。

检查所有依赖、模型和硬件设备是否就绪。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    PROJECT_ROOT, PATHS, MODEL_PATHS, DEVICE,
    OLLAMA_CONFIG, resolve_text_model, resolve_vl_model,
)


def check_python_version() -> bool:
    """检查 Python 版本 ≥ 3.10。"""
    version = sys.version_info
    ok = version >= (3, 10)
    print(f"[{'✓' if ok else '✗'}] Python 版本: {version.major}.{version.minor}.{version.micro}")
    return ok


def check_dependencies() -> bool:
    """检查关键依赖是否可导入。"""
    deps = [
        # 基础框架
        "torch", "transformers",
        # 语音
        "funasr", "webrtcvad", "edge_tts", "langid",
        # 文本大模型（GGUF 后端；本机默认走 Ollama，故不作硬性要求）
        "llama_cpp",
        # 视觉
        "cv2", "mediapipe", "qwen_vl_utils",
        # 知识库
        "sentence_transformers", "faiss", "fitz",
        # 界面与音频
        "PyQt5", "pyaudio", "pygame",
    ]
    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"[✓] 依赖 {dep}")
        except ImportError:
            if dep == "llama_cpp":
                # 本项目默认用 Ollama 做文本推理，llama-cpp-python 属可选项
                print(f"[!] 依赖 {dep} — 未安装（可接受：文本走 Ollama 后端）")
            else:
                print(f"[✗] 依赖 {dep} — 未安装")
                all_ok = False
    return all_ok


def check_ollama() -> bool:
    """检查 Ollama 服务与所需模型。

    本项目默认用 Ollama 提供文本对话与视觉理解，所以这一项是核心检查。
    """
    try:
        from src.ollama_backend import OllamaBackend
    except Exception as exc:
        print(f"[✗] Ollama 检查不可用：{exc}")
        return False

    backend = OllamaBackend()
    if not backend.is_available():
        print(f"[✗] Ollama 服务未启动（{backend.base_url}）")
        print("    请安装并启动 Ollama：https://ollama.com/download")
        print(f"    然后执行：ollama pull {OLLAMA_CONFIG['text_model']}")
        print(f"              ollama pull {OLLAMA_CONFIG['vl_model']}")
        return False

    models = backend.list_models()
    print(f"[✓] Ollama 服务在运行（{backend.base_url}），本地模型 {len(models)} 个")
    # 文本模型可能配置了多个候选（推理模型慢/不干净，非推理模型优先），
    # 这里显示**实际会被选中**的那一个，而不是配置里的第一个默认值。
    from src.ollama_backend import resolve_text_model

    chosen = resolve_text_model()
    print(f"[i] 文本对话实际使用：{chosen}"
          + ("（候选：" + " → ".join(OLLAMA_CONFIG.get('text_model_candidates', []))
             + "）" if OLLAMA_CONFIG.get("text_model_candidates") else ""))
    ok = True
    for key, name in (("text_model", chosen), ("vl_model", OLLAMA_CONFIG["vl_model"])):
        if backend.has_model(name):
            print(f"[✓] Ollama 模型 {key}: {name}")
        else:
            print(f"[✗] Ollama 模型 {key}: 缺少 {name} → ollama pull {name}")
            ok = False
    return ok


def check_models() -> bool:
    """检查模型文件是否存在。

    GGUF 是**文件**、其它是**目录**，分开校验；
    文本/视觉各自有备选模型，主模型缺失但备选可用时只警告不判失败。
    """
    all_ok = True

    # ---- 语音识别（必须） ----
    asr_path = Path(MODEL_PATHS["asr"])
    if asr_path.exists():
        print(f"[✓] 模型 asr: {asr_path}")
    else:
        print(f"[✗] 模型 asr: {asr_path} — 路径不存在")
        all_ok = False

    # ---- 文本大模型（主 GGUF / 备选 transformers，二选一） ----
    text_path, backend = resolve_text_model()
    if backend == "gguf":
        print(f"[✓] 模型 text (gguf): {text_path}")
    elif backend == "transformers":
        print(f"[!] 模型 text: 未找到 GGUF，回退 transformers 版 → {text_path}")
    else:
        # 本地没有模型文件不算失败——本项目默认用 Ollama 提供文本能力
        try:
            from src.ollama_backend import resolve_text_model as _ollama_text

            _chosen = _ollama_text()
        except Exception:
            _chosen = OLLAMA_CONFIG["text_model"]
        print(f"[!] 模型 text: 本地未找到 GGUF / transformers 模型，"
              f"将使用 Ollama（{_chosen}）")

    # ---- 视觉语言模型（主 Qwen2.5-VL / 备选 Qwen2-VL） ----
    vl_path = resolve_vl_model()
    if vl_path:
        is_main = Path(vl_path) == Path(MODEL_PATHS["vl"])
        print(f"[{'✓' if is_main else '!'}] 模型 vl: {vl_path}"
              f"{'' if is_main else '  (回退到备选模型)'}")
    else:
        print(f"[!] 模型 vl: 本地未找到模型目录，将使用 Ollama"
              f"（{OLLAMA_CONFIG['vl_model']}）")

    # ---- MediaPipe 模型包（人体 / 手势检测必需） ----
    try:
        from config.settings import MODEL_ASSETS

        missing_assets = [k for k, p in MODEL_ASSETS.items() if not Path(p).exists()]
        if missing_assets:
            print(f"[✗] MediaPipe 模型包缺失：{'、'.join(missing_assets)}"
                  "（应为 models/mediapipe/*.task）")
            all_ok = False
        else:
            print("[✓] MediaPipe 模型包 pose/hands/face 均已就位")
    except Exception as exc:
        print(f"[!] MediaPipe 模型包检查跳过：{exc}")

    # ---- 知识库嵌入模型（缺失时自动下载，仅提示） ----
    emb = str(MODEL_PATHS["embedding"])
    if Path(emb).exists():
        print(f"[✓] 模型 embedding (本地): {emb}")
    else:
        print(f"[!] 模型 embedding: 使用模型名 {emb}（首次运行将自动下载）")

    return all_ok


def check_hardware() -> bool:
    """检查麦克风和摄像头。"""
    all_ok = True

    # 检查麦克风
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        input_devices = [
            pa.get_device_info_by_index(i)["name"]
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]
        if input_devices:
            print(f"[✓] 麦克风: {len(input_devices)} 个输入设备")
        else:
            print("[✗] 麦克风: 未找到输入设备")
            all_ok = False
        pa.terminate()
    except Exception as e:
        print(f"[✗] 麦克风: {e}")
        all_ok = False

    # 检查摄像头
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print("[✓] 摄像头: 正常读取画面")
            else:
                print("[✗] 摄像头: 可打开但无法读取画面")
                all_ok = False
            cap.release()
        else:
            print("[✗] 摄像头: 无法打开 (ID=0)")
            all_ok = False
    except Exception as e:
        print(f"[✗] 摄像头: {e}")
        all_ok = False

    return all_ok


def main():
    """运行全部检查。"""
    print("=" * 50)
    print("多模态AI助手 — 环境自检")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"推理设备: {DEVICE}")
    print("=" * 50)

    results = [
        ("Python 版本", check_python_version()),
        ("依赖检查", check_dependencies()),
        ("Ollama 检查", check_ollama()),
        ("模型检查", check_models()),
        ("硬件检查", check_hardware()),
    ]

    print("\n" + "=" * 50)
    all_pass = all(r[1] for r in results)
    if all_pass:
        print("✓ 全部检查通过！可以开始项目开发。")
    else:
        print("✗ 存在未通过的检查项，请按上述提示解决。")
        failed = [r[0] for r in results if not r[1]]
        print(f"未通过项: {', '.join(failed)}")
    print("=" * 50)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
