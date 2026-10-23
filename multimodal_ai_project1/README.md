# 多模态AI助手实践 — 学生项目脚手架

## 项目简介

本项目是《行业工程实践》模块二"多模态AI助手实践"的学生项目脚手
架。项目目标是构建一个具备语音对话、视觉理解、知识检索和手势交互
能力的本地多模态AI助手系统。

**核心能力：**
- 🎤 实时语音识别（VAD + SenseVoiceSmall）
- 💬 大语言模型对话（Qwen3-4B-Instruct GGUF int4，llama.cpp 流式生成）
- 👁️ 视觉语言理解（Qwen2.5-VL-3B-Instruct，bfloat16）
- 📚 本地知识库 RAG（FAISS 向量检索 + all-MiniLM-L6-v2 嵌入）
- 🔊 语音合成（edge-tts 流式 TTS）
- ✋ 人体姿态与手势识别（MediaPipe）

**模型基线（2026-09 更新）：**

| 用途 | 模型 | 体积 | 推理方式 |
|------|------|------|----------|
| 语音识别 | SenseVoiceSmall | 0.9 GB | funasr |
| 文本对话 | **Qwen3-4B-Instruct-2507** | 2.4 GB | llama.cpp（GGUF int4） |
| 视觉理解 | **Qwen2.5-VL-3B-Instruct** | 7.2 GB | transformers（bfloat16） |
| 知识库嵌入 | all-MiniLM-L6-v2 | 0.09 GB | sentence-transformers |

> 文本模型从 Qwen2-1.5B 换成 Qwen3-4B 的 GGUF int4 后，**参数更多但体积更小、
> 速度更快、效果更好**——这就是量化的价值。视觉模型同理升级为 Qwen2.5-VL-3B。
> 两者都保留了旧版作为备选，缺失时自动回退（见 `config/settings.py`）。

## 快速开始

### 1. 环境要求

- Python 3.10+
- 内存 16 GB 以上（实测 CPU 全量加载常驻约 8 GB，VL 模型峰值约 10.7 GB）
- 推荐 GPU：NVIDIA 8GB+ 显存（可把 VL 与 LLM 都放显存）；无 GPU 也可运行，CPU 推理较慢
- 麦克风和摄像头
- 磁盘剩余 15 GB 以上（模型约 10.5 GB）
- Windows / macOS / Linux

### 2. 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n multimodal_ai python=3.10 -y
conda activate multimodal_ai

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
# 本项目在 macOS(Apple Silicon) 上用国内清华镜像安装（直连 PyPI 很慢）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 脚手架清单漏了 torchvision，而 qwen_vl_utils 依赖它，必须补装：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision

# pyaudio 在 Apple Silicon 上没有预编译轮子，需先装 portaudio 再用源码编译：
brew install portaudio
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyaudio
```

**关于 `llama-cpp-python`**：本项目文本推理默认走 **Ollama**，所以这个包
**不装也能跑**（`requirements.txt` 里保留它只是为了兼容"用本地 .gguf 文件"的路线）。
若在 Windows / macOS 上编译失败，直接忽略即可；`scripts/00_check_env.py`
会把它标为"可接受：文本走 Ollama 后端"。

**Windows 用户特别注意：**
- `pyaudio` 可能需要 Visual C++ Build Tools；若安装失败，优先查阅 PyAudio 的 Windows 安装说明或使用与 Python 版本匹配的预编译 wheel。
- `llama-cpp-python` 在 Windows 上常因缺少 C++ 编译环境而失败。装预编译 wheel：
  ```bash
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  ```
  这个包**缺失会导致阶段03直接起不来**，且不会自动回退到 transformers 版。
- 请在独立虚拟环境中安装依赖，避免影响系统 Python 环境。

### 4. 下载模型

#### 4.1 文本大模型与视觉模型 —— 用 Ollama（推荐，本项目默认）

Ollama 用一条命令就能拿到量化好的 Qwen 模型，并对外提供 HTTP 接口，
代码里只需发请求，不必自己编译推理引擎、也不必手动下载几 GB 权重。

```bash
# 1) 安装 Ollama（macOS 直接下载 App；也可 brew install ollama）
#    https://ollama.com/download

# 2) 拉取本项目配置的两个模型
ollama pull qwen3:4b          # 文本对话
ollama pull qwen2.5vl:3b      # 视觉理解（支持图片输入）

# 3) 确认服务在跑（默认 http://127.0.0.1:11434）
ollama list
```

模型名写在 `config/settings.py` 的 `OLLAMA_CONFIG` 里，改成别的量化档位（如
`qwen3:1.7b`）只需改这一处。也支持用环境变量 `OLLAMA_HOST` 指定服务地址。

> **为什么文本/视觉走 Ollama，而不用 llama.cpp 和 transformers？**
> 本机是 Apple M2、16GB 内存、无 NVIDIA 显卡：
> ① `llama-cpp-python` 在 macOS 上要现场编译，容易失败；
> ② Qwen2.5-VL-3B（bf16）权重就有 7.2GB，下载与加载都吃力。
> Ollama 的方案在**不改动脚手架任何函数签名**的前提下就能跑通，
> 且保留了 GGUF / transformers 两条备用后端（见 `create_text_llm()`）。

#### 4.2 语音识别与向量嵌入 —— 本地小模型（必须）

Ollama 不做语音识别，也不做文本向量嵌入，这两个小模型需要放在 `models/` 下：

```text
models/
├── SenseVoiceSmall/            # 语音识别（funasr），约 0.9 GB，ModelScope: iic/SenseVoiceSmall
├── all-MiniLM-L6-v2/           # 知识库嵌入，约 0.09 GB，ModelScope: sentence-transformers/all-MiniLM-L6-v2
└── mediapipe/                  # 人体/手势检测模型包（见 4.3）
```

> HuggingFace 在本机网络不可达，这两个模型请从 **ModelScope** 下载；
> 若 `all-MiniLM-L6-v2` 目录不存在，`sentence-transformers` 会尝试联网自动下载
> （网络不通时会失败，所以建议一并本地放到 `models/` 下）。

#### 4.3 MediaPipe 模型包（.task，已随项目下载好）

Apple Silicon 上只能装到 `mediapipe>=0.10.30`，而这些版本**已移除旧的
`mp.solutions` 接口**，只保留 Tasks 接口；Tasks 接口需要独立的 `.task` 模型文件：

```text
models/mediapipe/
├── pose_landmarker_full.task   # 9.0 MB
├── hand_landmarker.task        # 7.5 MB
└── face_landmarker.task        # 3.6 MB
```

来源为 Google 官方存储（`https://storage.googleapis.com/mediapipe-models/...`），
本项目已下载并放在上述位置。`config/settings.py` 的 `MODEL_ASSETS`
记录了路径、`LANDMARKER_URLS` 记录了下载地址，缺失时会给出明确提示。

### 5. 配置

编辑 `config/settings.py` 按需调整。**本项目默认配置已经可用**，
核心是这一段：

```python
OLLAMA_CONFIG = {
    "base_url": "http://127.0.0.1:11434",  # 也可用环境变量 OLLAMA_HOST 覆盖
    "text_model": "qwen3:4b",
    "vl_model":   "qwen2.5vl:3b",
    ...
}

MODEL_PATHS = {
    "asr":       "models/SenseVoiceSmall",
    "text_gguf": "models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",  # 备选
    "text":      "models/Qwen2-1.5B-Instruct",                       # 备选
    "vl":        "models/Qwen2.5-VL-3B-Instruct",                    # 备选
    "vl_legacy": "models/Qwen2-VL-2B-Instruct",                      # 备选
}
```

文本后端的优先级是：**Ollama（可用即用）→ GGUF → transformers**，
视觉则是：**本地模型目录 → Ollama**。`resolve_text_model()` /
`resolve_vl_model()` 的原有逻辑未改动，只是工厂函数新增了 Ollama 分支。

### 6. 环境自检

```bash
python scripts/00_check_env.py
```

全部通过后即可进入各阶段开发。

### 7. 启动系统（阶段06完成后）

```bash
python main.py
```

## 项目结构

```
multimodal_ai_project/
├── main.py                  # 系统主入口（阶段06）
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
├── 目录规范.md               # 目录职责说明
├── config/
│   ├── __init__.py          # 配置导出
│   └── settings.py          # 全局配置 ⭐需要填写
├── data/                    # 数据文件（知识库文档等）
├── models/                  # 预训练模型存放目录
│   ├── SenseVoiceSmall/
│   ├── gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
│   ├── Qwen2.5-VL-3B-Instruct/
│   └── all-MiniLM-L6-v2/
├── scripts/                 # 各阶段入口脚本
│   ├── 00_check_env.py      # 环境自检
│   ├── 01_capture_audio.py  # 阶段01：音频采集测试
│   ├── 02_capture_video.py  # 阶段01：视频采集测试
│   ├── 03_test_asr.py       # 阶段02：ASR测试
│   ├── 04_test_llm.py       # 阶段03：LLM测试
│   ├── 05_test_kb.py        # 阶段03：知识库测试
│   ├── 06_test_vl.py        # 阶段04：VL测试
│   ├── 07_test_tts.py       # 阶段04：TTS测试
│   ├── 08_test_detection.py # 阶段05：检测测试
│   └── 09_e2e_test.py       # 阶段06：端到端测试
├── src/                     # 核心模块（需要填充业务逻辑）⭐
│   ├── __init__.py
│   ├── audio_capture.py     # 阶段01
│   ├── video_capture.py     # 阶段01
│   ├── vad.py               # 阶段02
│   ├── asr.py               # 阶段02
│   ├── audio_thread.py      # 阶段02
│   ├── text_llm.py          # 阶段03
│   ├── knowledge_base.py    # 阶段03
│   ├── rag.py               # 阶段03
│   ├── vl_model.py          # 阶段04
│   ├── tts.py               # 阶段04
│   ├── tts_player.py        # 阶段04
│   ├── modal_router.py      # 阶段04
│   ├── human_detector.py    # 阶段05
│   ├── gesture.py           # 阶段05
│   ├── gesture_actions.py   # 阶段05
│   ├── bridge.py            # 阶段05
│   └── gui.py               # 阶段05
└── outputs/                 # 输出产物（WAV/MP3/JPG/CSV等）
```

## 开发说明

### 脚手架不是标准答案

本项目提供的 `src/` 模块包含函数签名和 TODO 注释，**不含完整业务
逻辑**。你需要：

1. 阅读阶段任务书理解需求
2. 阅读源码注释理解模块职责
3. 自主填充 TODO，实现完整功能

### 阶段依赖关系

```
阶段01（采集） → 阶段02（ASR） → 阶段03（LLM + RAG）
                                  → 阶段04（VL + TTS）
                                  → 阶段05（检测 + GUI）
                                    └→ 阶段06（集成 + 答辩）
```

每个阶段的输出是下一阶段的输入。建议完成一阶段并通过验收后再进入
下一阶段。

### 代码规范

- 使用 Python 类型注解（`list[str]`, `dict[str, Any]` 等）
- docstring 描述功能、参数和返回值
- 配置集中在 `config/settings.py`，避免硬编码
- SEED = 42，确保可复现
- 异常处理：不因单个模块异常导致整个系统崩溃

### 提交要求

- 每个阶段完成后的完整代码
- 输出产物（测试音频/图片/日志）
- 模块项目报告（阶段06）

## 考核方式

- 平时成绩 × 0.4（阶段验收 + 课堂参与）
- 模块项目成绩 × 0.6（项目报告 + 答辩演示）

---

## 实现说明与已实测结果

### 模块实现状态

所有阶段都已完成，`src/` 下原本的 TODO 均已填充，**类名与函数签名未做任何删改**，
只在原有签名基础上增加了少量辅助方法，并新增了一个模块：

| 阶段 | 完成内容 |
|------|----------|
| 01 采集 | `AudioCapture`（PyAudio）、`VideoCapture`（OpenCV，带打开超时保护） |
| 02 语音 | `VADProcessor`（WebRTC VAD + 句首预缓冲 + 句尾补齐）、`ASREngine`（FunASR SenseVoiceSmall）、`AudioStreamThread`（采集→断句→WAV→识别→回调） |
| 03 对话 | `TextLLM`（llama.cpp GGUF）、`TextLLMTransformers`（备用）、**`TextLLMOllama`（默认）**、`create_text_llm()` 三后端工厂；`DocumentParser` / `VectorStore`（FAISS）/ `RAGPipeline` |
| 04 视觉与语音 | `VLModel`（transformers 多帧 video 输入 + **Ollama 视觉后端**）、`TTSEngine`（edge-tts）、`StreamingTTSPlayer`（分句流式播放）、`ModalRouter` |
| 05 检测与界面 | `HumanDetector`（**MediaPipe Tasks 接口**）、`GestureRecognizer`（6 种手势）、`GestureActionMapper`（新增 `execute`）、`Bridge`（补齐 emit 方法）、`MainWindow`（视频+字幕+对话+控制栏+线程回收） |
| 06 集成 | `main.py`（`VideoThread` + 依赖顺序组装 + VL 懒加载 + 缺模型降级）、`scripts/00~09` 全部实现 |

新增文件：`src/ollama_backend.py`（只依赖标准库 `urllib` 的 Ollama 客户端）。

### 已实测结果（Apple M2 / 16GB / 无独显）

| 验证项 | 结果 |
|--------|------|
| 依赖导入（torch/transformers/funasr/mediapipe/pyaudio/cv2/PyQt5…） | 全部通过 |
| 麦克风 / 摄像头 | 5 个输入设备 / 摄像头 ID 0 可读 |
| `scripts/00_check_env.py` | **全部检查通过** |
| 语音识别（SenseVoiceSmall） | 端到端可用，实测识别延迟 **0.93 s**（rtf 0.165） |
| 实时语音链路 | 麦克风 → VAD 断句 → 识别，一次测试正确断出 3 句 |
| 文本对话（Ollama qwen2.5:3b） | 基础对话 3.97 s；流式首 token 3.64 s |
| 知识库 RAG | 导入 13→30 块，检索命中 3 块，问答 **1.65 s** |
| 视觉理解（Ollama qwen2.5vl:3b） | 2 帧 20.6 s / 4 帧 37.2 s，描述准确；临时图片零残留 |
| 手势识别（合成关键点） | **6/6 = 100%**；冷却机制正常 |
| `HumanDetector` 初始化 / 单帧推理 | 1.86 s / 0.177 s（真机画面） |
| VAD | 1.2 s 纯静音不触发；语音段正确断句 |
| PDF 解析 | 指导书解析出 37 个文本块 |
| TTS 合成 | edge-tts 中文 20 KB / 英文 18 KB，分句流式播放正常 |
| 端到端 E2E | 8 项通过；2 项为人工/长时用例（中英混合口述、30 分钟挂机） |
| `main.py` 启动 | 文本/TTS/摄像头/音频全部初始化成功；缺模型时明确降级不崩溃 |

指标见 `outputs/performance_metrics.csv`，用例记录见 `outputs/e2e_test_results.csv`。

### 与原脚手架的两处必要偏差（已在代码注释中说明）

1. **MediaPipe：`mp.solutions` → Tasks 接口**。Apple Silicon 上只能装
   `mediapipe>=0.10.30`，而这些版本已移除 Solutions，原代码会「导入即崩」。
   改用 `PoseLandmarker / HandLandmarker / FaceLandmarker` 实现同等能力，
   骨架绘制改为 OpenCV 手动画（Tasks 不再提供 `drawing_utils`）。
2. **手势判定新增"拇指伸展/收拢"判别**。仅比较 y 坐标无法区分拇指竖起与
   拇指收拢在掌心，会把"👇 食指向下"误判成"👎 倒赞"；改用「拇指尖到食指根
   的距离」判定后 6 种手势全部正确。

### 接上 Ollama 后实测踩到的三个坑（已在代码中修好）

1. **qwen3 的"思考"会吃掉 token 预算**。qwen3:4b 是推理模型，模板里
   `<think>` 是**无条件写入**的，所以在 prompt 里塞 `/no_think` 无效；
   而强制 `think:false` 又会让推演混进 `message.content`（回答变成
   "首先，用户要求我…"，不可用）。
   实测最优解：**保留 `think:true`**——思考走 `message.thinking`（不展示），
   `content` 才是干净回答；同时把推理模型的 token 下限提到 2048
   （设 512 时思考没结束、`content` 为空）。
2. **多帧图片会撑爆默认上下文**。Ollama 默认 `num_ctx=4096`，
   送 4 帧 640×480 图片约 8400 tokens，直接报 `exceed_context_size_error`。
   解决：`OLLAMA_CONFIG["num_ctx"]=8192`，并在送模型前把帧缩到最大边长
   `VL_CONFIG["max_side"]=512`。
3. **流式输出被自己的过滤器拖成"一次性返回"**。初版按 `</think>` 缓冲，
   但 `think:true` 时思考在单独字段、`content` 里没有该标记，导致整段回答
   被攒到流结束才吐出（首 token 延迟 = 总耗时）。现在只在"思考会混进
   content"时才缓冲。

### 文本模型建议

建议再拉一个**非推理**模型，回答更干净也更快：

```bash
ollama pull qwen2.5:3b
```

`OLLAMA_CONFIG["text_model_candidates"]` 已按 `["qwen2.5:3b", "qwen3:4b"]`
排好优先级，程序会自动挑第一个已 pull 的，**不需要改代码**。
实测对比：qwen3:4b 在知识库问答场景会"思考"4500+ 字，单轮约 100 秒且
可能给不出最终回答；qwen2.5:3b 没有思考过程，回答直接可用。

### 第四个坑：ASR 的 fbank 后端

`funasr` 提取特征需要 torchaudio 或 kaldi-native-fbank 之一，且 **torchaudio
必须和 torch 版本严格配套**。本机是 torch 2.14 + torchaudio 2.11（不配套），
于是 ASR 报：

```
torchaudio is not installed and neither is the kaldi-native-fbank fallback backend.
```

装上纯 CPU 的 `kaldi-native-fbank` 即可（与 torch 版本解耦，已写进
`requirements.txt`）：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kaldi-native-fbank
```

---

## 作为项目一（推荐系统）的后端：本地桥接服务

**为什么有这个东西**：项目一（`recommender_project`，电商商品智能推荐系统）的「智能推荐」
页原来用浏览器在线语音识别、纯 JS 肤色分割手势、云端大模型解析。本项目的这些本地能力
（SenseVoiceSmall 语音识别、MediaPipe 手部关键点、Qwen2.5-VL 视觉理解、Ollama 本地大模型、edge-tts）
正好可以替代它们，但两个项目跑在**不同的 conda 环境**里，而且都有一个顶层 `config` 包，
直接互相 import 会撞名。

所以本项目提供一个**只依赖 Python 标准库**的本地 HTTP 服务，把能力开放出去：

```
┌──────────────────────────┐         HTTP (127.0.0.1:8100)        ┌────────────────────────────┐
│ 项目一 Flask :5000        │ ───────────────────────────────────► │ bridge_server.py            │
│  智能推荐页面             │ /asr /gesture /vision /tts /llm     │  ASR / MediaPipe / Vision / │
└──────────────────────────┘ ◄─────────────────────────────────── │  Ollama / edge-tts          │
                                                                   └────────────────────────────┘
```

### 启动

```bash
conda activate multimodal_ai
cd multimodal_ai_project1
python bridge_server.py                 # 默认 127.0.0.1:8100
python bridge_server.py --port 8100 --verbose      # 打印每个请求
python bridge_server.py --full-detection           # 手势跑 Pose+Hands+Face（更慢）
```

服务是**可选**的：不启动它，项目一页面会自动降级回内置的浏览器语音 / 纯 JS 手势 /
规则解析，不会报错。

### 接口

| 接口 | 请求 | 响应 |
|------|------|------|
| `GET /health` | — | `{status, ollama:{...}, asr:{...}, mediapipe:{...}, vision:{...}, tts:{...}}` |
| `POST /asr` | `audio/wav` 原始字节（兼容 `{"audio_b64":...}`） | `{"text","elapsed_ms","engine"}` |
| `POST /gesture` | `image/jpeg` 原始字节 | `{"gesture":"thumbs_up"\|"thumbs_down"\|null,"label","hands","engine"}` |
| `POST /vision` | `{"image_b64":...,"categories":[...]}` | `{"scene","objects","colors","category_hints","search_query","model","elapsed_ms"}` |
| `POST /tts` | `{"text","voice"?}` | `audio/mpeg` 字节流 |
| `POST /llm` | `{"messages":[...],"model"?,"temperature"?,"max_tokens"?}` | `{"content","model","elapsed_ms"}` |

设计要点：

* **手势接口只做单帧判定**，"稳定保持 0.9 秒 + 冷却 1.5 秒"由前端做（前端能按时间窗口统计，
  比后端凭时间戳更准，也不怕请求排队）；
* **手势默认只跑 Hands 模型**：Pose+Hands+Face 全量单帧要 0.8~0.9 秒，而前端按 5 帧/秒轮询，
  喂不动；只用 Hands 后实测 0.03~0.05 秒/帧（约快 20 倍），精度不受影响；
* **视觉接口只接收单张图片**：最长边缩到 768 像素并在内存中重编码为 JPEG，去掉原始
  EXIF；只输出可见物体、颜色和项目一真实类目，不识别人物身份，也不把原图写入磁盘；
* 模型全部**懒加载**，加载失败会被记住并如实上报（`/health` 里能看到 error），不会反复重试刷屏；
* HTTP 层是多线程的，但模型推理用一把锁**串行化**（MediaPipe/FunASR 的推理不是线程安全的）。

### 自检

```bash
python scripts/10_test_bridge.py                  # 全部检查（含较慢的视觉推理）
python scripts/10_test_bridge.py --skip-asr       # 跳过较慢的语音识别
```

脚本会在内存里另起一个随机端口的服务，依次请求 6 个接口并打印耗时与结论：

```
  ✓ health     21ms   | ollama=True model=qwen2.5:3b | mediapipe=True
  ✓ asr        7791ms | 首次含模型加载；识别结果非空
  ✓ gesture    864ms  | snapshot_000.jpg → 1 只手（含首次加载）
  ✓ vision    15944ms | 家用电器.jpg → 类目 ['家用电器']
  ✓ tts        14264ms| 26784 字节，已保存 bridge_selftest.mp3
  ✓ llm        3234ms | model=qwen2.5:3b | 回答："北京"
  通过 5/5
```

> 依赖提醒：`/asr` 需要 `funasr` + 本地 `models/SenseVoiceSmall`；`/gesture` 需要
> `mediapipe` + `models/mediapipe/*.task`；`/vision` 需要 Ollama 已拉取
> `qwen2.5vl:3b`；`/tts` 需要联网（edge-tts 走微软服务）；
> `/llm` 需要 Ollama 常驻且已 `ollama pull qwen2.5:3b`。

---

**课程归属：《行业工程实践》模块二**
