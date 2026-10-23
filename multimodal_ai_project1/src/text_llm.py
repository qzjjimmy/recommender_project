"""阶段03 — 文本大语言模型模块。

默认基线：Qwen3-4B-Instruct-2507 **GGUF int4 量化版**，通过 llama.cpp 推理。

为什么不用 transformers 原生加载？
    同样是本地 CPU 推理，int4 量化把 4B 模型压到约 2.4 GB，
    比 fp16 的 Qwen2-1.5B（约 3 GB）体积更小、推理更快，
    而准确度（推理 / 数学 / 指令跟随）明显更好——
    这就是"量化"的价值：参数更多、体积更小、速度更快。

关键差异（最易踩的坑）：
    transformers 的 model_path 是**目录**；
    llama.cpp 的 model_path 是 **.gguf 文件**。两者语义不同，别混用。

若机器上没有 GGUF 文件，config.resolve_text_model() 会回退到
transformers 版（见 TextLLMTransformers）。
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator

from config.settings import (
    DEVICE,
    LLM_CONFIG,
    LLM_GGUF_CONFIG,
    OLLAMA_CONFIG,
    resolve_text_model,
)
from src.ollama_backend import (OllamaBackend, OllamaError,
                                resolve_text_model as _resolve_ollama_text_model)


class TextLLM:
    """文本大语言模型（GGUF 量化版，llama.cpp 后端）。

    提供 generate 与 stream_generate 两个接口，
    签名与 transformers 版本保持一致，可无缝替换。
    """

    SYSTEM_PROMPT = """你是一个多模态AI助手，具备以下能力：
1. 语音对话 — 理解用户的口语输入并给出自然回复
2. 视觉理解 — 分析摄像头画面并回答相关问题
3. 知识检索 — 基于本地知识库提供增强回答

你应当用友好、专业的中文回答用户的问题。回答简洁清晰，避免冗长。"""

    SYSTEM_PROMPT_KB = """你是一个严谨的问答助手。下面提供了【参考资料】。
如果参考资料中有与问题相关的内容，请优先基于参考资料回答，简明扼要，
尽量使用参考资料中的表述。
如果参考资料确实没有能回答该问题的内容，不要只回答"未找到"：
先用一句话说明知识库中没有相关内容，然后基于你自己的知识正常回答用户问题。"""

    def __init__(self, model_path: str | None = None,
                 n_ctx: int | None = None,
                 n_threads: int | None = None):
        """初始化文本 LLM。

        Args:
            model_path: **.gguf 文件路径**（不是目录）。
                留空则由 config.resolve_text_model() 自动挑选。
            n_ctx: 上下文窗口，默认取 LLM_GGUF_CONFIG。
            n_threads: 推理线程数，None 表示自动使用全部 CPU 核心。
        """
        if model_path is None:
            model_path, backend = resolve_text_model()
            if backend != "gguf":
                raise FileNotFoundError(
                    "未找到 GGUF 模型文件。请确认 models/gguf/*.gguf 存在，"
                    "或改用 TextLLMTransformers 加载 transformers 版模型。"
                )

        self.model_path = model_path
        self.n_ctx = n_ctx or LLM_GGUF_CONFIG["n_ctx"]
        self.n_threads = n_threads or LLM_GGUF_CONFIG["n_threads"]

        self.llm = None  # TODO: llama_cpp.Llama

    def load(self):
        """加载 GGUF 模型。

        要点：
            - `from llama_cpp import Llama` 放在函数内，避免模块导入时就拉起
              原生库（导入失败会连累不依赖 LLM 的脚本）。
            - 加载前先 os.path.exists 校验，给出清晰的中文报错。
        """
        if self.llm is not None:
            return self
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GGUF 模型文件不存在：{self.model_path}。\n"
                "请把 .gguf 文件放到 models/gguf/ 下，"
                "或使用 Ollama 后端（ollama pull qwen3:4b）。")
        # 延迟导入：llama_cpp 是原生库，导入失败不该连累其它脚本
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads or os.cpu_count(),
            verbose=LLM_GGUF_CONFIG["verbose"],
        )
        return self

    def build_messages(self, query: str,
                       context: str | None = None) -> list[dict]:
        """构建 OpenAI 风格的 messages（llama.cpp 直接吃这个格式，无需手动拼 prompt）。

        Args:
            query: 用户输入。
            context: RAG 知识库上下文（可选）。传入则切换到知识库模式。

        Returns:
            [{"role": "system", ...}, {"role": "user", ...}]
        """
        if context:
            system = f"{self.SYSTEM_PROMPT_KB}\n\n【参考资料】\n{context}"
        else:
            system = self.SYSTEM_PROMPT
        return [{"role": "system", "content": system},
                {"role": "user", "content": query}]

    def generate(self, query: str,
                 context: str | None = None,
                 max_new_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        """一次性生成完整回答。

        Args:
            query: 用户输入。
            context: RAG 知识库上下文（可选）。
            max_new_tokens: 最大生成 token 数。
            temperature: 温度参数。

        Returns:
            生成的完整文本。
        """
        self.load()
        messages = self.build_messages(query, context)
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"],
            temperature=(LLM_CONFIG["temperature"] if temperature is None
                         else temperature),
            stream=False,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def stream_generate(self, query: str,
                        context: str | None = None,
                        max_new_tokens: int | None = None,
                        temperature: float | None = None
                        ) -> Generator[str, None, None]:
        """流式生成，逐 token yield。

        Args:
            query: 用户输入。
            context: RAG 知识库上下文（可选）。
            max_new_tokens: 最大生成 token 数。
            temperature: 温度参数。

        Yields:
            每个生成的文本片段（不是完整累积文本）。
        """
        self.load()
        messages = self.build_messages(query, context)
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"],
            temperature=(LLM_CONFIG["temperature"] if temperature is None
                         else temperature),
            stream=True,
        )
        for part in stream:
            delta = part["choices"][0].get("delta") or {}
            text = delta.get("content") or ""
            if text:   # 首帧只有 role、没有 content，跳过以免界面多出空行
                yield text


class TextLLMTransformers:
    """备选后端：transformers 原生加载（无 GGUF 文件时的降级方案）。

    与 TextLLM 接口一致，可无缝替换。显存/内存充足且追求最佳效果时用这个。
    """

    def __init__(self, model_path: str, device: str | None = None):
        """初始化。

        Args:
            model_path: 模型**目录**路径（如 models/Qwen2-1.5B-Instruct）。
            device: 推理设备。
        """
        self.model_path = model_path
        self.device = device or DEVICE

        self.tokenizer = None  # TODO: AutoTokenizer
        self.model = None      # TODO: AutoModelForCausalLM

    def load(self):
        """加载模型和分词器。"""
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        use_cuda = self.device.startswith("cuda")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True)
        except TypeError:
            # 新版 transformers 用 dtype 取代了 torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True)
        if not use_cuda:
            self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return self

    def build_messages(self, query: str,
                       context: str | None = None) -> list[dict]:
        """构建 messages（与 TextLLM 同格式）。"""
        if context:
            system = f"{TextLLM.SYSTEM_PROMPT_KB}\n\n【参考资料】\n{context}"
        else:
            system = TextLLM.SYSTEM_PROMPT
        return [{"role": "system", "content": system},
                {"role": "user", "content": query}]

    def generate(self, query: str,
                 context: str | None = None,
                 max_new_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        """一次性生成完整回答。"""
        self.load()
        import torch

        messages = self.build_messages(query, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"],
                temperature=(LLM_CONFIG["temperature"] if temperature is None
                             else temperature),
                top_p=LLM_CONFIG["top_p"],
                do_sample=LLM_CONFIG["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id)
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(
            generated, skip_special_tokens=True).strip()

    def stream_generate(self, query: str,
                        context: str | None = None,
                        max_new_tokens: int | None = None,
                        temperature: float | None = None
                        ) -> Generator[str, None, None]:
        """流式生成（TextIteratorStreamer + 独立线程）。"""
        self.load()
        import torch
        from transformers import TextIteratorStreamer

        messages = self.build_messages(query, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"],
            temperature=(LLM_CONFIG["temperature"] if temperature is None
                         else temperature),
            top_p=LLM_CONFIG["top_p"],
            do_sample=LLM_CONFIG["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id)
        thread = threading.Thread(
            target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for token in streamer:
            if token:
                yield token
        thread.join()


class TextLLMOllama:
    """Ollama 后端（本项目默认）：通过 HTTP 调用本机 Ollama 服务。

    与 TextLLM / TextLLMTransformers 接口完全一致，可无缝替换。
    好处是不用自己编译 llama.cpp、不用手动下载几 GB 权重——
    `ollama pull qwen3:4b` 一条命令即可，模型由 Ollama 负责量化与常驻。
    """

    def __init__(self, model_path: str | None = None,
                 n_ctx: int | None = None,
                 n_threads: int | None = None):
        """初始化 Ollama 后端。

        Args:
            model_path: 这里传的是 **Ollama 模型名**（如 "qwen3:4b"），
                不是文件路径；留空则用 OLLAMA_CONFIG["text_model"]。
            n_ctx / n_threads: 由 Ollama 自行管理，这里仅保持接口一致。
        """
        name = model_path or _resolve_ollama_text_model()
        # 兼容误传本地路径的情况：路径不像模型名时回退到默认模型名
        if os.sep in name or name.endswith(".gguf"):
            name = _resolve_ollama_text_model()
        self.model_path = name
        self.model_name = name
        self.n_ctx = n_ctx or LLM_GGUF_CONFIG["n_ctx"]
        self.n_threads = n_threads or LLM_GGUF_CONFIG["n_threads"]
        self.backend = OllamaBackend()
        self.llm = None  # 保持与 TextLLM 相同的属性名（可用性标记）

    def load(self):
        """检查 Ollama 服务与模型是否就绪。"""
        if not self.backend.is_available():
            raise OllamaError(
                "Ollama 服务未启动。请先安装并启动 Ollama，"
                f"再执行：ollama pull {self.model_name}（地址 {self.backend.base_url}）")
        if not self.backend.has_model(self.model_name):
            raise OllamaError(
                f"Ollama 中还没有模型 {self.model_name}。请执行："
                f"ollama pull {self.model_name}")
        self.llm = self.backend
        return self

    def build_messages(self, query: str,
                       context: str | None = None) -> list[dict]:
        """构建 OpenAI 风格 messages（与 TextLLM 同格式）。"""
        if context:
            system = f"{TextLLM.SYSTEM_PROMPT_KB}\n\n【参考资料】\n{context}"
        else:
            system = TextLLM.SYSTEM_PROMPT
        return [{"role": "system", "content": system},
                {"role": "user", "content": query}]

    def generate(self, query: str,
                 context: str | None = None,
                 max_new_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        """一次性生成完整回答。"""
        self.load()
        return self.backend.chat(
            self.build_messages(query, context),
            model=self.model_name,
            temperature=(LLM_CONFIG["temperature"] if temperature is None
                         else temperature),
            top_p=LLM_CONFIG["top_p"],
            max_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"])

    def stream_generate(self, query: str,
                        context: str | None = None,
                        max_new_tokens: int | None = None,
                        temperature: float | None = None
                        ) -> Generator[str, None, None]:
        """流式生成，逐片段 yield 文本。"""
        self.load()
        yield from self.backend.chat_stream(
            self.build_messages(query, context),
            model=self.model_name,
            temperature=(LLM_CONFIG["temperature"] if temperature is None
                         else temperature),
            top_p=LLM_CONFIG["top_p"],
            max_tokens=max_new_tokens or LLM_CONFIG["max_new_tokens"])


def create_text_llm(**kwargs):
    """工厂函数：按本机可用模型自动选择后端。

    Returns:
        TextLLM（gguf）或 TextLLMTransformers 实例。
        两者接口一致，调用方无需关心具体后端。
    """
    # 优先级：Ollama（可用即用）→ GGUF → transformers。
    # 本机默认没有 .gguf 文件，因此正常走 Ollama；若你以后下了 GGUF，
    # 想强制用本地文件，把 OLLAMA_CONFIG 里的模型名清空即可。
    probe = OllamaBackend()
    if probe.is_available() and probe.has_model(OLLAMA_CONFIG["text_model"]):
        return TextLLMOllama(**kwargs)

    model_path, backend = resolve_text_model()
    if backend == "gguf":
        return TextLLM(model_path or None, **kwargs)
    if backend == "transformers":
        return TextLLMTransformers(model_path, **kwargs)
    raise FileNotFoundError(
        "未找到任何可用的文本大模型。\n"
        f"  方式一（推荐）：安装并启动 Ollama 后执行 ollama pull "
        f"{OLLAMA_CONFIG['text_model']}\n"
        "  方式二：把 .gguf 放到 models/gguf/，或把 transformers 版模型放到 "
        "models/Qwen2-1.5B-Instruct/。"
    )
