"""Ollama 后端客户端（文本对话 + 视觉理解统一入口）。

为什么单独抽一个模块
--------------------
文本大模型与视觉语言模型都要用到「向 Ollama 发请求」这件事，把它抽出来，
`text_llm.py` 与 `vl_model.py` 各自只做「组装 messages / 解析结果」，
不必重复实现 HTTP 细节。

实现要点：

* **只用标准库**（``urllib``），不额外引入 requests 等依赖；
* 走 Ollama 原生 ``/api/chat`` 接口，因为它支持消息里带 ``images``（视觉必需），
  OpenAI 兼容的 ``/v1/chat/completions`` 反而没有官方保证；
* 流式返回是「一行一个 JSON」，逐行解析出 ``message.content`` 即可；
* 任何网络异常都抛 :class:`OllamaError`，由调用方决定降级方式。
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Generator, Iterable
from typing import Any

from config.settings import OLLAMA_CONFIG

# 已知会输出"思考过程"的模型（qwen3 这类推理模型）。
# 它们默认先产出一大段 thinking 再给正式回答；不显式关闭的话，token 预算
# 可能被思考全部吃掉，content 直接为空——这是实测踩到的坑：
#   num_predict=512 → content=0 字、thinking=831 字、done_reason=length
# 改用官方 think=False 后 → content=837 字、thinking=0 字。
_THINKING_MODEL_HINTS = ("qwen3", "qwq", "deepseek-r1", "r1:", "magistral",
                         "reasoning", "thinking")


class OllamaError(RuntimeError):
    """Ollama 不可用或请求失败。"""


class OllamaBackend:
    """Ollama HTTP 客户端。

    Args:
        base_url: 服务地址，默认取 ``OLLAMA_CONFIG["base_url"]``。
        timeout: 请求超时秒数，默认取 ``OLLAMA_CONFIG["timeout"]``。
    """

    def __init__(self, base_url: str | None = None,
                 timeout: int | None = None):
        self.base_url = (base_url or OLLAMA_CONFIG["base_url"]).rstrip("/")
        self.timeout = timeout or OLLAMA_CONFIG["timeout"]
        self.default_model = OLLAMA_CONFIG["text_model"]

    # ------------------------------------------------------------------
    # 可用性探测
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Ollama 服务是否在运行。"""
        try:
            self.list_models()
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[str]:
        """列出本地已有的模型名（形如 ``qwen3:4b``）。"""
        data = self._request_json("/api/tags", payload=None, method="GET")
        models = data.get("models", []) if isinstance(data, dict) else []
        return [m.get("name", "") for m in models if m.get("name")]

    def has_model(self, name: str) -> bool:
        """本地是否已 pull 指定模型。

        允许省略 tag：``qwen3`` 能匹配到 ``qwen3:4b``。
        """
        names = self.list_models()
        if name in names:
            return True
        base = name.split(":")[0]
        return any(n.split(":")[0] == base for n in names)

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    def chat(self,
             messages: list[dict],
             model: str | None = None,
             temperature: float = 0.7,
             top_p: float = 0.9,
             max_tokens: int | None = None,
             keep_alive: str | None = None) -> str:
        """一次性对话，返回完整回答文本。"""
        payload = self._build_payload(messages, model, temperature, top_p,
                                      max_tokens, keep_alive, stream=False)
        data = self._post_chat("/api/chat", payload)
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            # 推理模型的思考把 token 预算吃光时会走到这里：此时 message.thinking
            # 里是"思考过程"而不是答案，直接当回答返回会让人以为模型在自言自语。
            # 与其糊弄，不如给出明确、可操作的提示。
            thinking = (message.get("thinking") or "").strip()
            if thinking:
                return ("（当前模型只产出了思考过程、没给出最终回答："
                        "思考已用尽 token 预算。建议改用非推理模型——"
                        "执行 ollama pull qwen2.5:3b 后程序会自动切换，"
                        "或调大 OLLAMA_CONFIG['thinking_max_tokens']。）")
        return self._strip_thinking(content)

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_thinking(text: str) -> str:
        """去掉 qwen3 等模型"思考过程 + </think> + 正式回答"里的思考部分。

        实测：即使 think=False，qwen3 仍会把推理写在 content 里，并以
        ``</think>`` 结尾，正式回答在其后。这里只保留 ``</think>`` 之后的内容。
        """
        if not text:
            return ""
        if "</think>" in text:
            return text.rsplit("</think>", 1)[-1].strip()
        # 有些版本用单独的 <think> 包裹
        if "<think>" in text and "</think>" not in text:
            return text.split("<think>", 1)[0].strip()
        return text.strip()

    def chat_stream(self,
                    messages: list[dict],
                    model: str | None = None,
                    temperature: float = 0.7,
                    top_p: float = 0.9,
                    max_tokens: int | None = None,
                    keep_alive: str | None = None) -> Generator[str, None, None]:
        """流式对话，逐片段 yield 文本（不是累积文本）。"""
        payload = self._build_payload(messages, model, temperature, top_p,
                                      max_tokens, keep_alive, stream=True)
        try:
            stream = self._request_stream("/api/chat", payload)
            # payload["think"] 为 True 时，思考内容在 message.thinking 里单独返回，
            # message.content 本身就是干净回答 —— 此时绝不能缓冲，
            # 否则整段回答会被压到流结束才吐出来（实测首 token 延迟 = 总耗时）。
            yield from self._filter_stream(
                stream, expect_inline_thinking=not payload.get("think", False))
        except OllamaError as exc:
            # 旧版 Ollama 不认识 think 参数时去掉它重试一次
            if "think" in payload and "think" in str(exc).lower():
                payload = {k: v for k, v in payload.items() if k != "think"}
                yield from self._filter_stream(
                    self._request_stream("/api/chat", payload),
                    expect_inline_thinking=True)
            else:
                raise

    def _filter_stream(self, stream,
                       expect_inline_thinking: bool = True
                       ) -> Generator[str, None, None]:
        """流式过滤。

        Args:
            expect_inline_thinking: 思考内容是否会混在 content 里。
                为 False 时（思考走 message.thinking 单独字段）直接透传，
                避免把整段回答缓冲到流结束才输出。
        """
        if not expect_inline_thinking:
            for piece in stream:
                yield piece
            return

        buffer = ""
        seen_end = False
        for piece in stream:
            if seen_end:
                yield piece
                continue
            buffer += piece
            if "</think>" in buffer:
                seen_end = True
                tail = buffer.rsplit("</think>", 1)[-1]
                buffer = ""
                if tail.strip():
                    yield tail
            elif len(buffer) > 200:
                # 一直没有 </think>：多半是普通模型，别再攒着，尽快吐出去
                seen_end = True
                yield buffer
                buffer = ""
        if not seen_end and buffer.strip():
            yield buffer

    def chat_with_images(self,
                         prompt: str,
                         images: Iterable[str | bytes],
                         system: str | None = None,
                         model: str | None = None,
                         temperature: float = 0.2,
                         max_tokens: int | None = None) -> str:
        """带图片的多模态问答。

        Args:
            prompt: 用户问题。
            images: 图片，支持文件路径（str）或原始字节（bytes）。
            system: 系统提示词（可选）。
            model: 视觉模型名，默认用 ``OLLAMA_CONFIG["vl_model"]``。
        """
        encoded = [self.encode_image(img) for img in images]
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": encoded})
        return self.chat(messages, model=model or OLLAMA_CONFIG["vl_model"],
                         temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    @staticmethod
    def encode_image(image: str | bytes) -> str:
        """把图片编码为 Ollama 需要的 base64 字符串。"""
        if isinstance(image, (bytes, bytearray)):
            raw = bytes(image)
        else:
            with open(image, "rb") as fh:
                raw = fh.read()
        return base64.b64encode(raw).decode("ascii")

    def _build_payload(self, messages: list[dict], model: str | None,
                       temperature: float, top_p: float,
                       max_tokens: int | None, keep_alive: str | None,
                       stream: bool) -> dict[str, Any]:
        """组装 /api/chat 请求体。"""
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
        if OLLAMA_CONFIG.get("num_ctx"):
            options["num_ctx"] = OLLAMA_CONFIG["num_ctx"]
        if max_tokens:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": keep_alive if keep_alive is not None
            else OLLAMA_CONFIG["keep_alive"],
        }
        # 推理模型（qwen3 等）的处理，实测结论见 config 里的注释：
        # 保留思考（think=True）时思考走单独字段、content 才是干净回答；
        # 强制 think=False 反而会让推演混进 content。这里同时给足 token 预算。
        if self._is_thinking_model(payload["model"]):
            payload["think"] = bool(OLLAMA_CONFIG.get("enable_thinking", True))
            floor = int(OLLAMA_CONFIG.get("thinking_max_tokens", 2048))
            if payload["think"] and options.get("num_predict", 0) < floor:
                options["num_predict"] = floor
        return payload

    @staticmethod
    def _is_thinking_model(model: str) -> bool:
        """判断模型是否属于"会输出思考过程"的一类。"""
        name = (model or "").lower()
        return any(hint in name for hint in _THINKING_MODEL_HINTS)

    def _post_chat(self, path: str, payload: dict) -> dict:
        """发送对话请求；旧版 Ollama 不认 think 参数时自动去掉重试。"""
        try:
            return self._request_json(path, payload)
        except OllamaError as exc:
            if "think" in payload and "think" in str(exc).lower():
                payload = {k: v for k, v in payload.items() if k != "think"}
                return self._request_json(path, payload)
            raise

    # ------------------------------------------------------------------
    def _request_json(self, path: str, payload: dict | None,
                      method: str = "POST") -> dict:
        """发一个非流式请求并解析 JSON 响应。"""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[:200]
            if exc.code == 404:
                raise OllamaError(
                    f"Ollama 中找不到模型（HTTP 404）：{body}。"
                    "请先执行 ollama pull 拉取对应模型。") from exc
            raise OllamaError(f"Ollama 请求失败 HTTP {exc.code}：{body}") from exc
        except Exception as exc:
            raise OllamaError(
                f"无法连接 Ollama（{self.base_url}）：{exc}。"
                "请确认已启动 Ollama 服务。") from exc

    def _request_stream(self, path: str,
                        payload: dict) -> Generator[str, None, None]:
        """发流式请求，逐行解析 NDJSON。"""
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    text = (obj.get("message") or {}).get("content") or ""
                    if text:
                        yield text
                    if obj.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[:200]
            if exc.code == 404:
                raise OllamaError(
                    f"Ollama 中找不到模型（HTTP 404）：{body}。"
                    "请先执行 ollama pull 拉取对应模型。") from exc
            raise OllamaError(f"Ollama 流式请求失败 HTTP {exc.code}：{body}") from exc
        except Exception as exc:
            raise OllamaError(
                f"无法连接 Ollama（{self.base_url}）：{exc}。"
                "请确认已启动 Ollama 服务。") from exc


# ----------------------------------------------------------------------
# 模块级便捷函数：直接用默认配置做状态检查（脚本/自检里最常用）
# ----------------------------------------------------------------------
def get_backend() -> OllamaBackend:
    """返回默认配置的 Ollama 客户端。"""
    return OllamaBackend()


def is_available() -> bool:
    """Ollama 服务是否可用。"""
    return get_backend().is_available()


def list_models() -> list[str]:
    """列出本地模型；服务不可用时返回空列表（不抛异常）。"""
    try:
        return get_backend().list_models()
    except OllamaError:
        return []


def has_model(name: str) -> bool:
    """本地是否有指定模型；服务不可用时返回 False。"""
    try:
        return get_backend().has_model(name)
    except OllamaError:
        return False


def resolve_text_model() -> str:
    """按优先级挑一个本地已有的文本模型。

    依次尝试 ``OLLAMA_CONFIG["text_model_candidates"]``，取第一个已 pull 的；
    都没有时返回 ``OLLAMA_CONFIG["text_model"]``（让后续报错信息更准确）。
    """
    backend = get_backend()
    try:
        for name in OLLAMA_CONFIG.get("text_model_candidates", []):
            if backend.has_model(name):
                return name
    except OllamaError:
        pass
    return OLLAMA_CONFIG["text_model"]
