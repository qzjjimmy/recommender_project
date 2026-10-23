"""可选的大模型查询解析（OpenAI 兼容接口）。

为什么要有这个东西
------------------
纯规则解析擅长"精确、可枚举"的表达（价格数字、已知类目名），但拿"便宜又好看的家居
小礼物"这类模糊表达没办法。大模型恰好相反：语义理解强，但**不能保证输出格式与数值准确**。
所以两者是互补关系，正确的组合方式是：

    规则先跑  →  规则够用就直接返回（零延迟、零成本、结果可解释）
              →  规则不够用才调大模型补充
              →  大模型的输出必须校验后才与规则结果合并

四条工程约束
------------
1. **可插拔**：不配置 API Key 就完全不工作，系统自动退回纯规则解析；
2. **按需调用**：只把规则搞不定的句子送进去，省调用次数与延迟；
3. **输出校验**：品类必须在已知类目里、价格必须是合理范围内的数字，
   不合法一律丢弃 —— 绝不让大模型的输出直接进入过滤逻辑；
4. **静默降级**：超时、网络错误、返回不是 JSON …… 任何异常都回退到规则结果，
   功能降级但不报错。

接口约定：任何 OpenAI 兼容服务都可以（DeepSeek、通义千问、vLLM、Ollama、…），
只要填对 ``base_url`` 与 ``model``。

两种后端（``search.llm.backend``）
--------------------------------
* ``"openai"``：直接调 ``base_url`` 上的 OpenAI 兼容接口（如 DeepSeek 云端）；
* ``"bridge"``（推荐）：调**项目二**的本地桥接服务 ``POST /llm``，由项目二用本地
  Ollama 模型解析——不联网、不需要 API Key，项目一的提示词与校验逻辑原样复用。

无论哪种后端，超时/网络错误/格式不对都只会让本次解析失败，调用方回退到规则解析。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from config.settings import PATHS
from src.query_parser import QueryConstraints, normalize_keyword

API_KEY_ENV = "RECOMMENDER_LLM_API_KEY"

VALID_BACKENDS = ("openai", "bridge")


def load_api_key(explicit: str = "") -> str:
    """按优先级取 API Key：显式传入 > 环境变量 > 本地文件。

    把 Key 放在 ``data/.llm_key``（已加进 .gitignore）里，
    就不必每次在 PyCharm 里配环境变量，也不会误提交到代码仓库。
    """
    if explicit:
        return explicit.strip()
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return env_key
    try:
        return Path(PATHS["llm_key_file"]).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_api_key(key: str, path: Path | None = None) -> Path:
    """把 Key 写到本地文件（仅本机可读的普通文件，不会进版本库）。"""
    target = Path(path or PATHS["llm_key_file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(key.strip(), encoding="utf-8")
    try:  # 尽量收紧权限（Windows 上可能不支持，忽略即可）
        target.chmod(0o600)
    except OSError:
        pass
    return target

SYSTEM_PROMPT = (
    "你是电商推荐系统的查询解析器。把用户的一句话解析成结构化条件，"
    "只输出一个 JSON 对象，不要任何解释文字。\n"
    '格式：{"categories": ["类目"], "min_price": 数字或null, '
    '"max_price": 数字或null, "keywords": ["关键词"]}\n'
    "要求：\n"
    "1. categories 只能从下方给出的候选类目里选，最多 2 个；选不出就填空数组；\n"
    "2. 价格只填数字（不要带单位）。用户表达「便宜 / 实惠 / 划算 / 不贵」时，"
    "参考下方价格信息给一个合理的 max_price；表达「高端 / 精致 / 贵一点」时给一个合理的 min_price；"
    "完全没有价格线索才填 null；\n"
    "3. keywords 只放**能出现在商品名称里的具体词**（颜色、材质、图案、用途、品类细分），最多 3 个。"
    "主观评价词（便宜、好看、高档、实用、精美）**不要**放进 keywords —— "
    "它们应该体现在价格或类目上，而不是去匹配商品名；\n"
    "4. 绝对不要编造用户没提到的条件。"
)

# 价格上限的表达：模型把"X以内"读成"min=max=X"时，用这组线索把它纠正回"只保留上限"。
# 这种退化区间会造出"≥X 且 ≤X"的必然空集，是页面回退规则解析的常见原因。
# 为什么不靠提示词解决：实测在本项目的 3B 本地模型上，往 SYSTEM_PROMPT 里加示例/加约束
# 虽然能让"8000以内"读对，却会把"适合送女朋友的礼物"这类模糊查询读坏（品类直接丢了），
# 属于把失败模式挪个地方。所以护栏放在**校验层**（确定性、可测），提示词保持原样。
_UPPER_BOUND_RE = re.compile(r"以内|以下|不超过|不高于|最多|至多|预算|左右|之内|以里")

# 纯数字/纯符号/纯标点的关键词没有任何意义（例如型号里的"60"），直接丢弃
_JUNK_KEYWORD_RE = re.compile(r"[\d\W_]+")


class LLMQueryParser:
    """把自然语言查询交给大模型解析，返回 :class:`QueryConstraints`。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        backend: str = "openai",
        base_url: str = "",
        model: str = "",
        api_key: str | None = None,
        timeout: float = 8.0,
        max_price: float = 100000.0,
        bridge=None,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend = (backend or "openai").strip().lower()
        self.base_url = (base_url or "").rstrip("/")
        self._model = model or ""
        self.api_key = load_api_key(api_key or "")
        self.timeout = float(timeout)
        self.max_price = float(max_price)
        # 只有 backend="bridge" 时才用得上；延迟构造避免 import 环
        self._bridge = bridge
        # 运行统计：便于在界面上说明"这次到底有没有用到大模型"
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------ 桥接
    @property
    def bridge(self):
        """项目二桥接服务的客户端（仅 ``backend="bridge"`` 时需要）。"""
        if self._bridge is None:
            from src.multimodal_client import get_client

            self._bridge = get_client()
        return self._bridge

    @property
    def model(self) -> str:
        """本次解析实际使用的模型名（桥接后端从桥接服务探测）。"""
        if self.backend == "bridge":
            try:
                return str(self.bridge.status().get("text_model") or "")
            except Exception:  # noqa: BLE001 - 探测失败不影响解析
                return ""
        return self._model

    # ------------------------------------------------------------ 状态
    @property
    def available(self) -> bool:
        """是否可用。两种后端的判据不同：

        * ``bridge``：桥接服务可达 **且** 它背后的本地 Ollama 有可用模型
          （不需要 API Key）。少了后半句会踩坑：桥接进程活着、Ollama 没启动时
          这里会误判为"可用"，用户点了按钮才失败，界面上只能看到"调用/失败次数"，
          排查方向被带偏到 API Key 上；
        * ``openai``：开关打开 + 有地址 + 有模型名 + 有 Key。
        """
        if not self.enabled:
            return False
        if self.backend == "bridge":
            try:
                status = self.bridge.status()
                return bool(status.get("available")) and bool(status.get("llm"))
            except Exception:  # noqa: BLE001
                return False
        return bool(self.base_url and self._model and self.api_key)

    def failure_hint(self) -> str:
        """解析失败时给用户的排查建议（按后端给不同提示，避免指错方向）。"""
        if self.backend == "bridge":
            return (
                "本次走的是项目二本地桥接（不需要 API Key）。请确认："
                "① 已启动 Ollama（打开 Ollama.app 或执行 ollama serve）；"
                "② 已 pull 文本模型（ollama pull qwen2.5:3b）；"
                "③ 项目二桥接服务在跑（cd multimodal_ai_project1 && python bridge_server.py）"
            )
        return "常见原因：网络不通、Key 无效或超时。可运行 scripts/11_set_llm_key.py --check 自检"

    def describe(self) -> str:
        if not self.enabled:
            return "未启用（在 config/settings.py 里打开 search.llm.enabled 即可）"
        if self.backend == "bridge":
            status = self.bridge.status()
            if not status.get("available"):
                return f"本地桥接服务不可用：{status.get('reason')}"
            if not status.get("llm"):
                return ("桥接服务在跑，但本地 Ollama 没有可用模型"
                        "（可执行 ollama pull qwen2.5:3b）")
            return f"已启用：{status.get('text_model')} @ 本地桥接服务（项目二）"
        if not self.api_key:
            return (
                f"未配置 API Key。二选一：① 运行 scripts/11_set_llm_key.py 保存到本地；"
                f"② 设置环境变量 {API_KEY_ENV}"
            )
        if not self.base_url or not self._model:
            return "缺少 base_url 或 model"
        return f"已启用：{self._model} @ {self.base_url}"

    # ------------------------------------------------------------ 主流程
    def parse(
        self,
        text: str,
        categories: Iterable[str],
        price_context: str = "",
    ) -> QueryConstraints | None:
        """调用大模型解析；失败返回 ``None``（由调用方回退到规则结果）。"""
        text = (text or "").strip()
        if not text:
            return None

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        (f"价格信息：{price_context}\n" if price_context else "")
                        + "候选类目：" + "、".join(str(c) for c in categories)
                        + f"\n用户输入：{text}"
                    ),
                },
            ],
        }

        self.calls += 1
        try:
            raw = self._post(payload)
            data = self._extract_json(raw)
            if data is None:
                self.failures += 1
                return None
            return self._validate(data, set(str(c) for c in categories), text)
        except Exception:  # noqa: BLE001 - 任何异常都降级，不能让页面 500
            self.failures += 1
            return None

    # ------------------------------------------------------------ HTTP
    def _post(self, payload: dict[str, Any]) -> str:
        """发一次解析请求，返回模型的原始文本回复。"""
        if self.backend == "bridge":
            return self._post_via_bridge(payload)
        return self._post_openai(payload)

    def _post_via_bridge(self, payload: dict[str, Any]) -> str:
        """走项目二桥接服务：把消息原样交给本地 Ollama 模型。"""
        messages = payload.get("messages") or []
        try:
            temperature = float(payload.get("temperature", 0.2))
        except (TypeError, ValueError):
            temperature = 0.2
        # 解析只要结构化 JSON，输出很短，给 512 上限既能容纳推理模型的思考，也不至于拖太久
        data = self.bridge.chat(messages, temperature=temperature, max_tokens=512)
        return str(data.get("content") or "")

    def _post_openai(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """从模型回复里抠出第一个 JSON 对象（模型常会包一层说明或代码块）。"""
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------ 校验
    def _validate(
        self, data: dict[str, Any], known_categories: set[str], raw_text: str
    ) -> QueryConstraints | None:
        """把模型输出校验成合法的 ``QueryConstraints``；无有效字段时返回 ``None``。"""
        result = QueryConstraints(raw=raw_text, source="llm")

        cats = data.get("categories") or []
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats if isinstance(cats, list) else []:
            name = str(cat).strip()
            if name and (not known_categories or name in known_categories):
                if name not in result.categories:
                    result.categories.append(name)
        result.categories = result.categories[:2]

        low, high = self._clean_price(data.get("min_price")), self._clean_price(data.get("max_price"))
        if low is not None and high is not None and low > high:
            low, high = high, low
        # "8000块钱以内"这类上限表达，模型常读成 min=max=8000。这种退化区间会让
        # 过滤条件变成"≥8000 且 ≤8000"（必然空集），按语义只保留上限。
        if (
            low is not None
            and high is not None
            and low == high
            and _UPPER_BOUND_RE.search(raw_text or "")
        ):
            low = None
        result.min_price, result.max_price = low, high

        words = data.get("keywords") or []
        if isinstance(words, str):
            words = [words]
        for word in words if isinstance(words, list) else []:
            text = str(word).strip()
            if not text or len(text) > 12:
                continue
            text = normalize_keyword(text)  # HUAWEI → 华为（商品库里只有中文品牌名）
            if not text or _JUNK_KEYWORD_RE.fullmatch(text):
                continue  # 纯数字/纯符号（型号里的"60"这种）
            # 被已识别类目名完全包含的词没有额外信息，还会把候选收窄：
            # "家居装饰"已经命中了类目，模型再给"家居"/"装饰"当关键词，
            # 只会把 874 件砍到 32 件。反向不判（"华为手机"这类带品牌的短语要留着）。
            if any(text in cat for cat in result.categories):
                continue
            if text not in result.keywords:
                result.keywords.append(text)
        result.keywords = result.keywords[:3]

        if not result.has_filter:
            self.failures += 1
            return None
        result.notes.append("以上条件由大模型补充解析")
        return result

    def _clean_price(self, value: Any) -> float | None:
        """把模型给的价格转成合法数字；不合理就丢弃。"""
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number <= 0 or number > self.max_price:
            return None
        return number


def build_llm_parser(config: dict[str, Any] | None = None) -> LLMQueryParser:
    """按配置构造解析器（供 :class:`SmartSearch` 使用）。"""
    cfg = dict(config or {})
    backend = str(cfg.get("backend", "openai")).strip().lower()
    if backend not in VALID_BACKENDS:
        backend = "openai"
    return LLMQueryParser(
        enabled=bool(cfg.get("enabled", False)),
        backend=backend,
        base_url=str(cfg.get("base_url", "")),
        model=str(cfg.get("model", "")),
        api_key=str(cfg.get("api_key", "")),
        timeout=float(cfg.get("timeout", 8.0)),
        max_price=float(cfg.get("max_price", 100000)),
    )
