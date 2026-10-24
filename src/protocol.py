"""项目三统一交互协议。

Flask 推荐服务与项目二桥接服务之间只通过 JSON 和 HTTP 字段交换数据。
这些数据结构负责把跨服务请求、推荐请求和推荐结果统一到同一条 trace 上，
同时为非法输入提供稳定的错误码。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


EVENT_SOURCES = frozenset({"text", "voice", "vision", "gesture", "system"})
_TRACE_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


class ContractError(ValueError):
    """输入不符合协议时抛出；``to_dict`` 可直接作为接口错误响应。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field_name: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.field_name = field_name
        self.trace_id = trace_id

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.message,
            "error_code": self.code,
            "message": self.message,
        }
        if self.field_name:
            payload["field"] = self.field_name
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        return payload


def new_trace_id() -> str:
    """生成跨 Flask 与多模态桥接共用的 trace ID。"""
    return uuid.uuid4().hex


def utc_now() -> str:
    """返回带时区、可排序的 ISO-8601 时间。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractError(
            "MISSING_FIELD", f"缺少必填字段：{field_name}", field_name=field_name
        )
    return text


def _validate_trace_id(trace_id: str) -> None:
    if not _TRACE_RE.fullmatch(trace_id):
        raise ContractError(
            "INVALID_TRACE_ID",
            "trace_id 必须由 8~128 个字母、数字、下划线、点、冒号或连字符组成",
            field_name="trace_id",
            trace_id=trace_id,
        )


def _validate_timestamp(timestamp: str) -> None:
    try:
        value = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "INVALID_TIMESTAMP",
            "timestamp 必须是 ISO-8601 时间",
            field_name="timestamp",
        ) from exc
    if parsed.tzinfo is None:
        raise ContractError(
            "INVALID_TIMESTAMP",
            "timestamp 必须包含时区，例如 2026-10-24T12:00:00+08:00",
            field_name="timestamp",
        )


@dataclass(frozen=True)
class InteractionEvent:
    """一次来自文字、语音、视觉、手势或系统的事件。"""

    trace_id: str
    user_id: str
    session_id: str
    timestamp: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_trace_id(self.trace_id)
        _require_text(self.user_id, "user_id")
        _require_text(self.session_id, "session_id")
        _validate_timestamp(str(self.timestamp))
        if self.source not in EVENT_SOURCES:
            raise ContractError(
                "INVALID_SOURCE",
                f"source 必须是 {', '.join(sorted(EVENT_SOURCES))} 之一",
                field_name="source",
                trace_id=self.trace_id,
            )
        if not isinstance(self.payload, Mapping):
            raise ContractError(
                "INVALID_PAYLOAD",
                "payload 必须是对象",
                field_name="payload",
                trace_id=self.trace_id,
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InteractionEvent":
        if not isinstance(data, Mapping):
            raise ContractError("INVALID_EVENT", "事件必须是 JSON 对象")
        raw_payload = data.get("payload", {})
        if not isinstance(raw_payload, Mapping):
            raise ContractError(
                "INVALID_PAYLOAD",
                "payload 必须是对象",
                field_name="payload",
                trace_id=str(data.get("trace_id") or ""),
            )
        values: dict[str, Any] = {}
        for name in ("trace_id", "user_id", "session_id", "timestamp", "source"):
            if name not in data:
                raise ContractError(
                    "MISSING_FIELD", f"缺少必填字段：{name}", field_name=name
                )
            values[name] = data[name]
        return cls(
            trace_id=_require_text(values["trace_id"], "trace_id"),
            user_id=_require_text(values["user_id"], "user_id"),
            session_id=_require_text(values["session_id"], "session_id"),
            timestamp=_require_text(values["timestamp"], "timestamp"),
            source=_require_text(values["source"], "source"),
            payload=dict(raw_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class RecommendationRequest:
    """从交互事件转换出的推荐请求，保留原始 trace。"""

    trace_id: str
    user_id: str
    session_id: str
    timestamp: str
    source: str
    text: str = ""
    top_k: int = 20
    intent: str = "discover"

    def __post_init__(self) -> None:
        _validate_trace_id(self.trace_id)
        _require_text(self.user_id, "user_id")
        _require_text(self.session_id, "session_id")
        _validate_timestamp(str(self.timestamp))
        if self.source not in EVENT_SOURCES:
            raise ContractError("INVALID_SOURCE", "source 取值非法")
        if not isinstance(self.top_k, int) or not 1 <= self.top_k <= 100:
            raise ContractError(
                "INVALID_TOP_K",
                "top_k 必须是 1~100 的整数",
                field_name="top_k",
                trace_id=self.trace_id,
            )

    @classmethod
    def from_event(
        cls,
        event: InteractionEvent,
        *,
        text: str = "",
        top_k: int = 20,
        intent: str = "discover",
    ) -> "RecommendationRequest":
        return cls(
            trace_id=event.trace_id,
            user_id=event.user_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            source=event.source,
            text=text,
            top_k=top_k,
            intent=intent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "text": self.text,
            "top_k": self.top_k,
            "intent": self.intent,
        }


@dataclass(frozen=True)
class RecommendationItem:
    """推荐适配器的统一输出，带来源与可核验证据。"""

    item_id: str
    name: str
    category: str
    price: float
    score: float
    reason: str
    evidence: tuple[Mapping[str, Any], ...]
    recall_sources: tuple[str, ...]
    trace_id: str

    def __post_init__(self) -> None:
        item_id = _require_text(self.item_id, "item_id")
        _require_text(self.trace_id, "trace_id")
        if float(self.price) < 0:
            raise ContractError("INVALID_PRICE", "price 不能为负数")
        if not self.evidence:
            raise ContractError(
                "MISSING_EVIDENCE",
                f"推荐商品 {item_id} 缺少可核验证据",
                field_name="evidence",
                trace_id=self.trace_id,
            )
        if not self.recall_sources:
            raise ContractError(
                "MISSING_RECALL_SOURCE",
                f"推荐商品 {item_id} 缺少召回来源",
                field_name="recall_sources",
                trace_id=self.trace_id,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category,
            "price": round(float(self.price), 2),
            "score": round(float(self.score), 4),
            "reason": self.reason,
            "evidence": [dict(item) for item in self.evidence],
            "recall_sources": list(self.recall_sources),
            "trace_id": self.trace_id,
        }


def contract_error_payload(
    exc: ContractError, *, trace_id: str | None = None
) -> dict[str, Any]:
    payload = exc.to_dict()
    effective_trace = trace_id or exc.trace_id
    if effective_trace:
        payload["trace_id"] = effective_trace
    return payload
