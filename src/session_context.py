"""六类推荐意图、槽位融合与会话上下文。

会话状态只保存本轮的约束、证据、撤销快照和短期反馈。点踩与浏览不会写进
用户级长期反馈库，因此不会污染下一次会话。点赞可以作为持久偏好由调用方
单独传给推荐器。
"""
from __future__ import annotations

import copy
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src import query_parser as qp


INTENTS = ("discover", "similar", "filter", "compare", "feedback", "explain")
FEEDBACK_ACTIONS = ("like", "dislike")
COLOR_TERMS = (
    "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "黑色", "白色",
    "灰色", "粉色", "棕色", "银色", "金色", "浅绿色", "浅蓝色",
)
_ITEM_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,3}\d{4,}|\d{5,})(?![A-Za-z0-9])")
_VISUAL_CONFIDENCE = 0.6


def _event_text(event) -> str:
    payload = dict(event.payload or {})
    return str(payload.get("text") or payload.get("query") or "").strip()


def detect_intent(event) -> str:
    """先用确定性规则识别六类意图，避免简单句依赖大模型。"""
    text = _event_text(event)
    payload = dict(event.payload or {})
    action = str(payload.get("action") or "").strip().lower()
    if (
        event.source == "gesture"
        or action in FEEDBACK_ACTIONS
        or any(word in text for word in ("点赞", "点踩", "喜欢这个", "不喜欢这个", "继续浏览"))
    ):
        return "feedback"
    if any(word in text for word in ("为什么", "解释", "依据", "推荐理由", "证据")):
        return "explain"
    if any(word in text for word in ("对比", "比较", "区别", "哪个更", "哪个好")):
        return "compare"
    if any(word in text for word in ("类似", "相似", "同款", "像这个")):
        return "similar"
    if any(
        word in text
        for word in (
            "筛选", "只看", "只要", "以内", "以下", "不超过", "元以上",
            "预算", "颜色", "不要", "排除",
        )
    ):
        return "filter"
    return "discover"


def _dedupe(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _evidence(value: Any, source: str, confidence: float = 0.95) -> dict[str, Any]:
    return {
        "value": value,
        "source": source,
        "confidence": round(float(confidence), 4),
    }


def extract_slots(event, allowed_categories: Sequence[str] = ()) -> dict[str, Any]:
    """从事件文本、视觉线索和手势动作中抽取槽位，并保留证据来源。"""
    payload = dict(event.payload or {})
    text = _event_text(event)
    categories_allowed = [str(c) for c in allowed_categories if str(c).strip()]
    constraints = qp.parse_query_rules(text, categories_allowed) if text else qp.QueryConstraints()

    categories = list(constraints.categories)
    keywords = list(constraints.keywords)
    colors = [color for color in COLOR_TERMS if color in text]
    objects: list[str] = []
    scene = ""
    visual_confidence = float(payload.get("confidence", 0.8) or 0.8)

    visual = payload.get("vision")
    if isinstance(visual, Mapping):
        visual_confidence = float(visual.get("confidence", visual_confidence) or visual_confidence)
        if visual_confidence >= _VISUAL_CONFIDENCE:
            categories = _dedupe(categories + list(visual.get("category_hints") or []))
            colors = _dedupe(colors + list(visual.get("colors") or []))
            objects = _dedupe(objects + list(visual.get("objects") or []))
            scene = str(visual.get("scene") or "").strip()

    if event.source == "vision" and visual is None:
        # 兼容调用方直接把视觉字段放在 payload 的情况。
        if visual_confidence >= _VISUAL_CONFIDENCE:
            categories = _dedupe(categories + list(payload.get("category_hints") or []))
            colors = _dedupe(colors + list(payload.get("colors") or []))
            objects = _dedupe(objects + list(payload.get("objects") or []))
            scene = str(payload.get("scene") or "").strip()

    objects = _dedupe(objects)
    target_items = _dedupe(
        list(payload.get("item_ids") or [])
        + (
            [payload.get("target_item")]
            if payload.get("target_item")
            else []
        )
        + _ITEM_ID_RE.findall(text)
    )
    target_item = str(payload.get("item_id") or payload.get("target_item") or "").strip()
    if not target_item and target_items:
        target_item = target_items[0]
    if target_item and target_item not in target_items:
        target_items.insert(0, target_item)

    action = str(payload.get("action") or "").strip().lower()
    if event.source == "gesture" and not action:
        gesture = str(payload.get("gesture") or "").lower()
        action = {"thumbs_up": "like", "thumbs_down": "dislike"}.get(gesture, "")

    source = event.source if event.source in {"voice", "vision", "gesture"} else "text"
    result: dict[str, Any] = {
        "categories": categories,
        "min_price": constraints.min_price,
        "max_price": constraints.max_price,
        "keywords": keywords,
        "colors": colors,
        "objects": objects,
        "scene": scene,
        "target_item": target_item,
        "target_items": target_items,
        "action": action,
        "evidence": {},
    }
    evidence: dict[str, list[dict[str, Any]]] = {}
    for value in categories:
        evidence.setdefault("categories", []).append(_evidence(value, source, 0.99))
    for value in colors:
        confidence = visual_confidence if value in (visual or {}) or event.source == "vision" else 0.99
        evidence.setdefault("colors", []).append(_evidence(value, source, confidence))
    for value in objects:
        confidence = visual_confidence if value in list((visual or {}).get("objects") or []) else 0.9
        evidence.setdefault("objects", []).append(_evidence(value, source, confidence))
    if scene:
        evidence["scene"] = [_evidence(scene, "vision", visual_confidence)]
    if constraints.min_price is not None:
        evidence["min_price"] = [_evidence(constraints.min_price, source, 0.99)]
    if constraints.max_price is not None:
        evidence["max_price"] = [_evidence(constraints.max_price, source, 0.99)]
    if target_item:
        evidence["target_item"] = [_evidence(target_item, source, 0.99)]
    if action:
        evidence["action"] = [
            _evidence(action, event.source, float(payload.get("confidence", 0.95) or 0.95))
        ]
    result["evidence"] = evidence
    return result


def _combined_constraints(slots: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "categories": list(slots.get("categories") or []),
        "min_price": slots.get("min_price"),
        "max_price": slots.get("max_price"),
        "keywords": list(slots.get("keywords") or []),
        "colors": list(slots.get("colors") or []),
        "objects": list(slots.get("objects") or []),
        "scene": slots.get("scene") or "",
    }


@dataclass
class SessionContext:
    """一个用户会话的约束、意图和短期反馈。"""

    user_id: str
    session_id: str
    slots: dict[str, Any] = field(default_factory=dict)
    slot_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    intent: str = "discover"
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    last_items: list[str] = field(default_factory=list)
    session_feedback: dict[str, str] = field(default_factory=dict)
    browse_items: list[str] = field(default_factory=list)
    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not self.slots:
            self.slots = {
                "categories": [],
                "min_price": None,
                "max_price": None,
                "keywords": [],
                "colors": [],
                "objects": [],
                "scene": "",
                "target_item": "",
                "target_items": [],
                "action": "",
            }

    def _snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "slots": self.slots,
                "slot_evidence": self.slot_evidence,
                "intent": self.intent,
                "conflicts": self.conflicts,
                "missing_slots": self.missing_slots,
                "last_items": self.last_items,
                "session_feedback": self.session_feedback,
                "browse_items": self.browse_items,
            }
        )

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        for key, value in snapshot.items():
            setattr(self, key, copy.deepcopy(value))

    def _append_evidence(self, slot: str, values: Iterable[dict[str, Any]]) -> None:
        bucket = self.slot_evidence.setdefault(slot, [])
        seen = {
            (str(item.get("source")), str(item.get("value")))
            for item in bucket
        }
        for item in values:
            key = (str(item.get("source")), str(item.get("value")))
            if key not in seen:
                bucket.append(dict(item))
                seen.add(key)

    def apply_event(
        self, event, allowed_categories: Sequence[str] = ()
    ) -> "SessionContext":
        """合并一个事件并重新计算缺失槽位与冲突。"""
        with self._lock:
            self._history.append(self._snapshot())
            self._history = self._history[-20:]
            if bool(dict(event.payload or {}).get("reset_context")):
                self._reset_constraints()
            self.intent = detect_intent(event)
            extracted = extract_slots(event, allowed_categories)
            before = _combined_constraints(self.slots)

            incoming_categories = list(extracted["categories"])
            old_categories = list(self.slots.get("categories") or [])
            if incoming_categories:
                overlap = set(old_categories) & set(incoming_categories)
                if old_categories and not overlap:
                    self.conflicts.append(
                        {
                            "slot": "categories",
                            "previous": old_categories,
                            "incoming": incoming_categories,
                            "message": "当前会话已有不同品类，请确认要保留哪一个。",
                        }
                    )
                self.slots["categories"] = _dedupe(old_categories + incoming_categories)

            for key in ("keywords", "colors", "objects", "target_items"):
                self.slots[key] = _dedupe(
                    list(self.slots.get(key) or []) + list(extracted.get(key) or [])
                )

            for key in ("min_price", "max_price"):
                value = extracted.get(key)
                if value is not None:
                    self.slots[key] = value

            if extracted.get("scene"):
                self.slots["scene"] = str(extracted["scene"])
            if extracted.get("target_item"):
                if (
                    self.slots.get("target_item")
                    and self.slots["target_item"] != extracted["target_item"]
                    and self.intent in {"similar", "explain"}
                ):
                    self.conflicts.append(
                        {
                            "slot": "target_item",
                            "previous": self.slots["target_item"],
                            "incoming": extracted["target_item"],
                            "message": "目标商品发生变化，请确认要继续哪一个。",
                        }
                    )
                self.slots["target_item"] = str(extracted["target_item"])
            if extracted.get("action"):
                self.slots["action"] = str(extracted["action"])

            min_price = self.slots.get("min_price")
            max_price = self.slots.get("max_price")
            if (
                min_price is not None
                and max_price is not None
                and float(min_price) > float(max_price)
            ):
                self.conflicts.append(
                    {
                        "slot": "price",
                        "previous": {"min_price": before["min_price"], "max_price": before["max_price"]},
                        "incoming": {"min_price": min_price, "max_price": max_price},
                        "message": "价格下限高于上限，请确认预算范围。",
                    }
                )

            for slot, values in (extracted.get("evidence") or {}).items():
                self._append_evidence(slot, values)

            self.missing_slots = self._find_missing()
            return self

    def _reset_constraints(self) -> None:
        """清空约束与当前意图，但保留反馈、浏览和撤销历史。"""
        self.slots = {
            "categories": [],
            "min_price": None,
            "max_price": None,
            "keywords": [],
            "colors": [],
            "objects": [],
            "scene": "",
            "target_item": "",
            "target_items": [],
            "action": "",
        }
        self.slot_evidence.clear()
        self.conflicts.clear()
        self.missing_slots.clear()
        self.last_items.clear()

    def _find_missing(self) -> list[str]:
        slots = self.slots
        if self.intent in {"discover", "filter"}:
            has_query = any(
                (
                    slots.get("categories"),
                    slots.get("keywords"),
                    slots.get("objects"),
                    slots.get("colors"),
                    slots.get("scene"),
                    slots.get("min_price") is not None,
                    slots.get("max_price") is not None,
                )
            )
            return [] if has_query else ["query"]
        if self.intent == "similar":
            return [] if slots.get("target_item") or slots.get("objects") else ["target_item"]
        if self.intent == "compare":
            count = len(set(slots.get("target_items") or []))
            return [] if count >= 2 else ["target_items"]
        if self.intent == "feedback":
            missing = []
            if not slots.get("target_item"):
                missing.append("target_item")
            if slots.get("action") not in FEEDBACK_ACTIONS:
                missing.append("action")
            return missing
        if self.intent == "explain":
            return [] if slots.get("target_item") or self.last_items else ["target_item"]
        return []

    @property
    def status(self) -> str:
        if self.conflicts:
            return "needs_clarification"
        if self.missing_slots:
            return "needs_input"
        return "ready"

    def question(self) -> str:
        if self.conflicts:
            return str(self.conflicts[-1].get("message") or "条件存在冲突，请确认。")
        labels = {
            "query": "想找什么商品、品类或价格范围？",
            "target_item": "要针对哪件商品操作？请选择商品或提供商品 ID。",
            "target_items": "请提供至少两件要对比的商品。",
            "action": "要记录喜欢还是不喜欢？",
        }
        if self.missing_slots:
            return labels.get(self.missing_slots[0], "请补充必要信息。")
        return ""

    def set_feedback(
        self,
        item_id: str,
        action: str,
        *,
        persistent_feedback: Mapping[str, str] | None = None,
    ) -> str | None:
        """更新短期反馈，保证 like/dislike 互斥。"""
        if action not in FEEDBACK_ACTIONS:
            raise ValueError(f"未知反馈动作：{action}")
        item_id = str(item_id or "").strip()
        if not item_id:
            raise ValueError("item_id 不能为空")
        with self._lock:
            self._history.append(self._snapshot())
            self._history = self._history[-20:]
            current = self.session_feedback.get(item_id)
            if not current:
                current = (persistent_feedback or {}).get(item_id)
            if current == action:
                self.session_feedback.pop(item_id, None)
                final = None
            else:
                self.session_feedback[item_id] = action
                final = action
            return final

    def record_browse(self, item_id: str) -> None:
        item_id = str(item_id or "").strip()
        if not item_id:
            raise ValueError("item_id 不能为空")
        with self._lock:
            if item_id in self.browse_items:
                self.browse_items.remove(item_id)
            self.browse_items.append(item_id)
            self.browse_items = self.browse_items[-50:]

    def clear_feedback(self, action: str) -> int:
        """只清空当前会话中指定动作的反馈，返回删除条数。"""
        if action not in FEEDBACK_ACTIONS:
            raise ValueError(f"未知反馈动作：{action}")
        with self._lock:
            targets = [
                item
                for item, value in self.session_feedback.items()
                if value == action
            ]
            for item in targets:
                self.session_feedback.pop(item, None)
            return len(targets)

    def effective_feedback(
        self, persistent_feedback: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """持久点赞只提供正向偏好；负反馈仅取当前会话。"""
        result = {
            str(item): "like"
            for item, action in (persistent_feedback or {}).items()
            if str(action) == "like"
        }
        result.update(self.session_feedback)
        return result

    def undo(self) -> bool:
        with self._lock:
            if not self._history:
                return False
            self._restore(self._history.pop())
            return True

    def clear(self) -> None:
        with self._lock:
            self._history.append(self._snapshot())
            self._history = self._history[-20:]
            self.slots = {
                "categories": [],
                "min_price": None,
                "max_price": None,
                "keywords": [],
                "colors": [],
                "objects": [],
                "scene": "",
                "target_item": "",
                "target_items": [],
                "action": "",
            }
            self.slot_evidence.clear()
            self.conflicts.clear()
            self.missing_slots.clear()
            self.last_items.clear()
            self.session_feedback.clear()
            self.browse_items.clear()
            self.intent = "discover"

    def set_last_items(self, item_ids: Sequence[str]) -> None:
        with self._lock:
            self.last_items = [str(item) for item in item_ids[:20]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "intent": self.intent,
            "status": self.status,
            "slots": copy.deepcopy(self.slots),
            "slot_evidence": copy.deepcopy(self.slot_evidence),
            "conflicts": copy.deepcopy(self.conflicts),
            "missing_slots": list(self.missing_slots),
            "question": self.question(),
            "session_feedback": dict(self.session_feedback),
            "browse_items": list(self.browse_items),
            "last_items": list(self.last_items),
            "constraints": _combined_constraints(self.slots),
        }


class SessionStore:
    """线程安全的内存会话存储；以 ``(user_id, session_id)`` 隔离。"""

    def __init__(
        self,
        *,
        idle_ttl_seconds: int = 1800,
        max_sessions: int = 2000,
    ) -> None:
        self.idle_ttl_seconds = int(idle_ttl_seconds)
        self.max_sessions = int(max_sessions)
        self._contexts: dict[tuple[str, str], SessionContext] = {}
        self._last_seen: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def _evict(self, now: float) -> None:
        expired = [
            key
            for key, seen in self._last_seen.items()
            if now - seen > self.idle_ttl_seconds
        ]
        for key in expired:
            self._contexts.pop(key, None)
            self._last_seen.pop(key, None)
        if len(self._contexts) > self.max_sessions:
            oldest = sorted(self._last_seen, key=self._last_seen.get)[
                : len(self._contexts) - self.max_sessions
            ]
            for key in oldest:
                self._contexts.pop(key, None)
                self._last_seen.pop(key, None)

    def get(self, user_id: str, session_id: str) -> SessionContext:
        user_id = str(user_id or "").strip()
        session_id = str(session_id or "").strip()
        if not user_id or not session_id:
            raise ValueError("user_id 和 session_id 都不能为空")
        key = (user_id, session_id)
        with self._lock:
            now = time.time()
            self._evict(now)
            context = self._contexts.get(key)
            if context is None:
                context = SessionContext(user_id=user_id, session_id=session_id)
                self._contexts[key] = context
            self._last_seen[key] = now
            return context

    def close(self, user_id: str, session_id: str) -> bool:
        key = (str(user_id or "").strip(), str(session_id or "").strip())
        with self._lock:
            self._last_seen.pop(key, None)
            return self._contexts.pop(key, None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._contexts.clear()
            self._last_seen.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._contexts)
