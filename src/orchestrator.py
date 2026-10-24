"""事件 → 意图 → 推荐请求 → 召回/排序 → 证据解释的在线编排器。"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping, Sequence

from src.evidence import build_evidence_pack, explain_item
from src.protocol import (
    InteractionEvent,
    RecommendationItem,
    RecommendationRequest,
    new_trace_id,
    utc_now,
)
from src.session_context import SessionContext, SessionStore
from src import visual_intent as vi


class Orchestrator:
    """复用现有 ``SmartSearch``，补齐会话、来源追踪和证据链。"""

    def __init__(self, searcher, store: SessionStore | None = None) -> None:
        self.searcher = searcher
        # SessionStore 定义了 __len__，空存储在布尔判断中为假；必须显式判断 None。
        self.store = store if store is not None else SessionStore()

    def _item_from_profile(self, item_id: str) -> dict[str, Any] | None:
        profile = getattr(self.searcher, "profile", None)
        if profile is None:
            return None
        matched = profile[profile["item_id"].astype(str) == str(item_id)]
        if matched.empty:
            return None
        row = matched.iloc[0]
        return {
            "item_id": str(row["item_id"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "price": float(row["price"]),
        }

    @staticmethod
    def _query_from_event(event: InteractionEvent, context: SessionContext) -> str:
        text = str(
            event.payload.get("text")
            or event.payload.get("query")
            or ""
        ).strip()
        visual = event.payload.get("vision")
        if isinstance(visual, Mapping):
            visual_query = vi.compose_search_query("", dict(visual))
            if visual_query:
                text = " ".join(part for part in (text, visual_query) if part).strip()
        if not text and event.source == "vision":
            text = vi.compose_search_query("", dict(event.payload))

        if context.intent in {"similar", "explain"} and context.slots.get("target_item"):
            profile_item = None
            profile = getattr(context, "_profile_item", None)
            if isinstance(profile_item, Mapping):
                profile_item = dict(profile_item)
            if profile_item:
                text = " ".join(
                    part
                    for part in (
                        text,
                        str(profile_item.get("category") or ""),
                        str(profile_item.get("name") or ""),
                    )
                    if part
                )
        if not text:
            parts: list[str] = []
            parts.extend(str(v) for v in context.slots.get("categories") or [])
            parts.extend(str(v) for v in context.slots.get("objects") or [])
            min_price = context.slots.get("min_price")
            max_price = context.slots.get("max_price")
            if min_price is not None or max_price is not None:
                if min_price is not None and max_price is not None:
                    parts.append(f"{float(min_price):g}到{float(max_price):g}元")
                elif max_price is not None:
                    parts.append(f"{float(max_price):g}元以内")
                else:
                    parts.append(f"{float(min_price):g}元以上")
            text = " ".join(parts)
        return text

    @staticmethod
    def _sources_for_row(row: Mapping[str, Any], degraded: bool = False) -> list[str]:
        sources: list[str] = []
        if float(row.get("fb_score", 0.0) or 0.0) > 0:
            sources.append("feedback")
        if float(row.get("sim_score", 0.0) or 0.0) > 0:
            sources.append("user_vec")
        if float(row.get("pop_score", 0.0) or 0.0) > 0:
            sources.append("popular")
        if not sources:
            sources.append("fallback_popular" if degraded else "content")
        return sources

    def handle_event(
        self,
        event: InteractionEvent,
        *,
        top_k: int = 20,
        force_llm: bool = False,
        persistent_feedback: Mapping[str, str] | None = None,
        degraded_reason: str | None = None,
    ) -> dict[str, Any]:
        """处理一个已经校验过的事件，并返回兼容现有页面的结构化结果。"""
        started = time.perf_counter()
        context = self.store.get(event.user_id, event.session_id)
        context.apply_event(
            event, allowed_categories=getattr(self.searcher, "categories", ())
        )
        request = RecommendationRequest.from_event(
            event,
            text=self._query_from_event(event, context),
            top_k=max(1, min(int(top_k), 100)),
            intent=context.intent,
        )
        trace = {
            "trace_id": event.trace_id,
            "source": event.source,
            "intent": context.intent,
            "started_at": event.timestamp,
            "status": context.status,
        }

        if context.status != "ready":
            return {
                "trace_id": event.trace_id,
                "session_id": event.session_id,
                "user_id": event.user_id,
                "intent": context.intent,
                "status": context.status,
                "question": context.question(),
                "query": request.text,
                "request": request.to_dict(),
                "trace": trace,
                "notes": [context.question()],
                "results": [],
                "returned": 0,
                "context": context.to_dict(),
            }

        if context.intent == "feedback":
            state = context.set_feedback(
                context.slots.get("target_item", ""),
                context.slots.get("action", ""),
                persistent_feedback=persistent_feedback,
            )
            trace.update(
                {
                    "status": "feedback_recorded",
                    "feedback_state": state,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            return {
                "trace_id": event.trace_id,
                "session_id": event.session_id,
                "user_id": event.user_id,
                "intent": context.intent,
                "status": "feedback_recorded",
                "question": "",
                "query": request.text,
                "feedback_state": state,
                "request": request.to_dict(),
                "trace": trace,
                "notes": ["会话反馈已更新"],
                "results": [],
                "returned": 0,
                "context": context.to_dict(),
            }

        if context.intent == "explain":
            item_id = str(
                context.slots.get("target_item")
                or (context.last_items[0] if context.last_items else "")
            )
            profile_item = self._item_from_profile(item_id)
            if profile_item is not None:
                pack = build_evidence_pack(
                    profile_item,
                    constraints=context.slots,
                    recall_sources=["content"],
                )
                reason, detail = explain_item(pack)
                item = RecommendationItem(
                    item_id=profile_item["item_id"],
                    name=profile_item["name"],
                    category=profile_item["category"],
                    price=float(profile_item["price"]),
                    score=0.0,
                    reason=reason,
                    evidence=tuple(pack.evidence),
                    recall_sources=("content",),
                    trace_id=event.trace_id,
                ).to_dict()
                item["reason_detail"] = detail
                trace.update(
                    {
                        "status": "ok",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
                return {
                    "trace_id": event.trace_id,
                    "session_id": event.session_id,
                    "user_id": event.user_id,
                    "intent": context.intent,
                    "status": "ok",
                    "question": "",
                    "query": request.text,
                    "request": request.to_dict(),
                    "trace": trace,
                    "notes": [],
                    "results": [item],
                    "returned": 1,
                    "context": context.to_dict(),
                }

        feedback_map = context.effective_feedback(persistent_feedback)
        try:
            raw = self.searcher.search(
                request.text,
                username=event.user_id,
                top_n=request.top_k,
                force_llm=force_llm,
                feedback_map=feedback_map,
            )
        except TypeError:
            # 第三方测试替身或旧适配器可能还没有 feedback_map 参数。
            raw = self.searcher.search(
                request.text,
                username=event.user_id,
                top_n=request.top_k,
                force_llm=force_llm,
            )

        visual_tags: list[dict[str, Any]] = []
        for slot in ("objects", "colors"):
            for evidence in context.slot_evidence.get(slot, []):
                if evidence.get("source") == "vision":
                    visual_tags.append(
                        {
                            "value": evidence.get("value"),
                            "confidence": evidence.get("confidence", 0.0),
                        }
                    )

        items: list[dict[str, Any]] = []
        for row in raw.get("results", []):
            sources = self._sources_for_row(row, degraded=bool(degraded_reason))
            pack = build_evidence_pack(
                row,
                constraints=context.slots,
                recall_sources=sources,
                visual_tags=visual_tags,
            )
            reason, detail = explain_item(pack)
            parsed = RecommendationItem(
                item_id=str(row.get("item_id") or ""),
                name=str(row.get("name") or ""),
                category=str(row.get("category") or ""),
                price=float(row.get("price", 0.0) or 0.0),
                score=float(row.get("score", 0.0) or 0.0),
                reason=reason,
                evidence=tuple(pack.evidence),
                recall_sources=tuple(sources),
                trace_id=event.trace_id,
            ).to_dict()
            preserved = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "item_id",
                    "name",
                    "category",
                    "price",
                    "score",
                    "reason",
                    "evidence",
                    "recall_sources",
                    "trace_id",
                }
            }
            parsed.update(preserved)
            parsed["reason_detail"] = detail
            items.append(parsed)

        if items:
            context.set_last_items([item["item_id"] for item in items])
            status = "degraded" if degraded_reason else "ok"
        else:
            status = "empty"

        source_counts = Counter(
            source for item in items for source in item.get("recall_sources", [])
        )
        raw.update(
            {
                "trace_id": event.trace_id,
                "session_id": event.session_id,
                "user_id": event.user_id,
                "intent": context.intent,
                "status": status,
                "question": (
                    "没有符合硬约束的商品，请放宽品类或价格条件。"
                    if status == "empty"
                    else ""
                ),
                "degraded_reason": degraded_reason or "",
                "request": request.to_dict(),
                "context": context.to_dict(),
                "results": items,
                "returned": len(items),
                "candidate_sources": dict(source_counts),
            }
        )
        trace.update(
            {
                "status": status,
                "query": request.text,
                "returned": len(items),
                "sources": dict(source_counts),
                "degraded_reason": degraded_reason or "",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        raw["trace"] = trace
        if degraded_reason:
            raw.setdefault("notes", []).append(
                f"已降级：{degraded_reason}；推荐继续使用可用的离线路由。"
            )
        return raw

    def handle_text(
        self,
        *,
        text: str,
        user_id: str,
        session_id: str,
        trace_id: str | None = None,
        top_k: int = 20,
        force_llm: bool = False,
        persistent_feedback: Mapping[str, str] | None = None,
        degraded_reason: str | None = None,
    ) -> dict[str, Any]:
        event = InteractionEvent(
            trace_id=trace_id or new_trace_id(),
            user_id=user_id,
            session_id=session_id,
            timestamp=utc_now(),
            source="text",
            payload={"text": text},
        )
        return self.handle_event(
            event,
            top_k=top_k,
            force_llm=force_llm,
            persistent_feedback=persistent_feedback,
            degraded_reason=degraded_reason,
        )


def context_events(context: SessionContext) -> Sequence[dict[str, Any]]:
    """辅助函数：返回会话摘要副本，避免调用方直接修改运行态。"""
    return [context.to_dict()]
