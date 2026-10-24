from __future__ import annotations

from src.protocol import InteractionEvent, new_trace_id, utc_now
from src.session_context import SessionContext, SessionStore


CATEGORIES = ["手机", "家用电器", "收纳用品"]


def event(text: str, *, source: str = "text", payload: dict | None = None) -> InteractionEvent:
    data = dict(payload or {})
    if text:
        data["text"] = text
    return InteractionEvent(
        trace_id=new_trace_id(),
        user_id="u1",
        session_id="s1",
        timestamp=utc_now(),
        source=source,
        payload=data,
    )


def test_gesture_maps_to_feedback_and_missing_target_is_explicit() -> None:
    context = SessionContext("u1", "s1")
    context.apply_event(
        event("", source="gesture", payload={"gesture": "thumbs_up"}),
        allowed_categories=CATEGORIES,
    )
    assert context.intent == "feedback"
    assert context.slots["action"] == "like"
    assert context.status == "needs_input"
    assert "target_item" in context.missing_slots


def test_category_conflict_and_undo() -> None:
    context = SessionContext("u1", "s1")
    context.apply_event(event("手机"), allowed_categories=CATEGORIES)
    context.apply_event(event("家用电器"), allowed_categories=CATEGORIES)
    assert context.status == "needs_clarification"
    assert context.undo()
    assert context.status == "ready"
    assert context.slots["categories"] == ["手机"]


def test_low_confidence_visual_tag_is_ignored() -> None:
    context = SessionContext("u1", "s1")
    context.apply_event(
        event(
            "",
            source="vision",
            payload={
                "vision": {
                    "objects": ["冰箱"],
                    "category_hints": ["家用电器"],
                    "confidence": 0.3,
                }
            },
        ),
        allowed_categories=CATEGORIES,
    )
    assert context.status == "needs_input"
    assert context.slots["objects"] == []


def test_negative_feedback_is_isolated_by_session() -> None:
    store = SessionStore()
    first = store.get("u1", "s1")
    first.set_feedback("X1", "dislike")
    second = store.get("u1", "s2")
    assert first.effective_feedback() == {"X1": "dislike"}
    assert second.effective_feedback() == {}
