from __future__ import annotations

import pytest

from src.protocol import (
    ContractError,
    InteractionEvent,
    RecommendationItem,
    new_trace_id,
    utc_now,
)


def event_data() -> dict:
    return {
        "trace_id": new_trace_id(),
        "user_id": "u1",
        "session_id": "s1",
        "timestamp": utc_now(),
        "source": "text",
        "payload": {"text": "手机"},
    }


def test_event_roundtrip_preserves_trace() -> None:
    raw = event_data()
    event = InteractionEvent.from_dict(raw)
    assert event.to_dict() == raw
    assert event.trace_id == raw["trace_id"]


@pytest.mark.parametrize("field", ["trace_id", "user_id", "session_id", "timestamp", "source"])
def test_event_rejects_missing_required_field(field: str) -> None:
    raw = event_data()
    raw.pop(field)
    with pytest.raises(ContractError) as error:
        InteractionEvent.from_dict(raw)
    assert error.value.code == "MISSING_FIELD"
    assert error.value.field_name == field


def test_event_rejects_non_object_payload() -> None:
    raw = event_data()
    raw["payload"] = []
    with pytest.raises(ContractError) as error:
        InteractionEvent.from_dict(raw)
    assert error.value.code == "INVALID_PAYLOAD"


def test_recommendation_item_requires_recall_source_and_evidence() -> None:
    with pytest.raises(ContractError) as error:
        RecommendationItem(
            item_id="X1",
            name="测试",
            category="测试",
            price=1.0,
            score=0.1,
            reason="测试",
            evidence=(),
            recall_sources=("popular",),
            trace_id=new_trace_id(),
        )
    assert error.value.code == "MISSING_EVIDENCE"
