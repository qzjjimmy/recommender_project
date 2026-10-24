from __future__ import annotations

from src.orchestrator import Orchestrator
from src.protocol import InteractionEvent, new_trace_id, utc_now
from src.session_context import SessionStore


class FakeSearcher:
    categories = ["收纳用品"]

    def search(self, query, *, username=None, top_n=20, force_llm=False, feedback_map=None):
        liked = bool((feedback_map or {}).get("X1") == "like")
        return {
            "query": query,
            "constraints": {"categories": ["收纳用品"], "max_price": 20},
            "notes": [],
            "personalized": liked,
            "results": [
                {
                    "item_id": "X1",
                    "name": "蓝色收纳盒",
                    "category": "收纳用品",
                    "price": 12.5,
                    "users": 10,
                    "score": 0.9 if liked else 0.3,
                    "sim_score": 0.7,
                    "pop_score": 0.2,
                    "fb_score": 1.0 if liked else 0.0,
                    "feedback": "like" if liked else "",
                }
            ],
        }


def _event(text: str) -> InteractionEvent:
    return InteractionEvent(
        trace_id=new_trace_id(),
        user_id="u1",
        session_id="s1",
        timestamp=utc_now(),
        source="text",
        payload={"text": text, "reset_context": True},
    )


def test_orchestrator_keeps_trace_sources_and_session_feedback() -> None:
    store = SessionStore()
    orchestrator = Orchestrator(FakeSearcher(), store)
    before = orchestrator.handle_event(_event("20元以内 收纳用品"), top_k=1)
    assert before["status"] == "ok"
    item = before["results"][0]
    assert item["trace_id"] == before["trace_id"]
    assert item["recall_sources"]
    assert item["evidence"]

    store.get("u1", "s1").set_feedback("X1", "like")
    after = orchestrator.handle_event(_event("20元以内 收纳用品"), top_k=1)
    assert after["results"][0]["score"] > item["score"]
    assert after["results"][0]["feedback"] == "like"


def test_orchestrator_returns_question_when_target_is_missing() -> None:
    orchestrator = Orchestrator(FakeSearcher(), SessionStore())
    result = orchestrator.handle_event(_event("找类似商品"), top_k=1)
    assert result["status"] == "needs_input"
    assert result["question"]
    assert result["results"] == []
