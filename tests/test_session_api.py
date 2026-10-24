from __future__ import annotations

from web.app import PROFILE_INDEX, SESSION_STORE, app


def _logged_client():
    client = app.test_client()
    with client.session_transaction() as session:
        session["username"] = "qzj"
    return client


def test_session_feedback_is_not_shared_across_login_sessions() -> None:
    SESSION_STORE.clear_all()
    first = _logged_client()
    second = _logged_client()
    item_id = next(iter(PROFILE_INDEX))

    response = first.post(
        "/feedback",
        json={"item_id": item_id, "action": "dislike"},
        headers={"X-Trace-ID": "trace-session-api-01"},
    )
    assert response.status_code == 200
    assert response.get_json()["state"] == "dislike"
    assert response.headers["X-Trace-ID"] == "trace-session-api-01"
    assert first.get("/api/disliked").get_json()["count"] == 1
    assert second.get("/api/disliked").get_json()["count"] == 0


def test_session_context_undo_and_clear_routes() -> None:
    SESSION_STORE.clear_all()
    client = _logged_client()
    search = client.get("/api/search?q=%E6%94%B6%E7%BA%B3%E7%94%A8%E5%93%81&top_n=1")
    assert search.status_code == 200
    context = client.get("/api/session/context").get_json()
    assert context["intent"] == "discover"
    assert context["status"] == "ready"
    cleared = client.post("/api/session/clear").get_json()
    assert cleared["slots"]["categories"] == []
