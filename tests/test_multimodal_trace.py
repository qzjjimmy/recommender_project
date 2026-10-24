from __future__ import annotations

import json
from unittest.mock import patch

from src.multimodal_client import MultimodalClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps({"ok": True}).encode("utf-8")


def test_client_propagates_trace_header() -> None:
    client = MultimodalClient(
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:8100",
            "timeout": {"vision": 1},
        }
    )
    trace_id = "trace-client-0001"
    with patch("src.multimodal_client.urllib.request.urlopen", return_value=FakeResponse()) as mocked:
        result = client.vision(
            b"\xff\xd8\xffimage",
            categories=["家用电器"],
            trace_id=trace_id,
        )
    assert result == {"ok": True}
    request = mocked.call_args.args[0]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["x-trace-id"] == trace_id
