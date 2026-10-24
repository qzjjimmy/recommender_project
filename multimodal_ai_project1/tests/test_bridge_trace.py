from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from bridge_server import BridgeHandler


class BridgeTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_error_json_and_header_preserve_trace(self) -> None:
        trace_id = "trace-bridge-0001"
        request = urllib.request.Request(
            f"{self.base_url}/vision",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-Trace-ID": trace_id},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=3)
        response = raised.exception
        body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(body["trace_id"], trace_id)
        self.assertEqual(body["error_code"], "MISSING_IMAGE")
        self.assertEqual(response.headers.get("X-Trace-ID"), trace_id)


if __name__ == "__main__":
    unittest.main()
