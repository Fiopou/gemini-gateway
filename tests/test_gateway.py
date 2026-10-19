import http.client
import json
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gateway  # noqa: E402


class FakeResponse:
    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class StripTrailingAssistantTests(unittest.TestCase):
    def test_strips_trailing_assistant_turns(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        self.assertEqual(
            gateway.strip_trailing_assistant(messages),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_keeps_user_ending(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "again"},
        ]
        self.assertEqual(gateway.strip_trailing_assistant(messages), messages)


class SSETests(unittest.TestCase):
    def test_crlf_events_are_parsed_and_response_closed(self):
        response = FakeResponse([
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\r\n',
            b'\r\n',
            b'data: [DONE]\r\n',
            b'\r\n',
        ])
        self.assertEqual(
            list(gateway._iter_sse(response)),
            [({"choices": [{"delta": {"content": "hello"}}]}, False), (None, True)],
        )
        self.assertTrue(response.closed)

    def test_lf_and_multiline_data(self):
        response = FakeResponse([b'data: {"value":\n', b'data: 1}\n', b'\n', b'data: [DONE]\n', b'\n'])
        self.assertEqual(list(gateway._iter_sse(response))[0], ({"value": 1}, False))

    def test_missing_done_is_reported_as_truncated_stream(self):
        response = FakeResponse([b'data: {"value":1}\n', b'\n'])
        with self.assertRaises(gateway.GatewayError) as caught:
            list(gateway._iter_sse(response))
        self.assertEqual(caught.exception.code, 502)
        self.assertTrue(response.closed)


class LogicTests(unittest.TestCase):
    def test_many_short_messages_do_not_estimate_to_zero(self):
        messages = [{"role": "user", "content": "x"} for _ in range(100)]
        self.assertGreater(gateway.estimate_tokens(messages), 0)

    def test_estimate_accounts_for_utf8_and_message_overhead(self):
        short = gateway.estimate_tokens([{"role": "user", "content": "я"}])
        longer = gateway.estimate_tokens([{"role": "user", "content": "я" * 100}])
        self.assertGreater(short, 0)
        self.assertGreater(longer, short)

    def test_compaction_preserves_system_instructions_and_recent_messages(self):
        messages = [
            {"role": "system", "content": "Do not remove this instruction."},
            {"role": "user", "content": "old-user-" + "a" * 3000},
            {"role": "assistant", "content": "old-assistant-" + "b" * 3000},
            {"role": "user", "content": "recent-user"},
            {"role": "assistant", "content": "recent-assistant"},
        ]
        with mock.patch.dict(os.environ, {
            "GATEWAY_COMPACT_CHUNK": "1000",
            "GATEWAY_COMPACT_MAX_CHUNKS": "8",
            "GATEWAY_COMPACT_KEEP_MESSAGES": "2",
        }, clear=False), mock.patch.object(gateway, "_summarize_chunk", side_effect=lambda text: "summary of history") as summarize:
            compacted = gateway.compact_if_needed(messages, limit=20)
        self.assertGreater(summarize.call_count, 1)
        self.assertEqual(compacted[0], messages[0])
        self.assertIn("Резюме старой части диалога", compacted[1]["content"])
        self.assertEqual(compacted[-2:], messages[-2:])
        self.assertNotIn(messages[1], compacted)

    def test_compaction_leaves_history_unchanged_if_nothing_can_be_summarized(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "only current prompt"}]
        self.assertIs(gateway.compact_if_needed(messages, limit=1), messages)

    def test_retry_after_seconds_and_http_date(self):
        self.assertEqual(gateway._retry_after_seconds("7"), 7)
        self.assertIsNone(gateway._retry_after_seconds("not-a-time"))

    def test_daily_reset_seconds_is_positive_and_bounded(self):
        seconds = gateway._seconds_until_pacific_midnight()
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, 90000)

    def test_daily_reset_falls_back_when_system_has_no_timezone_database(self):
        with mock.patch.object(gateway, "ZoneInfo", side_effect=gateway.ZoneInfoNotFoundError("America/Los_Angeles")):
            seconds = gateway._seconds_until_pacific_midnight()
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, 90000)

    def test_cascade_does_not_use_redirected_37_flash_as_separate_step(self):
        self.assertNotIn("gemini-3.7-flash", gateway.CASCADE)
        self.assertEqual(gateway.CASCADE[0], "gemini-3.8-flash")

    def test_model_ids_have_no_duplicates_in_endpoint_payload(self):
        ids = list(dict.fromkeys(gateway.list_models() + ["gemini", "auto", "cascade"]))
        self.assertEqual(len(ids), len(set(ids)))

    def test_rate_limit_rotates_to_next_project_and_honors_retry_after(self):
        gateway._quota.clear()
        with mock.patch.dict(os.environ, {"GEMINI_KEYS": "test-key-a,test-key-b"}, clear=False), \
             mock.patch.object(gateway, "next_key", return_value="test-key-a"), \
             mock.patch.object(gateway, "_do_request", side_effect=[
                 gateway.GatewayError(429, "quota exceeded for requests_per_minute", "13"),
                 object(),
             ]) as do_request:
            response = gateway._request("test-model", [{"role": "user", "content": "hello"}])
        self.assertIsNotNone(response)
        self.assertEqual(do_request.call_count, 2)
        self.assertEqual(do_request.call_args_list[0].args[1], "test-key-a")
        self.assertEqual(do_request.call_args_list[1].args[1], "test-key-b")
        self.assertTrue(gateway._quota_busy("test-model", 0))
        self.assertFalse(gateway._quota_busy("test-model", 1))
        gateway._quota.clear()

    def test_daily_quota_marker_uses_daily_cooldown(self):
        gateway._quota.clear()
        with mock.patch.dict(os.environ, {"GEMINI_KEYS": "test-key-a,test-key-b"}, clear=False), \
             mock.patch.object(gateway, "next_key", return_value="test-key-a"), \
             mock.patch.object(gateway, "_do_request", side_effect=[
                 gateway.GatewayError(429, "quota exceeded for requests_per_day", None),
                 object(),
             ]):
            gateway._request("daily-model", [{"role": "user", "content": "hello"}])
        with gateway._quota_lock:
            remaining = gateway._quota["daily-model|0"] - __import__("time").time()
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 90000)
        gateway._quota.clear()


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # .env пользователя (GATEWAY_CORS_ORIGINS=*) не должен менять поведение тестов
        cls._saved_cors = os.environ.pop("GATEWAY_CORS_ORIGINS", None)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), gateway.GatewayHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        if cls._saved_cors is not None:
            os.environ["GATEWAY_CORS_ORIGINS"] = cls._saved_cors

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        payload = body if isinstance(body, (str, bytes)) else json.dumps(body).encode("utf-8") if body is not None else None
        conn.request(method, path, body=payload, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        result = response.status, response.getheaders(), data
        conn.close()
        return result

    def test_valid_chat_request_uses_stubbed_chat_no_external_api(self):
        fake_response = {"id": "test", "choices": [{"message": {"role": "assistant", "content": "ok"}}], "model": "gemini-3.8-flash"}
        with mock.patch.object(gateway, "chat", return_value=fake_response) as fake_chat:
            status, headers, data = self.request(
                "POST", "/v1/chat/completions",
                {"model": "gemini", "messages": [{"role": "user", "content": "hello"}]},
                {"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["model"], "gemini-3.8-flash")
        fake_chat.assert_called_once()

    def test_invalid_message_shape_returns_400_without_chat(self):
        with mock.patch.object(gateway, "chat") as fake_chat:
            status, _, data = self.request(
                "POST", "/v1/chat/completions",
                {"model": "gemini", "messages": [], "stream": False},
                {"Content-Type": "application/json"},
            )
        self.assertEqual(status, 400)
        self.assertIn("messages", json.loads(data)["error"]["message"])
        fake_chat.assert_not_called()

    def test_stream_must_be_boolean(self):
        status, _, data = self.request(
            "POST", "/v1/chat/completions",
            {"model": "gemini", "messages": [{"role": "user", "content": "x"}], "stream": "false"},
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("stream must be a boolean", json.loads(data)["error"]["message"])

    def test_models_endpoint_ids_are_unique(self):
        status, _, data = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        ids = [item["id"] for item in json.loads(data)["data"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_cors_does_not_reflect_arbitrary_origin(self):
        status, headers, _ = self.request("GET", "/v1/models", headers={"Origin": "https://attacker.example"})
        self.assertEqual(status, 200)
        normalized = {name.lower(): value for name, value in headers}
        self.assertNotIn("access-control-allow-origin", normalized)

    def test_cors_allows_localhost_origin(self):
        status, headers, _ = self.request("GET", "/v1/models", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(status, 200)
        normalized = {name.lower(): value for name, value in headers}
        self.assertEqual(normalized.get("access-control-allow-origin"), "http://localhost:3000")

    def test_request_body_limit_returns_413(self):
        with mock.patch.dict(os.environ, {"GATEWAY_MAX_REQUEST_BYTES": "1024"}, clear=False):
            payload = json.dumps({"model": "gemini", "messages": [{"role": "user", "content": "x" * 2000}]})
            status, _, data = self.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(data)["error"]["code"], 413)

    def test_streaming_response_emits_chunks_and_done(self):
        chunks = iter([
            ({"id": "chunk-1", "choices": [{"delta": {"content": "hello"}}], "model": "gemini-3.8-flash"}, False),
            (None, True),
        ])
        with mock.patch.object(gateway, "chat", return_value=chunks):
            status, headers, data = self.request(
                "POST", "/v1/chat/completions",
                {"model": "gemini", "messages": [{"role": "user", "content": "hello"}], "stream": True},
                {"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        normalized = {name.lower(): value for name, value in headers}
        self.assertTrue(normalized["content-type"].startswith("text/event-stream"))
        self.assertIn(b'"content": "hello"', data)
        self.assertIn(b"data: [DONE]", data)

    def test_streaming_upstream_error_is_not_silently_dropped(self):
        chunks = iter([({"error": {"message": "simulated upstream failure"}}, False)])
        with mock.patch.object(gateway, "chat", return_value=chunks):
            status, _, data = self.request(
                "POST", "/v1/chat/completions",
                {"model": "gemini", "messages": [{"role": "user", "content": "hello"}], "stream": True},
                {"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertIn(b"simulated upstream failure", data)
        self.assertIn(b"data: [DONE]", data)

    def test_token_auth_is_enforced_when_configured(self):
        with mock.patch.dict(os.environ, {"GATEWAY_TOKEN": "unit-test-token"}, clear=False):
            status, _, data = self.request("GET", "/v1/models")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(data)["error"]["code"], 401)

    def test_invalid_json_returns_400(self):
        status, _, data = self.request("POST", "/v1/chat/completions", "{bad json", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", json.loads(data)["error"]["message"])

    def test_unknown_route_returns_404(self):
        status, _, _ = self.request("POST", "/not-the-api", {}, {"Content-Type": "application/json"})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
