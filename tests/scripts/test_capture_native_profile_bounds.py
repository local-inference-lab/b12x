#!/usr/bin/env python3
"""Focused security tests for bounded HTTP/stream reads in the native profile
capture scripts.

These tests spin up a minimal in-process HTTP server that deliberately returns
oversized, endless, redirecting, truncated, and malformed payloads and assert
that the capture scripts abort at the declared bound with a deterministic error.
Normal profiles are checked for byte-identical output.

Run:  python3 tests/scripts/test_capture_native_profile_bounds.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest
import contextlib

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(module_name: str, filename: str):
    path = _SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


vllm = _load("b12x_test_vllm_prof", "capture_vllm_native_profile.py")
sglang = _load("b12x_test_sglang_prof", "capture_sglang_native_profile.py")

_MODULES = [vllm, sglang]
_MODULE_IDS = ["vllm", "sglang"]


# ---------------------------------------------------------------------------
# Fake server helpers
# ---------------------------------------------------------------------------

class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    """A handler whose behaviour is configured per-test via class attributes."""

    mode: str = "normal_stream"
    payload_size: int = 1024
    header_count: int = 0
    status_reason: str = "OK"
    redirect_target: str = ""
    content_length_declared: int = -1  # -1 = don't override
    stall_seconds: float = 0.0  # stall before sending response body
    raw_status_line: str = ""  # override the raw status line entirely

    def log_message(self, *args: Any) -> None:
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            return self.rfile.read(length)
        return b""

    def _send_normal_models(self) -> None:
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_oversized_body_chunked(self) -> None:
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunk = b"A" * self.payload_size
        for _ in range(10):
            try:
                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_oversized_body_content_length(self) -> None:
        """Declare a large Content-Length but send a shorter body (premature EOF)."""
        self.send_response(200)
        self.send_header("Content-Length", str(self.content_length_declared))
        self.end_headers()
        if self.stall_seconds > 0:
            time.sleep(self.stall_seconds)
        short_body = b'{"ok": true}'
        try:
            self.wfile.write(short_body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_oversized_body_content_length_full(self) -> None:
        """Send a body that actually matches a large Content-Length."""
        self.send_response(200)
        self.send_header("Content-Length", str(self.content_length_declared))
        self.end_headers()
        chunk = b"A" * 65536
        sent = 0
        while sent < self.content_length_declared:
            to_send = min(chunk, self.content_length_declared - sent)
            try:
                self.wfile.write(to_send)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            sent += len(to_send)

    def _send_endless_line(self) -> None:
        self.send_response(200)
        self.end_headers()
        try:
            self.wfile.write(b"data: " + b"X" * self.payload_size)
            self.wfile.flush()
            time.sleep(5)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_endless_stream(self) -> None:
        self.send_response(200)
        self.end_headers()
        line = b'data: {"choices":[{"text":"tok"}]}\n'
        try:
            while True:
                self.wfile.write(line)
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_slow_trickle_stream(self) -> None:
        """Send one byte at a time to test watchdog deadline enforcement."""
        self.send_response(200)
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"X")
                self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_oversized_headers(self) -> None:
        self.send_response(200, message=self.status_reason)
        for i in range(self.header_count):
            self.send_header(f"X-Junk-{i}", "A" * 4096)
        self.end_headers()
        self.wfile.write(b"{}")

    def _send_oversized_status_reason(self) -> None:
        reason = "X" * self.payload_size
        self.send_response(200, message=reason)
        self.end_headers()
        self.wfile.write(b"{}")

    def _send_redirect(self) -> None:
        self.send_response(302)
        self.send_header("Location", self.redirect_target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_normal_stream(self) -> None:
        self.send_response(200)
        self.end_headers()
        lines = [
            b'data: {"choices":[{"text":"hello"}]}\n',
            b'data: {"choices":[{"text":" world"}]}\n',
            b"data: [DONE]\n",
        ]
        for line in lines:
            self.wfile.write(line)
            self.wfile.flush()

    def _send_invalid_utf8_stream(self) -> None:
        self.send_response(200)
        self.end_headers()
        # Invalid UTF-8 bytes that expand when decoded with errors="replace"
        lines = [
            b'data: {"choices":[{"text":"\xff\xfe"}]}\n',
            b'data: {"choices":[{"text":"\x80\x81\x82"}]}\n',
            b"data: [DONE]\n",
        ]
        for line in lines:
            self.wfile.write(line)
            self.wfile.flush()

    def _send_exact_limit_line(self) -> None:
        """Send a line that is exactly max_line_bytes long with no newline, then EOF."""
        self.send_response(200)
        self.end_headers()
        try:
            self.wfile.write(b"X" * self.payload_size)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_one_byte_over_line(self) -> None:
        """Send a line that is max_line_bytes + 1 long with no newline."""
        self.send_response(200)
        self.end_headers()
        try:
            self.wfile.write(b"X" * (self.payload_size + 1))
            self.wfile.flush()
            time.sleep(5)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_stalling_headers(self) -> None:
        """Send a partial header block that never terminates."""
        self.send_response(200)
        self.send_header("X-Partial", "A" * 1024)
        # Don't call end_headers - just keep the connection open
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            time.sleep(10)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_normal_models()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        self._read_body()
        handlers: dict[str, Callable[[], None]] = {
            "normal_stream": self._send_normal_stream,
            "oversized_body_chunked": self._send_oversized_body_chunked,
            "oversized_body_cl": self._send_oversized_body_content_length,
            "oversized_body_cl_full": self._send_oversized_body_content_length_full,
            "endless_line": self._send_endless_line,
            "endless_stream": self._send_endless_stream,
            "slow_trickle": self._send_slow_trickle_stream,
            "oversized_headers": self._send_oversized_headers,
            "oversized_status_reason": self._send_oversized_status_reason,
            "redirect": self._send_redirect,
            "invalid_utf8_stream": self._send_invalid_utf8_stream,
            "exact_limit_line": self._send_exact_limit_line,
            "one_byte_over_line": self._send_one_byte_over_line,
            "stalling_headers": self._send_stalling_headers,
        }
        handlers.get(self.mode, self._send_normal_stream)()


class _FakeServer:
    def __init__(self, **handler_attrs: Any) -> None:
        handler = type("Handler", (_Handler,), handler_attrs)
        self._srv = _Server(("127.0.0.1", _free_port()), handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def set_mode(self, mode: str) -> None:
        self._srv.RequestHandlerClass.mode = mode

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture()
def fake_server():
    servers: list[_FakeServer] = []

    def _make(**handler_attrs: Any) -> _FakeServer:
        srv = _FakeServer(**handler_attrs)
        servers.append(srv)
        return srv

    yield _make

    for srv in servers:
        srv.stop()


# ---------------------------------------------------------------------------
# http_json: oversized chunked body
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_oversized_chunked_body_aborts(fake_server, mod):
    srv = fake_server(mode="oversized_body_chunked", payload_size=4 * 1024 * 1024)
    headers = {"Content-Type": "application/json"}
    with pytest.raises(mod.ProfilerSizeError, match="max_response_bytes"):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_response_bytes=1024)


# ---------------------------------------------------------------------------
# http_json: oversized Content-Length body
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_oversized_content_length_aborts(fake_server, mod):
    srv = fake_server(mode="oversized_body_cl_full", content_length_declared=4 * 1024 * 1024)
    headers = {"Content-Type": "application/json"}
    with pytest.raises(mod.ProfilerSizeError, match="Content-Length"):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_response_bytes=1024)


# ---------------------------------------------------------------------------
# http_json: premature EOF (Content-Length declared longer than actual)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_premature_eof_raises(fake_server, mod):
    srv = fake_server(mode="oversized_body_cl", content_length_declared=65536)
    headers = {"Content-Type": "application/json"}
    import http.client
    with pytest.raises(http.client.IncompleteRead):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_response_bytes=65536)


# ---------------------------------------------------------------------------
# http_json: normal body byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_normal_body_byte_identical(fake_server, mod):
    srv = fake_server(mode="normal_stream")
    headers = {"Content-Type": "application/json"}
    status, body = mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_response_bytes=4096)
    assert status == 200
    expected = (
        b'data: {"choices":[{"text":"hello"}]}\n'
        b'data: {"choices":[{"text":" world"}]}\n'
        b"data: [DONE]\n"
    )
    assert body == expected


# ---------------------------------------------------------------------------
# http_json: oversized headers (raw byte enforcement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_oversized_headers_aborts(fake_server, mod):
    srv = fake_server(mode="oversized_headers", header_count=50)
    headers = {"Content-Type": "application/json"}
    with pytest.raises(mod.ProfilerSizeError, match="max bytes"):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_header_bytes=1024)


# ---------------------------------------------------------------------------
# http_json: oversized status reason (raw byte enforcement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_oversized_status_reason_aborts(fake_server, mod):
    srv = fake_server(mode="oversized_status_reason", payload_size=16 * 1024)
    headers = {"Content-Type": "application/json"}
    with pytest.raises(mod.ProfilerSizeError, match="max_line_bytes"):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_status_line_bytes=256)


# ---------------------------------------------------------------------------
# http_json: redirect rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_redirect_not_followed(fake_server, mod):
    # The server sends a 302 redirect.  Our http_json should not follow it.
    srv = fake_server(mode="redirect", redirect_target="http://evil.example.com/v1/models")
    headers = {"Content-Type": "application/json"}
    status, body = mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, max_response_bytes=4096)
    assert b"evil.example.com" in body or body == b""


# ---------------------------------------------------------------------------
# http_json: stalling headers (deadline watchdog)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_stalling_headers_deadline(fake_server, mod):
    srv = fake_server(mode="stalling_headers")
    headers = {"Content-Type": "application/json"}
    with pytest.raises((mod.ProfilerTimeoutError, Exception)):
        mod.http_json(f"{srv.base_url}/start_profile", headers=headers, payload={}, timeout_s=1.0)


# ---------------------------------------------------------------------------
# resolve_model: normal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolve_model_normal(fake_server, mod):
    srv = fake_server(mode="normal_stream")
    headers = {"Content-Type": "application/json"}
    model = mod.resolve_model(srv.base_url, headers, max_response_bytes=4096)
    assert model == "test-model"


# ---------------------------------------------------------------------------
# resolve_model: oversized body aborts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolve_model_oversized_body_aborts(fake_server, mod):
    srv = fake_server(mode="oversized_body_chunked", payload_size=4 * 1024 * 1024)

    class _GetHandler(_Handler):
        def do_GET(self):
            if self.path == "/v1/models":
                self._send_oversized_body_chunked()
            else:
                self.send_response(404)
                self.end_headers()

    srv._srv.RequestHandlerClass = _GetHandler
    headers = {"Content-Type": "application/json"}
    with pytest.raises(mod.ProfilerSizeError, match="max_response_bytes"):
        mod.resolve_model(srv.base_url, headers, max_response_bytes=1024)


# ---------------------------------------------------------------------------
# Stream: endless line (max_line_bytes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_endless_line_aborts(fake_server, mod, tmp_path):
    srv = fake_server(mode="endless_line", payload_size=2 * 1024 * 1024)
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=4096, max_stream_bytes=1024 * 1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=30)
    assert stream.error is not None
    assert "max_line_bytes" in stream.error
    assert error_path.exists()
    assert "max_line_bytes" in error_path.read_text()
    # Partial log must be deleted on overflow.
    assert not log_path.exists(), "partial log should be deleted on overflow"


# ---------------------------------------------------------------------------
# Stream: endless total bytes (max_stream_bytes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_endless_total_bytes_aborts(fake_server, mod, tmp_path):
    srv = fake_server(mode="endless_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=4096, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=30)
    assert stream.error is not None
    assert "max_stream_bytes" in stream.error
    # Partial log must be deleted on overflow.
    assert not log_path.exists(), "partial log should be deleted on overflow"


# ---------------------------------------------------------------------------
# Stream: endless time (watchdog deadline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_endless_time_aborts(fake_server, mod, tmp_path):
    srv = fake_server(mode="endless_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=64 * 1024 * 1024, max_stream_seconds=1.0,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is not None
    assert "max_stream_seconds" in stream.error
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# Stream: slow trickle (watchdog closes socket at deadline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_slow_trickle_deadline(fake_server, mod, tmp_path):
    srv = fake_server(mode="slow_trickle")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=64 * 1024 * 1024, max_stream_seconds=2.0,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is not None
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# Stream: normal byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_normal_byte_identical(fake_server, mod, tmp_path):
    srv = fake_server(mode="normal_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=1024 * 1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is None, f"unexpected error: {stream.error}"
    expected = (
        b'data: {"choices":[{"text":"hello"}]}\n'
        b'data: {"choices":[{"text":" world"}]}\n'
        b"data: [DONE]\n"
    )
    assert log_path.read_bytes() == expected
    assert stream.first_token_event.is_set()


# ---------------------------------------------------------------------------
# Stream: invalid UTF-8 does not exceed byte limit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_invalid_utf8_byte_exact(fake_server, mod, tmp_path):
    """Invalid UTF-8 bytes must not expand beyond the raw byte count."""
    srv = fake_server(mode="invalid_utf8_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024, max_stream_bytes=1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is None, f"unexpected error: {stream.error}"
    # Log file must not exceed max_stream_bytes (binary mode, no UTF-8 expansion).
    assert log_path.stat().st_size <= 1024


# ---------------------------------------------------------------------------
# Stream: exact-limit line accepted (max_line_bytes bytes, no LF, then EOF)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_exact_limit_line_accepted(fake_server, mod, tmp_path):
    srv = fake_server(mode="exact_limit_line", payload_size=100)
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=100, max_stream_bytes=1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is None, f"exact-limit line should be accepted, got: {stream.error}"
    assert log_path.read_bytes() == b"X" * 100


# ---------------------------------------------------------------------------
# Stream: one-byte-over line rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_one_byte_over_line_rejected(fake_server, mod, tmp_path):
    srv = fake_server(mode="one_byte_over_line", payload_size=100)
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=100, max_stream_bytes=1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert stream.error is not None
    assert "max_line_bytes" in stream.error
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# Stream: stale file cleanup (pre-existing log file is removed on overflow)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_stale_file_cleanup(fake_server, mod, tmp_path):
    """A pre-existing stale stream.log must be removed on overflow, not extended."""
    srv = fake_server(mode="endless_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    # Create a stale log file.
    stale_content = b"STALE DATA FROM PREVIOUS RUN\n" * 1000
    log_path.write_bytes(stale_content)
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=4096, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=30)
    assert stream.error is not None
    assert "max_stream_bytes" in stream.error
    assert not log_path.exists(), "stale log should be deleted on overflow"


# ---------------------------------------------------------------------------
# Stream: redirect rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_redirect_rejected(fake_server, mod, tmp_path):
    srv = fake_server(mode="redirect", redirect_target="http://evil.example.com/v1/completions")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=1024 * 1024, max_stream_seconds=30,
    )
    stream.start()
    assert stream.finished_event.wait(timeout=15)
    assert "non-success" in stream.error.lower() or "redirect" in stream.error.lower()
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# Stream: stop/join behavior (thread terminates after stop)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_stop_joins_thread(fake_server, mod, tmp_path):
    srv = fake_server(mode="normal_stream")
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    stream = mod.StreamRequest(
        url=f"{srv.base_url}/v1/completions", headers=headers, payload=payload,
        log_path=log_path, error_path=error_path,
        max_line_bytes=1024 * 1024, max_stream_bytes=1024 * 1024, max_stream_seconds=30,
    )
    stream.start()
    # Wait briefly then stop before the stream finishes.
    time.sleep(0.1)
    stream.stop()
    # Thread must not be alive after stop().
    assert not stream._thread.is_alive()


# ---------------------------------------------------------------------------
# Stream: invalid construction parameters rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_stream_invalid_limits_rejected(mod, tmp_path):
    headers = {"Content-Type": "application/json"}
    payload = {"model": "test", "prompt": "hi", "max_tokens": 1, "stream": True}
    log_path = tmp_path / "stream.log"
    error_path = tmp_path / "request.error.txt"
    common = dict(url="http://127.0.0.1:1/v1/completions", headers=headers, payload=payload,
                  log_path=log_path, error_path=error_path)

    with pytest.raises(ValueError, match="max_line_bytes"):
        mod.StreamRequest(**common, max_line_bytes=0, max_stream_bytes=1024, max_stream_seconds=30)
    with pytest.raises(ValueError, match="max_line_bytes"):
        mod.StreamRequest(**common, max_line_bytes=-1, max_stream_bytes=1024, max_stream_seconds=30)
    with pytest.raises(ValueError, match="max_stream_bytes"):
        mod.StreamRequest(**common, max_line_bytes=1024, max_stream_bytes=0, max_stream_seconds=30)
    with pytest.raises(ValueError, match="max_stream_seconds"):
        mod.StreamRequest(**common, max_line_bytes=1024, max_stream_bytes=1024, max_stream_seconds=0)
    with pytest.raises(ValueError, match="max_stream_seconds"):
        mod.StreamRequest(**common, max_line_bytes=1024, max_stream_bytes=1024, max_stream_seconds=float("nan"))
    with pytest.raises(ValueError, match="max_stream_seconds"):
        mod.StreamRequest(**common, max_line_bytes=1024, max_stream_bytes=1024, max_stream_seconds=float("inf"))


# ---------------------------------------------------------------------------
# http_json: invalid limits rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_http_json_invalid_limits_rejected(mod):
    headers = {"Content-Type": "application/json"}
    with pytest.raises(ValueError, match="max_response_bytes"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, max_response_bytes=0)
    with pytest.raises(ValueError, match="max_header_bytes"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, max_header_bytes=-1)
    with pytest.raises(ValueError, match="max_status_line_bytes"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, max_status_line_bytes=0)
    with pytest.raises(ValueError, match="timeout_s"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, timeout_s=float("nan"))
    with pytest.raises(ValueError, match="timeout_s"):
        mod.http_json("http://127.0.0.1:1/", headers=headers, payload=None, timeout_s=float("inf"))


# ---------------------------------------------------------------------------
# CLI: invalid limit values rejected by argparse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_cli_rejects_invalid_limits(mod, monkeypatch):
    base = ["--base-url", "http://127.0.0.1:1"]
    for arg in ["--max-line-bytes", "--max-response-bytes", "--max-stream-bytes",
                "--max-status-line-bytes", "--max-header-bytes"]:
        for bad in ["0", "-1", "abc"]:
            monkeypatch.setattr(sys, "argv", ["prog"] + base + [arg, bad])
            with pytest.raises(SystemExit):
                mod.parse_args()
    for arg in ["--max-stream-seconds"]:
        for bad in ["0", "-1", "nan", "inf", "abc"]:
            monkeypatch.setattr(sys, "argv", ["prog"] + base + [arg, bad])
            with pytest.raises(SystemExit):
                mod.parse_args()


# ---------------------------------------------------------------------------
# Shell wrapper: env validation
# ---------------------------------------------------------------------------

def test_shell_rejects_invalid_env():
    sh = _SCRIPTS / "capture_sglang_native_profile.sh"
    # Empty string defaults to built-in value via :=, so only test non-empty bad values.
    for bad in ["0", "-1", "abc"]:
        result = subprocess.run(
            ["bash", str(sh), "start"],
            env={**os.environ, "B12X_PROFILE_MAX_FILESIZE": bad},
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert "positive integer" in result.stderr or "Error" in result.stderr

    for bad in ["0", "-1", "abc"]:
        result = subprocess.run(
            ["bash", str(sh), "start"],
            env={**os.environ, "B12X_PROFILE_MAX_TIME": bad},
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert "positive integer" in result.stderr or "Error" in result.stderr

# ---------------------------------------------------------------------------
# Shell wrapper: normal byte-identical output
# ---------------------------------------------------------------------------

def test_shell_normal_output(fake_server):
    """Shell wrapper produces byte-identical output for a normal response."""
    srv = fake_server(mode="normal_stream")
    sh = _SCRIPTS / "capture_sglang_native_profile.sh"
    result = subprocess.run(
        ["bash", str(sh), "start", "--base-url", srv.base_url, "--mode", "all"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    expected = (
        'data: {"choices":[{"text":"hello"}]}\n'
        'data: {"choices":[{"text":" world"}]}\n'
        'data: [DONE]\n\n'
    )
    assert result.stdout == expected


# ---------------------------------------------------------------------------
# Shell wrapper: oversized body rejected
# ---------------------------------------------------------------------------

def test_shell_oversized_body_rejected(fake_server):
    srv = fake_server(mode="oversized_body_chunked", payload_size=4 * 1024 * 1024)
    sh = _SCRIPTS / "capture_sglang_native_profile.sh"
    result = subprocess.run(
        ["bash", str(sh), "start", "--base-url", srv.base_url, "--mode", "all"],
        env={**os.environ, "B12X_PROFILE_MAX_FILESIZE": "1024"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
