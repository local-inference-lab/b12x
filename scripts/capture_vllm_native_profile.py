#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional
import contextlib


# --- configurable hard maxima for profiler HTTP/stream reads -----------------
_DEFAULT_MAX_STATUS_LINE_BYTES = 8 * 1024
_DEFAULT_MAX_HEADER_BYTES = 256 * 1024
_DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_LINE_BYTES = 1024 * 1024
_DEFAULT_MAX_STREAM_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_STREAM_SECONDS = 3600

_READ_CHUNK = 65536


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProfilerSizeError(Exception):
    """Raised when an HTTP response or stream exceeds a configured byte bound."""


class ProfilerTimeoutError(Exception):
    """Raised when a profiler operation exceeds its absolute deadline."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_positive_finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    fval = float(value)
    if fval <= 0 or math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return fval


# ---------------------------------------------------------------------------
# Deadline + watchdog
# ---------------------------------------------------------------------------

class _Deadline:
    """Tracks an absolute monotonic deadline."""

    def __init__(self, duration_s: float) -> None:
        _validate_positive_finite(duration_s, "duration_s")
        self._deadline = time.monotonic() + duration_s

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline


class _SocketWatchdog:
    """Closes a socket at an absolute deadline to enforce wall-clock limits."""

    def __init__(self, sock: socket.socket, deadline: _Deadline) -> None:
        self._sock = sock
        self._deadline = deadline
        self._timer: Optional[threading.Timer] = None
        self.fired = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        remaining = self._deadline.remaining()
        if remaining <= 0:
            self._fire()
            return
        self._timer = threading.Timer(remaining, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired.set()
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._sock.close()

    def cancel(self) -> None:
        with self._lock:
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1.0)

    @property
    def is_fired(self) -> bool:
        return self.fired.is_set()


# ---------------------------------------------------------------------------
# Bounded raw socket reader (status line + headers, no buffered read-ahead)
# ---------------------------------------------------------------------------

class _RawSocketReader:
    """Reads directly from a socket via recv(), enforcing per-line and
    aggregate raw-byte limits plus a monotonic deadline on every recv.

    Unlike BufferedReader, this never reads ahead beyond the requested bytes,
    so accounting is exact.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        max_line_bytes: int,
        max_total_bytes: int,
        deadline: _Deadline,
    ) -> None:
        self._sock = sock
        self._max_line_bytes = max_line_bytes
        self._max_total_bytes = max_total_bytes
        self._total = 0
        self._deadline = deadline
        self._buffer = bytearray()
        self._closed = False

    def _recv_with_deadline(self, bufsize: int) -> bytes:
        remaining = self._deadline.remaining()
        if remaining <= 0:
            raise ProfilerTimeoutError("profiler deadline exceeded")
        self._sock.settimeout(remaining)
        try:
            data = self._sock.recv(bufsize)
        except (socket.timeout, TimeoutError) as exc:
            if self._deadline.expired():
                raise ProfilerTimeoutError("profiler deadline exceeded") from exc
            raise
        return data

    def _account(self, n: int) -> None:
        self._total += n
        if self._total > self._max_total_bytes:
            raise ProfilerSizeError(
                f"HTTP read exceeded max bytes={self._max_total_bytes}"
            )

    def readline(self, size: int = -1) -> bytes:
        """Read one line (terminated by \\n) from the socket, byte by byte.

        Never reads beyond the newline.  Enforces both per-line and aggregate
        byte limits with exact accounting.
        """
        if size < 0 or size > self._max_line_bytes:
            size = self._max_line_bytes + 1
        line = bytearray()
        while len(line) < size:
            if self._buffer:
                idx = self._buffer.find(b"\n")
                if idx >= 0:
                    line.extend(self._buffer[:idx + 1])
                    del self._buffer[:idx + 1]
                    self._account(len(line))
                    if len(line) > self._max_line_bytes:
                        raise ProfilerSizeError(
                            f"line exceeded max_line_bytes={self._max_line_bytes}"
                        )
                    return bytes(line)
                else:
                    take = min(len(self._buffer), size - len(line))
                    line.extend(self._buffer[:take])
                    del self._buffer[:take]
                    if len(line) >= size:
                        self._account(len(line))
                        if len(line) > self._max_line_bytes:
                            raise ProfilerSizeError(
                                f"line exceeded max_line_bytes={self._max_line_bytes}"
                            )
                        return bytes(line)
                    continue
            data = self._recv_with_deadline(1)
            if not data:
                self._account(len(line))
                return bytes(line)
            self._buffer.extend(data)
        self._account(len(line))
        if len(line) > self._max_line_bytes:
            raise ProfilerSizeError(
                f"line exceeded max_line_bytes={self._max_line_bytes}"
            )
        return bytes(line)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._max_total_bytes - self._total + 1
        data = bytearray()
        while len(data) < size:
            if self._buffer:
                take = min(len(self._buffer), size - len(data))
                data.extend(self._buffer[:take])
                del self._buffer[:take]
                continue
            chunk = self._recv_with_deadline(min(size - len(data), 4096))
            if not chunk:
                break
            data.extend(chunk)
        self._account(len(data))
        return bytes(data)

    def close(self) -> None:
        self._closed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


# ---------------------------------------------------------------------------
# Bounded HTTP response (enforces limits during status + header parsing)
# ---------------------------------------------------------------------------

class _BoundedHTTPResponse(http.client.HTTPResponse):
    """HTTPResponse that enforces configurable raw-byte limits while parsing
    the status line and headers, including interim 1xx responses."""

    _bounds: Optional[dict[str, Any]] = None
    _sock_ref: Optional[socket.socket] = None

    def begin(self) -> None:
        if self._bounds is None or self._sock_ref is None:  # pragma: no cover
            super().begin()
            return
        orig_fp = self.fp
        self.fp = _RawSocketReader(
            self._sock_ref,
            max_line_bytes=self._bounds["max_status_line_bytes"],
            max_total_bytes=self._bounds["max_header_bytes"],
            deadline=self._bounds["deadline"],
        )
        try:
            super().begin()
            # Validate Content-Length grammar: must be a non-negative integer.
            cl = self.headers.get("Content-Length")
            if cl is not None:
                try:
                    cl_int = int(cl)
                except ValueError:
                    raise http.client.HTTPException(f"invalid Content-Length: {cl!r}") from None
                if cl_int < 0:
                    raise http.client.HTTPException(f"negative Content-Length: {cl!r}")
        finally:
            self.fp = orig_fp

# ---------------------------------------------------------------------------
# Bounded DNS resolution
# ---------------------------------------------------------------------------
def _bounded_getaddrinfo(
    host: str, port: int, deadline: _Deadline
) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
    """Resolve host:port in a subprocess, bounded by the absolute deadline.

    A subprocess is truly killable: if the deadline expires, we terminate
    it and raise ProfilerTimeoutError.  No thread or resource is retained.
    """
    remaining = max(deadline.remaining(), 0.001)
    code = (
        "import json,socket,sys;"
        "rows=socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),0,socket.SOCK_STREAM);"
        "json.dump([[int(f),int(t),p,c,list(a)] for f,t,p,c,a in rows],sys.stdout)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, host, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, _ = proc.communicate(timeout=remaining)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise ProfilerTimeoutError("DNS resolution exceeded deadline") from None
    if proc.returncode != 0:
        raise OSError(f"DNS resolution failed for {host}:{port} (exit {proc.returncode})")
    rows = json.loads(stdout)
    return [(int(f), int(t), int(p), str(c), tuple(a)) for f, t, p, c, a in rows]


# ---------------------------------------------------------------------------
# Bounded HTTP connection (deadline on connect/send/getresponse)
# ---------------------------------------------------------------------------

class _BoundedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that enforces an absolute deadline on DNS resolution,
    every socket operation, and produces _BoundedHTTPResponse objects."""

    _deadline: Optional[_Deadline] = None
    _response_bounds: Optional[dict[str, Any]] = None

    def _check_deadline(self) -> None:
        if self._deadline is not None:
            remaining = self._deadline.remaining()
            if remaining <= 0:
                raise ProfilerTimeoutError("profiler deadline exceeded")
            if self.sock is not None:
                self.sock.settimeout(remaining)

    def connect(self) -> None:
        assert self._deadline is not None
        remaining = self._deadline.remaining()
        if remaining <= 0:
            raise ProfilerTimeoutError("profiler deadline exceeded before connect")

        # Bounded DNS resolution.
        addrs = _bounded_getaddrinfo(self.host, self.port, self._deadline)

        # Try each address using freshly computed remaining time.
        for af, socktype, proto, _canonname, sa in addrs:
            remaining = self._deadline.remaining()
            if remaining <= 0:
                raise ProfilerTimeoutError("profiler deadline exceeded during connect")
            try:
                self.sock = socket.socket(af, socktype, proto)
                self.sock.settimeout(remaining)
                self.sock.connect(sa)
                if self._deadline is not None:
                    self.sock.settimeout(self._deadline.remaining())
                break
            except OSError:
                if self.sock is not None:
                    with contextlib.suppress(OSError):
                        self.sock.close()
                    self.sock = None
                continue
        else:
            raise OSError(f"could not connect to {self.host}:{self.port}")

    def send(self, data: bytes) -> None:
        self._check_deadline()
        super().send(data)

    def getresponse(self) -> _BoundedHTTPResponse:
        self._check_deadline()
        if self.debuglevel > 0:
            response = self._response_class(self.sock, self.debuglevel, method=self._method)
        else:
            response = self._response_class(self.sock, method=self._method)
        if self._response_bounds is not None:
            response._bounds = self._response_bounds
            response._sock_ref = self.sock
        try:
            response.begin()
        except BaseException:
            response.close()
            raise
        assert response.will_close != http.client._UNKNOWN
        self._HTTPConnection__state = http.client._CS_IDLE  # type: ignore[attr-defined]
        if response.will_close:
            self.sock = None
        else:
            self._HTTPConnection__response = response  # type: ignore[attr-defined]
        return response


class _BoundedHTTPSConnection(http.client.HTTPSConnection):
    _deadline: Optional[_Deadline] = None
    _response_bounds: Optional[dict[str, Any]] = None
    _response_class = _BoundedHTTPResponse

    def _check_deadline(self) -> None:
        if self._deadline is not None:
            remaining = self._deadline.remaining()
            if remaining <= 0:
                raise ProfilerTimeoutError("profiler deadline exceeded")
            if self.sock is not None:
                self.sock.settimeout(remaining)

    def connect(self) -> None:
        assert self._deadline is not None
        remaining = self._deadline.remaining()
        if remaining <= 0:
            raise ProfilerTimeoutError("profiler deadline exceeded before connect")

        # Bounded DNS resolution.
        addrs = _bounded_getaddrinfo(self.host, self.port, self._deadline)

        # Try each address using freshly computed remaining time.
        for af, socktype, proto, _canonname, sa in addrs:
            remaining = self._deadline.remaining()
            if remaining <= 0:
                raise ProfilerTimeoutError("profiler deadline exceeded during connect")
            try:
                self.sock = socket.socket(af, socktype, proto)
                self.sock.settimeout(remaining)
                self.sock.connect(sa)
                break
            except OSError:
                if self.sock is not None:
                    with contextlib.suppress(OSError):
                        self.sock.close()
                    self.sock = None
                continue
        else:
            raise OSError(f"could not connect to {self.host}:{self.port}")

        # Reset timeout before TLS handshake using remaining absolute deadline.
        if self._deadline is not None and self.sock is not None:
            self.sock.settimeout(self._deadline.remaining())

        # Perform TLS handshake (from HTTPSConnection.connect, minus the TCP part).
        if self._tunnel_host:
            self._tunnel()
        server_hostname = self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)
        if self._deadline is not None and self.sock is not None:
            self.sock.settimeout(self._deadline.remaining())

    def send(self, data: bytes) -> None:
        self._check_deadline()
        super().send(data)

    def getresponse(self) -> _BoundedHTTPResponse:
        self._check_deadline()
        if self.debuglevel > 0:
            response = self._response_class(self.sock, self.debuglevel, method=self._method)
        else:
            response = self._response_class(self.sock, method=self._method)
        if self._response_bounds is not None:
            response._bounds = self._response_bounds
            response._sock_ref = self.sock
        try:
            response.begin()
        except BaseException:
            response.close()
            raise
        assert response.will_close != http.client._UNKNOWN
        self._HTTPConnection__state = http.client._CS_IDLE  # type: ignore[attr-defined]
        if response.will_close:
            self.sock = None
        else:
            self._HTTPConnection__response = response  # type: ignore[attr-defined]
        return response


_BoundedHTTPConnection._response_class = _BoundedHTTPResponse


def _create_connection(
    parsed: urllib.parse.ParseResult,
    *,
    deadline: _Deadline,
    response_bounds: dict[str, Any],
) -> http.client.HTTPConnection:
    """Create a bounded HTTP(S) connection.  No redirects are followed.

    Requires an absolute URL with scheme exactly 'http' or 'https' and a
    non-empty hostname.
    """
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http or https, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL must include a non-empty hostname")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        cls: type = _BoundedHTTPSConnection
    else:
        cls = _BoundedHTTPConnection
    conn = cls(host, port, timeout=deadline.remaining())
    conn._deadline = deadline
    conn._response_bounds = response_bounds
    return conn


# ---------------------------------------------------------------------------
# Bounded body reader
# ---------------------------------------------------------------------------

def _read_bounded_body(
    response: http.client.HTTPResponse,
    max_bytes: int,
    deadline: _Deadline,
    sock: socket.socket,
    watchdog: _SocketWatchdog,
) -> bytes:
    """Read a response body incrementally as raw bytes, aborting on byte or
    deadline overflow.  Never decodes, so persistence is byte-exact.

    Reads at most min(chunk_size, remaining_cap + 1) per iteration so peak
    allocation never exceeds max_bytes + 1.  Detects premature EOF for
    Content-Length responses.
    """
    buf = bytearray()
    declared_length = response.length  # Content-Length or None (chunked)
    try:
        while True:
            remaining = deadline.remaining()
            if remaining <= 0:
                raise ProfilerTimeoutError("profiler deadline exceeded during body read")
            sock.settimeout(remaining)
            # Read at most remaining_cap + 1 so we can detect overflow without
            # allocating beyond max_bytes + 1.
            to_read = min(_READ_CHUNK, max_bytes - len(buf) + 1)
            if to_read <= 0:
                raise ProfilerSizeError(
                    f"response body exceeded max_response_bytes={max_bytes}"
                )
            chunk = response.read(to_read)
            if not chunk:
                if watchdog.is_fired or deadline.expired():
                    raise ProfilerTimeoutError("profiler deadline exceeded during body read")
                break
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise ProfilerSizeError(
                    f"response body exceeded max_response_bytes={max_bytes}"
                )
    except (ProfilerSizeError, ProfilerTimeoutError):
        raise
    except (socket.timeout, TimeoutError) as exc:
        if deadline.expired() or watchdog.is_fired:
            raise ProfilerTimeoutError("profiler deadline exceeded during body read") from exc
        raise
    except OSError as exc:
        if deadline.expired() or watchdog.is_fired:
            raise ProfilerTimeoutError("profiler deadline exceeded during body read") from exc
        raise
    except http.client.IncompleteRead as exc:
        if deadline.expired() or watchdog.is_fired:
            raise ProfilerTimeoutError("profiler deadline exceeded during body read") from exc
        raise

    # Detect premature EOF for Content-Length responses.
    if declared_length is not None and len(buf) < declared_length:
        raise http.client.IncompleteRead(bytes(buf), declared_length)

    return bytes(buf)


# ---------------------------------------------------------------------------
# HTTP JSON helper (no redirects, bounded reads)
# ---------------------------------------------------------------------------

def http_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: Optional[dict[str, Any]] = None,
    timeout_s: float = 3600,
    max_status_line_bytes: int = _DEFAULT_MAX_STATUS_LINE_BYTES,
    max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
) -> tuple[int, bytes]:
    """Make an HTTP request with bounded reads and an absolute deadline.

    Redirects are never followed.  Non-2xx responses are returned with their
    status code and bounded raw bytes rather than raising HTTPError.
    """
    _validate_positive_finite(timeout_s, "timeout_s")
    _validate_positive_int(max_status_line_bytes, "max_status_line_bytes")
    _validate_positive_int(max_header_bytes, "max_header_bytes")
    _validate_positive_int(max_response_bytes, "max_response_bytes")

    deadline = _Deadline(timeout_s)
    parsed = urllib.parse.urlparse(url)
    method = "POST" if payload is not None else "GET"
    data = None if payload is None else json.dumps(payload).encode("utf-8")

    response_bounds = {
        "max_status_line_bytes": max_status_line_bytes,
        "max_header_bytes": max_header_bytes,
        "deadline": deadline,
    }

    conn = _create_connection(parsed, deadline=deadline, response_bounds=response_bounds)
    watchdog: Optional[_SocketWatchdog] = None
    response: Optional[http.client.HTTPResponse] = None
    sock_ref: Optional[socket.socket] = None
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn.request(method, path, body=data, headers=headers or {})
        # Save socket reference before getresponse() may detach it.
        sock_ref = conn.sock
        if sock_ref is not None:
            watchdog = _SocketWatchdog(sock_ref, deadline)
            watchdog.start()
        response = conn.getresponse()

        # Fast rejection on Content-Length.
        cl = response.getheader("Content-Length")
        if cl is not None:
            try:
                cl_int = int(cl)
            except ValueError:
                cl_int = None
            if cl_int is not None and cl_int > max_response_bytes:
                raise ProfilerSizeError(
                    f"Content-Length {cl_int} exceeded max_response_bytes={max_response_bytes}"
                )

        body = _read_bounded_body(response, max_response_bytes, deadline, sock_ref, watchdog)
        return response.status, body
    except (ProfilerSizeError, ProfilerTimeoutError):
        raise
    except (socket.timeout, TimeoutError) as exc:
        if deadline.expired() or (watchdog is not None and watchdog.is_fired):
            raise ProfilerTimeoutError("profiler deadline exceeded") from exc
        raise
    except OSError as exc:
        if deadline.expired() or (watchdog is not None and watchdog.is_fired):
            raise ProfilerTimeoutError("profiler deadline exceeded") from exc
        raise
    finally:
        if response is not None:
            with contextlib.suppress(Exception):
                response.close()
        if watchdog is not None:
            watchdog.cancel()
        if sock_ref is not None:
            with contextlib.suppress(Exception):
                sock_ref.close()
        with contextlib.suppress(Exception):
            conn.close()


def resolve_model(
    base_url: str,
    headers: dict[str, str],
    *,
    timeout_s: float = 30,
    max_status_line_bytes: int = _DEFAULT_MAX_STATUS_LINE_BYTES,
    max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
) -> str:
    status, body = http_json(
        f"{base_url}/v1/models",
        headers=headers,
        payload=None,
        timeout_s=timeout_s,
        max_status_line_bytes=max_status_line_bytes,
        max_header_bytes=max_header_bytes,
        max_response_bytes=max_response_bytes,
    )
    if status != 200:
        raise RuntimeError(f"/v1/models failed with status {status}: {body.decode('utf-8', errors='replace').strip()}")
    payload = json.loads(body.decode("utf-8"))
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no models")
    return models[0]["id"]


# ---------------------------------------------------------------------------
# Streaming request
# ---------------------------------------------------------------------------

class StreamRequest:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        log_path: Path,
        error_path: Path,
        timeout_s: int = 3600,
        max_line_bytes: int = _DEFAULT_MAX_LINE_BYTES,
        max_stream_bytes: int = _DEFAULT_MAX_STREAM_BYTES,
        max_stream_seconds: float = _DEFAULT_MAX_STREAM_SECONDS,
        max_status_line_bytes: int = _DEFAULT_MAX_STATUS_LINE_BYTES,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    ) -> None:
        _validate_positive_int(max_line_bytes, "max_line_bytes")
        _validate_positive_int(max_stream_bytes, "max_stream_bytes")
        _validate_positive_finite(max_stream_seconds, "max_stream_seconds")
        _validate_positive_int(max_status_line_bytes, "max_status_line_bytes")
        _validate_positive_int(max_header_bytes, "max_header_bytes")

        self.url = url
        self.headers = headers
        self.payload = payload
        self.log_path = log_path
        self.error_path = error_path
        self.timeout_s = timeout_s
        self.max_line_bytes = max_line_bytes
        self.max_stream_bytes = max_stream_bytes
        self.max_stream_seconds = max_stream_seconds
        self.max_status_line_bytes = max_status_line_bytes
        self.max_header_bytes = max_header_bytes
        self.first_token_event = threading.Event()
        self.finished_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._conn: Optional[http.client.HTTPConnection] = None
        self._response: Optional[http.client.HTTPResponse] = None
        self._watchdog: Optional[_SocketWatchdog] = None
        self._sock_ref: Optional[socket.socket] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Close the underlying socket to unblock any blocking read.
        sock = self._sock_ref
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()
        if self._response is not None:
            with contextlib.suppress(Exception):
                self._response.close()
        if self._watchdog is not None:
            self._watchdog.cancel()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("stream worker thread did not terminate after stop()")

    def wait_for_first_token(self, timeout_s: float) -> bool:
        return self.first_token_event.wait(timeout_s)

    def _record_error(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        with contextlib.suppress(Exception):
            self.error_path.write_text(self.error + "\n", encoding="utf-8")

    def _delete_partial_log(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.log_path.unlink()

    def _run(self) -> None:
        data = json.dumps(self.payload).encode("utf-8")
        parsed = urllib.parse.urlparse(self.url)
        deadline = _Deadline(self.max_stream_seconds)

        response_bounds = {
            "max_status_line_bytes": self.max_status_line_bytes,
            "max_header_bytes": self.max_header_bytes,
            "deadline": deadline,
        }

        conn = _create_connection(parsed, deadline=deadline, response_bounds=response_bounds)
        self._conn = conn
        log_file = None
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn.request("POST", path, body=data, headers=self.headers)
            # Save socket reference before getresponse() may detach it.
            sock_ref = conn.sock
            self._sock_ref = sock_ref
            if sock_ref is not None:
                self._watchdog = _SocketWatchdog(sock_ref, deadline)
                self._watchdog.start()
            response = conn.getresponse()
            self._response = response

            # Reject non-2xx (including redirects, 4xx, 5xx).
            if not (200 <= response.status < 300):
                raise ProfilerSizeError(
                    f"server returned non-success status {response.status}"
                )

            # Truncate/remove stale log before any stream I/O.
            log_file = self.log_path.open("wb")
            written = 0
            while not self._stop_event.is_set():
                remaining = deadline.remaining()
                if remaining <= 0:
                    raise ProfilerTimeoutError(
                        f"stream exceeded max_stream_seconds={self.max_stream_seconds}"
                    )
                sock_ref.settimeout(remaining)

                # Read min(max_line_bytes+1, remaining_stream+1) to bound peak read.
                read_size = min(self.max_line_bytes + 1, self.max_stream_bytes - written + 1)
                if read_size <= 0:
                    raise ProfilerSizeError(
                        f"stream log exceeded max_stream_bytes={self.max_stream_bytes}"
                    )
                line = response.readline(read_size)
                if not line:
                    if deadline.expired() or (self._watchdog is not None and self._watchdog.is_fired):
                        raise ProfilerTimeoutError(
                            f"stream exceeded max_stream_seconds={self.max_stream_seconds}"
                        )
                    # Check premature Content-Length EOF.
                    if response.length is not None and response.length > 0:
                        raise http.client.IncompleteRead(b"", response.length)
                    break
                if len(line) > self.max_line_bytes:
                    raise ProfilerSizeError(
                        f"stream line exceeded max_line_bytes={self.max_line_bytes}"
                    )

                # Write raw bytes; exact persisted-byte accounting.
                written += len(line)
                if written > self.max_stream_bytes:
                    raise ProfilerSizeError(
                        f"stream log exceeded max_stream_bytes={self.max_stream_bytes}"
                    )
                log_file.write(line)
                log_file.flush()

                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    self.first_token_event.set()

        except (ProfilerSizeError, ProfilerTimeoutError) as exc:
            self._record_error(exc)
            self._delete_partial_log()
        except (socket.timeout, TimeoutError, OSError) as exc:
            if self._stop_event.is_set():
                pass  # intentional cancellation via stop()
            elif deadline.expired() or (self._watchdog is not None and self._watchdog.is_fired):
                self._record_error(ProfilerTimeoutError(
                    f"stream exceeded max_stream_seconds={self.max_stream_seconds}"
                ))
                self._delete_partial_log()
            else:
                self._record_error(exc)
                self._delete_partial_log()
        except http.client.HTTPException as exc:
            if self._stop_event.is_set():
                pass  # intentional cancellation via stop()
            else:
                self._record_error(exc)
                self._delete_partial_log()
        except Exception as exc:
            if self._stop_event.is_set():
                pass  # intentional cancellation via stop()
            else:
                self._record_error(exc)
                self._delete_partial_log()
        finally:
            if log_file is not None:
                with contextlib.suppress(Exception):
                    log_file.close()
            if self._watchdog is not None:
                self._watchdog.cancel()
            if self._sock_ref is not None:
                with contextlib.suppress(Exception):
                    self._sock_ref.close()
            if self._response is not None:
                with contextlib.suppress(Exception):
                    self._response.close()
            with contextlib.suppress(Exception):
                conn.close()
            self.finished_event.set()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def build_request(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_int_arg(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _positive_finite_arg(value: str) -> float:
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value!r}") from None
    if fvalue <= 0 or math.isnan(fvalue) or math.isinf(fvalue):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value!r}")
    return fvalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach to an already-running vLLM server, start one streaming request, "
            "wait for the first token, then use vLLM's native profiling endpoints."
        )
    )
    parser.add_argument("--base-url", required=True, help="Server base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("decode", "mtp"), default="decode")
    parser.add_argument("--model", default=None, help="Model id. Auto-detected from /v1/models when omitted.")
    parser.add_argument("--prompt", default=None, help="Inline prompt. Ignored when --prompt-file is set.")
    parser.add_argument("--prompt-file", default=None, help="Read the prompt from a file.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--out-dir", default=None, help="Local output directory for logs and metadata.")
    parser.add_argument("--api-key", default=None, help="Optional bearer token. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--first-token-timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-line-bytes", type=_positive_int_arg, default=_DEFAULT_MAX_LINE_BYTES,
        help=f"Hard maximum bytes per streamed line; abort on overflow. Default: {_DEFAULT_MAX_LINE_BYTES}",
    )
    parser.add_argument(
        "--max-response-bytes", type=_positive_int_arg, default=_DEFAULT_MAX_RESPONSE_BYTES,
        help=f"Hard maximum bytes for a single HTTP response body. Default: {_DEFAULT_MAX_RESPONSE_BYTES}",
    )
    parser.add_argument(
        "--max-stream-bytes", type=_positive_int_arg, default=_DEFAULT_MAX_STREAM_BYTES,
        help=f"Hard maximum total bytes persisted to the stream log. Default: {_DEFAULT_MAX_STREAM_BYTES}",
    )
    parser.add_argument(
        "--max-stream-seconds", type=_positive_finite_arg, default=_DEFAULT_MAX_STREAM_SECONDS,
        help=f"Hard wall-clock maximum for the streaming request. Default: {_DEFAULT_MAX_STREAM_SECONDS}",
    )
    parser.add_argument(
        "--max-status-line-bytes", type=_positive_int_arg, default=_DEFAULT_MAX_STATUS_LINE_BYTES,
        help=f"Hard maximum bytes for an HTTP status line. Default: {_DEFAULT_MAX_STATUS_LINE_BYTES}",
    )
    parser.add_argument(
        "--max-header-bytes", type=_positive_int_arg, default=_DEFAULT_MAX_HEADER_BYTES,
        help=f"Hard maximum total bytes for HTTP response headers. Default: {_DEFAULT_MAX_HEADER_BYTES}",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return (
        "Repeat the exact word token separated by spaces. "
        "Do not stop early. Keep emitting tokens until the token limit is reached."
    )


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    out_dir = Path(args.out_dir or f"/tmp/vllm-native-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args)
    model = args.model or resolve_model(
        base_url,
        headers,
        max_status_line_bytes=args.max_status_line_bytes,
        max_header_bytes=args.max_header_bytes,
        max_response_bytes=args.max_response_bytes,
    )
    request_body = build_request(model, prompt, args.max_tokens)

    request_path = out_dir / "request.json"
    stream_log = out_dir / "stream.log"
    error_log = out_dir / "request.error.txt"
    start_response = out_dir / "start_profile.response"
    stop_response = out_dir / "stop_profile.response"
    manifest_path = out_dir / "manifest.json"
    request_path.write_text(json.dumps(request_body, indent=2) + "\n", encoding="utf-8")

    print(f"output dir: {out_dir}")
    print(f"model: {model}")
    print(f"mode: {args.mode}")
    print("starting streaming request...")

    stream = StreamRequest(
        url=f"{base_url}/v1/completions",
        headers=headers,
        payload=request_body,
        log_path=stream_log,
        error_path=error_log,
        max_line_bytes=args.max_line_bytes,
        max_stream_bytes=args.max_stream_bytes,
        max_stream_seconds=args.max_stream_seconds,
        max_status_line_bytes=args.max_status_line_bytes,
        max_header_bytes=args.max_header_bytes,
    )
    stream.start()
    started_at = time.time()
    profile_started = False

    try:
        if not stream.wait_for_first_token(args.first_token_timeout):
            raise RuntimeError(
                "timed out waiting for the first streamed token."
                + (f" request error: {stream.error}" if stream.error else "")
            )

        # Check if the stream already errored (overflow during first-token wait).
        if stream.finished_event.is_set() and stream.error:
            raise RuntimeError(f"stream error: {stream.error}")

        first_token_at = time.time()
        print("first streamed token observed; starting native profile...")

        # Mark profile_attempted before sending the request so cleanup runs
        # even if the POST activates profiling but the response fails.
        profile_started = True
        status, body = http_json(
            f"{base_url}/start_profile",
            headers=headers,
            payload={},
            max_status_line_bytes=args.max_status_line_bytes,
            max_header_bytes=args.max_header_bytes,
            max_response_bytes=args.max_response_bytes,
        )
        start_response.write_bytes(body)
        if status == 404:
            raise RuntimeError(
                "/start_profile is not available on this server. "
                "For vLLM this usually means the server was not launched with --profiler-config."
            )
        if status != 200:
            raise RuntimeError(f"/start_profile failed with status {status}: {body.decode('utf-8', errors='replace').strip()}")

        deadline = time.time() + args.capture_seconds
        while time.time() < deadline:
            if stream.finished_event.wait(timeout=0.2):
                break

        print("stopping native profile...")
        status, body = http_json(
            f"{base_url}/stop_profile",
            headers=headers,
            payload={},
            max_status_line_bytes=args.max_status_line_bytes,
            max_header_bytes=args.max_header_bytes,
            max_response_bytes=args.max_response_bytes,
        )
        stop_response.write_bytes(body)
        if status != 200:
            raise RuntimeError(f"/stop_profile failed with status {status}: {body.decode('utf-8', errors='replace').strip()}")

        finished_at = time.time()
    except BaseException:
        # Stop local stream promptly before remote cleanup.
        with contextlib.suppress(Exception):
            stream.stop()
        # Best-effort remote profiler cleanup with short explicit deadline.
        if profile_started:
            with contextlib.suppress(Exception):
                http_json(
                    f"{base_url}/stop_profile",
                    headers=headers,
                    payload={},
                    timeout_s=5.0,
                    max_status_line_bytes=args.max_status_line_bytes,
                    max_header_bytes=args.max_header_bytes,
                    max_response_bytes=args.max_response_bytes,
                )
        raise
    finally:
        with contextlib.suppress(Exception):
            stream.stop()

    # After stop(), check for stream errors (overflow during capture window).
    if stream.error:
        # Stream error after successful start: bounded remote stop.
        with contextlib.suppress(Exception):
            http_json(
                f"{base_url}/stop_profile",
                headers=headers,
                payload={},
                timeout_s=5.0,
                max_status_line_bytes=args.max_status_line_bytes,
                max_header_bytes=args.max_header_bytes,
                max_response_bytes=args.max_response_bytes,
            )
        raise RuntimeError(f"stream error: {stream.error}")

    manifest = {
        "base_url": base_url,
        "mode": args.mode,
        "model": model,
        "out_dir": str(out_dir),
        "started_at_epoch_s": started_at,
        "first_token_at_epoch_s": first_token_at,
        "profile_stopped_at_epoch_s": finished_at,
        "capture_seconds": args.capture_seconds,
        "request_body": request_body,
        "notes": [
            "The background request was started before profiling and profiling began after the first streamed token.",
            "vLLM native HTTP profiling is window-based and does not provide a stage filter over the server API.",
            "For speculative servers, draft or verify regions should show up in the same trace window.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("native profiling completed.")
    print(f"stream log: {stream_log}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
