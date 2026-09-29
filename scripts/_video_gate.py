"""Shared harness for the video-posting gates.

Three things every video gate needs and none of the account gates provide:

* ``VideoServer``, a real HTTP server on 127.0.0.1 that serves video bytes the
  way a CDN does (Content-Length, redirects, chunked bodies, error statuses), so
  the downloader is exercised over a socket rather than against a mock of
  itself.
* ``FakeApi``, which stands in for ``requests.post/put/get`` and records every
  call. A platform's whole upload protocol is then asserted from the recorded
  bytes: the gate compares a hash of what the "platform" received with a hash of
  the file that was served, so a dropped, duplicated or reordered chunk fails.
* ``make_sample_mp4``, a genuine H.264 file made with ffmpeg, for the gates that
  need something ffprobe can really read.

Nothing here reaches a real network host or account.
"""

import glob
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import requests
from requests.structures import CaseInsensitiveDict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import check, fail  # noqa: E402,F401  (re-exported for the gates)

MB = 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def temp_videos():
    """Video temp files currently on disk; a leak shows up as a growing set."""
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "insights_video_*")))


# ---------------------------------------------------------------------------
# A real HTTP server for the downloader
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep gate output readable
        pass

    def _serve(self, head_only):
        server = self.server
        server.hits.append((self.command, self.path))
        route = server.routes.get(self.path.split("?")[0])
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = route.get("status", 200)
        self.send_response(status)
        for name, value in route.get("headers", {}).items():
            self.send_header(name, value)
        body = route.get("body", b"")
        if route.get("chunked"):
            self.send_header("Transfer-Encoding", "chunked")
            if route.get("type"):
                self.send_header("Content-Type", route["type"])
            self.end_headers()
            if head_only:
                return
            for start in range(0, len(body), 65536):
                piece = body[start:start + 65536]
                self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if route.get("type"):
            self.send_header("Content-Type", route["type"])
        declared = route.get("declared_length", len(body))
        self.send_header("Content-Length", str(declared))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_GET(self):
        self._serve(False)

    def do_HEAD(self):
        self._serve(True)


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A client that hangs up mid-body (the downloader refusing an oversize
        # file) is the behaviour under test, not an error worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class VideoServer:
    """``routes`` maps a path to ``{body, type, status, headers, chunked}``."""

    def __init__(self, routes):
        self.httpd = _QuietServer(("127.0.0.1", 0), _Handler)
        self.httpd.routes = routes
        self.httpd.hits = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self):
        return self.httpd.server_address[1]

    def url(self, path, host="127.0.0.1"):
        return f"http://{host}:{self.port}{path}"

    @property
    def hits(self):
        return self.httpd.hits

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# A fake platform API
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = text if text is not None else json.dumps(body if body is not None else {})

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def close(self):
        pass


class FakeApi:
    """Replaces ``requests.post/put/get`` and records what each call carried.

    ``on(method, fragment, handler)`` registers a handler for calls whose URL
    contains ``fragment``; the first registered match wins, so a gate lists the
    specific route before the general one. A handler receives the recorded call
    and returns a ``FakeResponse``. A call nobody registered fails the gate: an
    unexpected request is exactly what these gates exist to catch.
    """

    def __init__(self):
        self.calls = []
        self.routes = []
        self._real = {}

    def on(self, method, fragment, handler):
        self.routes.append((method.upper(), fragment, handler))

    def _dispatch(self, method, url, **kwargs):
        call = {"method": method, "url": url, **kwargs}
        self.calls.append(call)
        for route_method, fragment, handler in self.routes:
            if route_method == method and fragment in url:
                return handler(call)
        fail(f"unexpected {method} {url}")

    def calls_to(self, method, fragment):
        return [c for c in self.calls if c["method"] == method.upper() and fragment in c["url"]]

    def __enter__(self):
        for name in ("post", "put", "get"):
            self._real[name] = getattr(requests, name)
            setattr(requests, name, lambda url, _m=name.upper(), **kw: self._dispatch(_m, url, **kw))
        return self

    def __exit__(self, *exc):
        for name, fn in self._real.items():
            setattr(requests, name, fn)


def form_bytes(call, field):
    """The bytes of a multipart file field in a recorded call."""
    entry = call["files"][field]
    return entry[1]


# ---------------------------------------------------------------------------
# A genuine MP4
# ---------------------------------------------------------------------------


def make_sample_mp4(directory, seconds=2, size="320x240"):
    """A real H.264/AAC MP4 of ``seconds`` length, or fail if ffmpeg can't make one."""
    ffmpeg = shutil.which("ffmpeg")
    check(ffmpeg, "ffmpeg is required for this gate and was not found on PATH")
    path = os.path.join(directory, f"sample_{seconds}s.mp4")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path],
        capture_output=True, text=True, timeout=120,
    )
    check(result.returncode == 0 and os.path.getsize(path) > 1000,
          f"ffmpeg could not make a sample video: {result.stderr.strip()[:300]}")
    return path


def no_wait(*modules):
    """Stop status polling from really sleeping in the given client modules."""
    for module in modules:
        module._sleep = lambda _seconds: None
