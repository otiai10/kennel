import os
from pathlib import Path

import pytest

from kennel.permissions import PermissionManager
from kennel.tools.base import ToolContext, ToolLimits
from kennel.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test's session event log inside its own tmp directory.

    The CLI keeps an event log by default, so without this a test run would write into the
    developer's real ``~/.local/state/kennel``. ``monkeypatch.setenv`` also reaches the CLI
    subprocesses, which inherit ``os.environ``.
    """
    directory = tmp_path / "state"
    monkeypatch.setenv("KENNEL_STATE_DIR", str(directory))
    return directory


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    """A workspace with text, binary, hidden, ignored and symlinked entries plus an outside sibling."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (root / "docs").mkdir()
    (root / "docs" / "b.md").write_text("# Title\n\nhello World\nhello again\n")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    print('Hello')\n")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.txt").write_text("hidden hello\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("hello from node_modules\n")
    (root / "bin.dat").write_bytes(b"\x00\x01hello\x02")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside hello\n")
    os.symlink(outside / "secret.txt", root / "link_out")
    os.symlink(outside, root / "dir_out")
    os.symlink(root / "a.txt", root / "link_in")
    return root


@pytest.fixture
def ws(ws_dir: Path) -> Workspace:
    return Workspace(ws_dir)


@pytest.fixture
def limits() -> ToolLimits:
    return ToolLimits(max_output_bytes=4096, max_read_lines=50, glob_max_results=100, grep_max_results=50)


@pytest.fixture
def ctx(ws: Workspace, limits: ToolLimits) -> ToolContext:
    return ToolContext(workspace=ws, permission_manager=PermissionManager(), limits=limits)


@pytest.fixture
def meeting_ws(tmp_path: Path) -> Path:
    """Copy of the meeting fixture so tests can mutate it."""
    import shutil

    dst = tmp_path / "meeting_project"
    shutil.copytree(FIXTURES / "meeting_project", dst)
    return dst


# -- a fake HTTP search service (issue #61) -------------------------------------


class FakeSearchServer:
    """A loopback HTTP server that answers every GET with ``reply`` (tests never leave the machine).

    ``stall`` holds the response back; ``body_chunks`` streams that many copies of ``body``;
    ``requests`` records ``(path, headers)`` and ``disconnected`` is set when a client leaves
    before the response was complete.
    """

    def __init__(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.status = 200
        self.body: bytes = b'{"results": []}'
        self.headers: dict[str, str] = {}
        self.stall = 0.0
        self.body_chunks = 1
        self.send_length = True
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.disconnected = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_GET(self):  # noqa: N802 - http.server's naming
                import time

                server.requests.append((self.path, dict(self.headers)))
                deadline = time.monotonic() + server.stall
                while time.monotonic() < deadline:
                    time.sleep(0.02)
                try:
                    self.send_response(server.status)
                    for key, value in server.headers.items():
                        self.send_header(key, value)
                    if server.send_length:
                        self.send_header("Content-Length", str(len(server.body) * server.body_chunks))
                    self.end_headers()
                    for _ in range(server.body_chunks):
                        self.wfile.write(server.body)
                        self.wfile.flush()
                except OSError:
                    server.disconnected.set()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def reply(self, status: int = 200, body=None, headers: dict[str, str] | None = None) -> None:
        import json

        self.status = status
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = headers or {}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def search_server():
    server = FakeSearchServer()
    try:
        yield server
    finally:
        server.close()
