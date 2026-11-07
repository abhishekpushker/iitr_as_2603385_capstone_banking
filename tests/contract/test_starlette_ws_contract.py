"""Exceptions raised on WebSocket receive/send after a client disconnect (§ARCH 2.4, 7.16).

Pinned against a real uvicorn server on 127.0.0.1 (the protocol implementation `auto` picks is
what production runs), driven by the `websockets` sync client.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import anyio
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketDisconnected
from websockets.sync.client import connect

pytestmark = pytest.mark.enable_socket

WAIT_S = 5.0


@dataclass
class Observed:
    errors: dict[str, BaseException] = field(default_factory=dict)
    client_closed: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


def _app(obs: Observed) -> Starlette:
    async def receive_after_close(ws: WebSocket) -> None:
        await ws.accept()
        try:
            await ws.receive_text()
        except Exception as exc:
            obs.errors["receive"] = exc
        obs.done.set()

    async def send_after_close(ws: WebSocket) -> None:
        await ws.accept()
        await anyio.to_thread.run_sync(obs.client_closed.wait, WAIT_S)
        for attempt in ("send1", "send2"):
            try:
                await ws.send_text("late answer")
            except Exception as exc:
                obs.errors[attempt] = exc
        obs.done.set()

    return Starlette(
        routes=[
            WebSocketRoute("/receive", receive_after_close),
            WebSocketRoute("/send", send_after_close),
        ]
    )


@pytest.fixture
def server() -> Iterator[tuple[int, Observed]]:
    obs = Observed()
    # Listening before uvicorn starts: connects queue in the backlog, so no start-up polling.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    config = uvicorn.Config(_app(obs), log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    yield port, obs
    srv.should_exit = True
    thread.join(WAIT_S)
    sock.close()


def test_default_ws_implementation_is_websockets_sansio() -> None:
    config = uvicorn.Config(Starlette())
    config.load()
    assert config.ws == "auto"
    assert config.ws_protocol_class.__name__ == "WebSocketsSansIOProtocol"


def test_receive_after_client_close_raises_websocket_disconnect(
    server: tuple[int, Observed],
) -> None:
    port, obs = server
    with connect(f"ws://127.0.0.1:{port}/receive"):
        pass
    assert obs.done.wait(WAIT_S)
    exc = obs.errors["receive"]
    assert type(exc) is WebSocketDisconnect
    assert exc.code == 1000


def test_send_after_client_close_raises_disconnect_then_disconnected(
    server: tuple[int, Observed],
) -> None:
    """The mid-turn case: first send → WebSocketDisconnect(1006); any later send →
    WebSocketDisconnected, a RuntimeError subclass. The WS loop must catch both."""
    port, obs = server
    with connect(f"ws://127.0.0.1:{port}/send"):
        pass
    obs.client_closed.set()
    assert obs.done.wait(WAIT_S)
    first, second = obs.errors["send1"], obs.errors["send2"]
    assert type(first) is WebSocketDisconnect
    assert first.code == 1006
    assert type(second) is WebSocketDisconnected
    assert isinstance(second, RuntimeError)
    assert "once a close message has been sent" in str(second)


def test_testclient_websocket_connect_mutates_the_headers_dict() -> None:
    """BUG-004: `TestClient.websocket_connect(headers=d)` adds the handshake headers to `d` itself
    (`sec-websocket-key`, `connection`, …). A shared headers constant passed to it then carries a
    second `Sec-WebSocket-Key` into a real client's handshake, which the server rejects (400).
    Tests therefore pass a copy."""
    from starlette.testclient import TestClient

    async def accept_close(ws: WebSocket) -> None:
        await ws.accept()
        await ws.close()

    app = Starlette(routes=[WebSocketRoute("/ws", accept_close)])
    headers = {"X-API-Key": "k"}
    with TestClient(app).websocket_connect("/ws", headers=headers):
        pass
    assert "sec-websocket-key" in headers
