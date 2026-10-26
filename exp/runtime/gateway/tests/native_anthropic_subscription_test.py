"""Serve Claude plan requests through the Rust gateway with a local provider."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.models import GatewayDeploymentCapabilities, GatewayTokenPrices, ModelCapabilities
from exp.runtime.gateway.catalog_authority import (
    ConnectionConfig,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
_REQUEST_TIMEOUT_SECONDS = 30.0
_BEARER = "synthetic-dispatch-bearer"

_DRIVER_SOURCE = textwrap.dedent(
    r'''
    """Serve one local Anthropic subscription route through the native gateway."""

    import json
    import os
    import socket
    import sys
    from pathlib import Path

    from exp.common.auth import StoredOAuthTokens
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.native_bridge import NativeControlPlane
    from exp.runtime.models.providers.anthropic_subscription import AnthropicOAuthApp
    from exp.runtime.models.providers.subscription_tokens import SubscriptionSignInError
    from exp.runtime.models.registry import RuntimeModelCatalog as RealRuntimeModelCatalog

    import exp.runtime.gateway.lifecycle as lifecycle
    import exp_gateway_native


    class Source:
        """Return synthetic tokens and persist every access for the parent test."""

        connection_id = "provider-main"

        def __init__(self, path: Path, *, failure: bool) -> None:
            """Store the counter path and the dispatch-failure mode."""
            self._path = path
            self._failure = failure

        def current(self) -> StoredOAuthTokens:
            """Count one access and fail only after the parent arms dispatch."""
            state = json.loads(self._path.read_text())
            calls = int(state["calls"]) + 1
            state["calls"] = calls
            self._path.write_text(json.dumps(state))
            if self._failure and bool(state["armed"]):
                raise SubscriptionSignInError("synthetic subscription sign-in failure")
            return StoredOAuthTokens(
                access_token="synthetic-dispatch-bearer",
                refresh_token="synthetic-refresh",
                expires_at_ms=4_000_000_000_000,
            )


    class CatalogComponents:
        """Delegate gateway components while replacing only runtime catalogs."""

        def __init__(self, inner: object, runtime_catalogs: object) -> None:
            """Keep the real lifecycle components and injected catalog mapping."""
            self._inner = inner
            self.runtime_catalogs = runtime_catalogs

        def __getattr__(self, name: str) -> object:
            """Delegate every component except the explicitly replaced catalogs."""
            return getattr(self._inner, name)


    def main() -> None:
        """Build the real control plane and expose its native HTTP server."""
        config = json.loads(sys.argv[1])
        state_path = Path(config["state_path"])
        source = Source(state_path, failure=bool(config["failure"]))
        app = AnthropicOAuthApp(
            client_id="experiential-issued-client",
            redirect_uri="https://platform.example.test/oauth/anthropic/callback",
            dispatch_headers={"anthropic-beta": "oauth-2025-04-20"},
        )

        def source_factory(*, connection_id: str, connection: object) -> Source:
            """Build the source seam used by every real runtime catalog."""
            del connection
            assert connection_id == "provider-main"
            return source

        def runtime_catalog(catalog: object, *, environment: dict[str, str]) -> object:
            """Construct the production runtime catalog with local-only seams."""
            return RealRuntimeModelCatalog(
                catalog,
                environment=environment,
                subscription_token_source_factory=source_factory,
                anthropic_oauth_app=app,
            )

        lifecycle.RuntimeModelCatalog = runtime_catalog
        components = load_gateway_components(Path(config["root"]), environment={})
        admission_path = Path(config["admission_path"])

        class LoopbackControlPlane(NativeControlPlane):
            """Redirect only the admitted Anthropic destination to the local provider."""

            def admit(self, argument: str) -> str:
                """Run real admission, record static headers, and rewrite only its URL."""
                admitted = json.loads(super().admit(argument))
                routes = admitted["route"]
                source_state = json.loads(state_path.read_text())
                admission_path.write_text(
                    json.dumps(
                        {
                            "calls_after_admit": source_state["calls"],
                            "headers": [
                                route.get("headers", {})
                                for route in routes
                                if isinstance(route, dict)
                            ]
                        }
                    )
                )
                for route in routes:
                    assert route["url"] == config["expected_destination"]
                    route["url"] = config["mock_destination"]
                return json.dumps(admitted, separators=(",", ":"))

            def sign_dispatch(self, argument: str) -> str:
                """Restore the admitted URL before real dispatch signing validates it."""
                signed = json.loads(argument)
                signed["url"] = config["expected_destination"]
                return super().sign_dispatch(json.dumps(signed, separators=(",", ":")))

        control_plane = LoopbackControlPlane(
            CatalogComponents(components, components.runtime_catalogs),
            request_timeout_seconds=config["request_timeout_seconds"],
        )
        last_error = None
        for _attempt in range(5):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            sys.stdout.write(json.dumps({"port": port}) + "\n")
            sys.stdout.flush()
            try:
                exp_gateway_native.serve(
                    control_plane,
                    json.dumps(
                        {
                            "host": "127.0.0.1",
                            "port": port,
                            "max_active_requests": 8,
                            "request_timeout_seconds": config["request_timeout_seconds"],
                            "graceful_timeout_seconds": 2.0,
                        }
                    ),
                )
                return
            except RuntimeError as error:
                if "failed to bind" not in str(error):
                    raise
                last_error = error
        raise SystemExit(f"no loopback port could be bound: {last_error}")


    if __name__ == "__main__":
        main()
    '''
).strip()


def _anthropic_frames(model: str) -> bytes:
    """Return one minimal successful Anthropic Messages event stream."""
    events = (
        {
            "type": "message_start",
            "message": {
                "id": "msg-loopback",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "loopback"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
    frames: list[bytes] = []
    for event in events:
        event_type = event["type"]
        assert isinstance(event_type, str)
        frames.append(
            b"event: "
            + event_type.encode()
            + b"\n"
            + b"data: "
            + json.dumps(event, separators=(",", ":")).encode()
            + b"\n\n"
        )
    return b"".join(frames)


class _MessagesUpstream(BaseHTTPRequestHandler):
    """Record one loopback provider request and return Anthropic SSE."""

    request_count = 0
    authorization_headers: list[str] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Record the bearer and return a bounded Messages response."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.lock:
            type(self).request_count += 1
            type(self).authorization_headers.append(self.headers.get("authorization", ""))
        if self.path != "/v1/messages":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_anthropic_frames(str(payload["model"])))
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs from the bounded local provider."""
        del format, args


@dataclass(frozen=True)
class _ServingEngine:
    """One live native gateway and its local observation files.

    Attributes:
        port: Bound loopback gateway port.
        raw_key: Synthetic virtual key scoped to the local test identity.
        state_path: Token-source access count and failure trigger.
        admission_path: Static headers and count captured after admission.
        failure: Whether the test arms a dispatch-time sign-in error.
    """

    port: int
    raw_key: str
    state_path: Path
    admission_path: Path
    failure: bool

    @property
    def base(self) -> str:
        """Return the native gateway HTTP origin."""
        return f"http://{_HOST}:{self.port}"


def _configured_anthropic_subscription(root: Path) -> tuple[GatewayManagement, str]:
    """Create one real granted Anthropic subscription alias."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(provider="anthropic", subscription="anthropic"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="claude-sonnet-5",
        exact_model_id="model-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(maximum_output_tokens=8192, supports_reasoning=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


@pytest.fixture(params=[False, True], name="engine")
def _engine(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[_ServingEngine]:
    """Serve success and dispatch-sign-in-failure cases through Rust."""
    failure = bool(getattr(request, "param", False))
    root = tmp_path / "gateway-root"
    root.mkdir()
    state_path = root / "token-state.json"
    state_path.write_text(json.dumps({"calls": 0, "armed": False}))
    admission_path = root / "admission.json"
    upstream = ThreadingHTTPServer((_HOST, 0), _MessagesUpstream)
    _MessagesUpstream.request_count = 0
    _MessagesUpstream.authorization_headers = []
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    _manager, raw_key = _configured_anthropic_subscription(root)
    driver = root / "native_anthropic_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "state_path": str(state_path),
            "admission_path": str(admission_path),
            "failure": failure,
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
            "mock_destination": f"http://{_HOST}:{upstream.server_address[1]}/v1/messages",
            "expected_destination": "https://api.anthropic.com/v1/messages",
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Record every native port announcement."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        reader = threading.Thread(target=_collect_announcements, daemon=True)
        reader.start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                    if live.status_code == 200 and live.json() == {"status": "live"}:
                        models = httpx.get(
                            f"http://{_HOST}:{port}/v1/models",
                            headers={"authorization": f"Bearer {raw_key}"},
                            timeout=2.0,
                        )
                        if models.status_code == 200 and [
                            item["id"] for item in models.json()["data"]
                        ] == ["coding"]:
                            break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(
            port=port,
            raw_key=raw_key,
            state_path=state_path,
            admission_path=admission_path,
            failure=failure,
        )
    finally:
        try:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise
        finally:
            stderr_sink.close()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=5)


def _read_state(path: Path) -> dict[str, object]:
    """Read one source observation document."""
    return json.loads(path.read_text())


def test_native_anthropic_subscription_defers_token_access_until_dispatch(
    engine: _ServingEngine,
) -> None:
    """Serve Messages without admission token access and sign exactly once."""
    state = _read_state(engine.state_path)
    assert state["calls"] == 0
    if engine.failure:
        engine.state_path.write_text(json.dumps({**state, "armed": True}))
        response = httpx.post(
            f"{engine.base}/v1/messages",
            headers={"x-api-key": engine.raw_key},
            json={
                "model": "coding",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        assert response.status_code == 502, response.text
        assert response.json()["error"]["type"] == "api_error"
        assert "provider dispatch signing failed" in response.json()["error"]["message"]
        assert "synthetic subscription sign-in failure" not in response.text
        assert _MessagesUpstream.request_count == 0
    else:
        response = httpx.post(
            f"{engine.base}/v1/messages",
            headers={"x-api-key": engine.raw_key},
            json={
                "model": "coding",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"][0]["text"] == "loopback"
        assert _MessagesUpstream.request_count == 1
        assert _MessagesUpstream.authorization_headers == [f"Bearer {_BEARER}"]
    admission = _read_state(engine.admission_path)
    headers = admission["headers"]
    assert admission["calls_after_admit"] == 0
    assert isinstance(headers, list) and headers
    for item in headers:
        assert isinstance(item, dict)
        for name in item:
            assert isinstance(name, str)
            assert name.lower() != "authorization"
    assert _read_state(engine.state_path)["calls"] == 1
