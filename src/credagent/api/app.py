"""The FastAPI application (§ARCH 3.5, 7.16, 8). `uvicorn credagent.api.app:app --workers 1
--no-access-log` imports `app`, which is created on first access (so importing this module
configures no logging and touches no service).

Startup: refuse to run as one of several workers (P14: session memory and the cache are
in-process); load the embedder (fail fast: a missing model exits non-zero); then build the
services in the background. Chroma may still be starting, so the build is retried; until it
succeeds `/healthz` is 200, `/readyz` is 503 and the other endpoints answer 503 `NOT_READY`.
Tests and the transcripts inject ready-made `services` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio.to_thread
from fastapi import FastAPI

from credagent.api import routes_http, routes_ws
from credagent.api.errors import install_handlers
from credagent.api.middleware import BodyLimitMiddleware, TimingMiddleware, TraceLogMiddleware
from credagent.guardrails.pii import PiiMasker
from credagent.observability.logging import configure_logging
from credagent.pipeline.factory import Services, build_services
from credagent.rag.embedder import get_embedder
from credagent.settings import Settings, get_settings

Build = Callable[[], Services]


def assert_single_worker() -> None:
    """With `--workers N > 1` uvicorn serves from child processes; refuse to start there."""
    if multiprocessing.parent_process() is not None:
        raise RuntimeError("credagent must run as a single worker: state is in-process (P14)")


async def initialize(app: FastAPI, build: Build, *, retry_s: float) -> None:
    """Build the services, retrying until they are ready; then the app is ready."""
    log = app.state.log
    while True:
        try:
            services = await anyio.to_thread.run_sync(build)
        except Exception as exc:  # noqa: BLE001  (logged and retried: Chroma may be starting)
            log.event("startup_error", level=logging.WARNING, error=type(exc).__name__)
            await anyio.sleep(retry_s)
        else:
            app.state.services = services
            log.event("ready")
            return


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    build: Build | None = None,
) -> FastAPI:
    s = get_settings() if settings is None else settings
    masker = PiiMasker()
    log = configure_logging(s.log_path, lambda text: masker.mask(text).text)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        assert_single_worker()
        log.event("startup", mock_llm=s.mock_llm, workers=1)
        task: asyncio.Task[None] | None = None
        if app.state.services is None:
            if build is None:
                await anyio.to_thread.run_sync(get_embedder)  # fail fast without the model
            builder: Build = (lambda: build_services(s)) if build is None else build
            task = asyncio.create_task(initialize(app, builder, retry_s=s.chroma_backoff_cap_s))
        yield
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Cred loan-support agent", version="1.0.0", lifespan=lifespan)
    app.state.settings, app.state.services = s, services
    app.state.log, app.state.masker = log, masker
    install_handlers(app)
    app.include_router(routes_http.router)
    app.include_router(routes_ws.router)
    # add_middleware wraps: the last added runs first → trace/log, then body cap, then timing.
    app.add_middleware(TimingMiddleware)
    app.add_middleware(BodyLimitMiddleware, max_bytes=s.max_body_bytes)
    app.add_middleware(TraceLogMiddleware, log=log, masker=masker)
    return app


def __getattr__(name: str) -> Any:
    if name == "app":
        created = create_app()
        globals()["app"] = created
        return created
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
