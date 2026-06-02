"""HTTP/SSE BrainClient for gabagent's voice server (the Brain protocol).

As-built protocol (loopback only): `GET /health`; `POST /respond` and `POST /confirm` each
return a `text/event-stream` of `data: {json}\\n\\n` VoiceEvents (the stream ends after `done`,
or after a `confirm`); `POST /cancel {session_id}` aborts the in-flight turn. Each respond/confirm
call owns its own short-lived stream — no held-open socket — so it maps 1:1 onto BrainLLMService's
turn-based flow.

Optionally manages the brain subprocess: `start()` spawns `gab --voice-serve …` and waits for
`/health`; `aclose()` tears it down.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Sequence

import httpx
from loguru import logger

from brains.brain_client import BrainEvent

_EVENT_FIELDS = set(BrainEvent.__dataclass_fields__)


class HttpBrainClient:
    """Talks to a gabagent voice-mode server over HTTP/SSE; satisfies BrainClient."""

    def __init__(
        self,
        base_url: str,
        *,
        launch: Sequence[str] | None = None,
        cwd: str | None = None,
        start_timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._launch = list(launch) if launch else None
        self._cwd = cwd
        self._start_timeout = start_timeout
        # No overall read timeout — turns can run long (tools / model). Keep a connect timeout.
        # `transport` lets tests inject an in-process ASGI app (httpx.ASGITransport).
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(None, connect=10.0),
            transport=transport,
        )
        self._proc: asyncio.subprocess.Process | None = None
        self._active_session: str | None = None

    # --- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Connect-or-spawn: attach to an existing brain on the port, else spawn our own.

        Symmetric with gabagent's `/voice on` launcher so the two never double-spawn (port
        clash). We only ever tear down a brain WE spawned (`self._proc` is set only here).
        """
        if await self._is_healthy():
            logger.info(f"Brain: attached to existing server at {self._base_url} (not spawned)")
            return
        if self._launch:
            logger.info(f"Brain: launching: {' '.join(self._launch)}")
            self._proc = await asyncio.create_subprocess_exec(*self._launch, cwd=self._cwd)
        await self._await_health()

    async def _is_healthy(self) -> bool:
        try:
            r = await self._http.get("/health", timeout=2.0)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:  # noqa: BLE001 - nothing listening yet
            return False

    async def _await_health(self) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._start_timeout
        while loop.time() < deadline:
            if self._proc is not None and self._proc.returncode is not None:
                raise RuntimeError(f"Brain process exited early (code {self._proc.returncode}).")
            try:
                r = await self._http.get("/health")
                if r.status_code == 200 and r.json().get("status") == "ok":
                    logger.info(f"Brain: healthy at {self._base_url}")
                    return
            except Exception:  # noqa: BLE001 - server not up yet
                pass
            await asyncio.sleep(0.3)
        raise RuntimeError(f"Brain not healthy within {self._start_timeout}s at {self._base_url}")

    # --- protocol ----------------------------------------------------------
    async def _sse(self, path: str, payload: dict) -> AsyncIterator[BrainEvent]:
        async with self._http.stream("POST", path, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Brain: unparseable SSE data: {data!r}")
                    continue
                yield BrainEvent(**{k: v for k, v in obj.items() if k in _EVENT_FIELDS})

    async def respond(self, session_id: str, text: str) -> AsyncIterator[BrainEvent]:
        self._active_session = session_id
        async for ev in self._sse(
            "/respond", {"session_id": session_id, "text": text, "override_token": None}
        ):
            yield ev

    async def confirm(
        self, session_id: str, confirm_id: str, approved: bool, passphrase: str | None = None
    ) -> AsyncIterator[BrainEvent]:
        self._active_session = session_id
        async for ev in self._sse(
            "/confirm",
            {"session_id": session_id, "id": confirm_id, "approved": approved, "passphrase": passphrase},
        ):
            yield ev

    async def cancel(self, session_id: str) -> None:
        """Abort the in-flight turn (barge-in). Keeps the client/subprocess alive."""
        try:
            await self._http.post("/cancel", json={"session_id": session_id})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Brain: /cancel failed (already done?): {e}")

    async def duck(self, session_id: str, on: bool) -> None:
        """Duck/pause (on=True) or restore (on=False) the brain's active media. Best-effort:
        a brain without `/media/duck` (404) or no active media is a harmless no-op."""
        try:
            await self._http.post(
                "/media/duck", json={"session_id": session_id, "on": on},
                timeout=httpx.Timeout(5.0),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Brain: /media/duck failed (ignored): {e}")

    async def aclose(self) -> None:
        """Full teardown: cancel any active turn, close HTTP, stop the spawned process."""
        if self._active_session:
            await self.cancel(self._active_session)
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
            except ProcessLookupError:
                pass
