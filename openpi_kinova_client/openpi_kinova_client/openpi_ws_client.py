from __future__ import annotations

import logging
import time
from typing import Any

from .config import PolicyServerConfig
from . import msgpack_numpy


logger = logging.getLogger(__name__)


class OpenPIWebsocketClient:
    """Small openpi-compatible WebSocket client.

    Protocol used by openpi:
    - server sends metadata immediately after connection
    - client sends a msgpack_numpy-packed observation dict
    - server replies with a msgpack_numpy-packed dict, normally including "actions"
    """

    def __init__(self, config: PolicyServerConfig) -> None:
        self.config = config
        if config.host.startswith("ws://") or config.host.startswith("wss://"):
            self.uri = config.host
        else:
            self.uri = f"ws://{config.host}:{config.port}"
        self._websockets_client = None
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self.metadata: dict[str, Any] = {}

    def _ensure_deps(self) -> None:
        if self._websockets_client is not None:
            return
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise RuntimeError(
                "Missing WebSocket client dependencies. Run: pip install -r requirements.txt"
            ) from exc
        self._websockets_client = websockets.sync.client

    def connect(self) -> dict[str, Any]:
        self._ensure_deps()
        assert self._websockets_client is not None
        headers = {"Authorization": f"Api-Key {self.config.api_key}"} if self.config.api_key else None
        while True:
            try:
                logger.info("Connecting to openpi policy server at %s", self.uri)
                connect_kwargs = {
                    "compression": None,
                    "max_size": None,
                    "ping_interval": self.config.ping_interval_s,
                    "ping_timeout": self.config.ping_timeout_s,
                    "close_timeout": self.config.close_timeout_s,
                }
                try:
                    self._ws = self._websockets_client.connect(
                        self.uri,
                        **connect_kwargs,
                        additional_headers=headers,
                    )
                except TypeError:
                    # Older websockets versions used extra_headers.
                    self._ws = self._websockets_client.connect(
                        self.uri,
                        **connect_kwargs,
                        extra_headers=headers,
                    )
                self.metadata = msgpack_numpy.unpackb(self._ws.recv())
                logger.info("Policy server metadata: %s", self.metadata)
                return self.metadata
            except ConnectionRefusedError:
                logger.warning("Server not ready, retrying in %.1fs", self.config.connect_retry_s)
                time.sleep(self.config.connect_retry_s)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self.connect()
        assert self._ws is not None
        assert self._packer is not None
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"openpi inference server returned an error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "OpenPIWebsocketClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
