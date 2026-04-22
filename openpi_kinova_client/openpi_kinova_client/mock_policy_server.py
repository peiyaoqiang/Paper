from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import websockets.sync.server

from . import msgpack_numpy


logger = logging.getLogger(__name__)


def handle_connection(websocket) -> None:  # type: ignore[no-untyped-def]
    packer = msgpack_numpy.Packer()
    websocket.send(packer.pack({"mock": True, "action_dim": 7, "action_horizon": 8}))
    while True:
        try:
            _obs = msgpack_numpy.unpackb(websocket.recv())
            actions = np.zeros((8, 7), dtype=np.float32)
            actions[:, 0] = 0.003
            actions[:, 2] = 0.001
            actions[:, 6] = np.linspace(0.0, 1.0, 8)
            websocket.send(
                packer.pack(
                    {
                        "actions": actions,
                        "server_timing": {"infer_ms": 1.0},
                        "policy_timing": {"mock_ms": 1.0},
                    }
                )
            )
        except Exception:
            logger.exception("mock server connection closed")
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting mock openpi policy server on ws://%s:%s", args.host, args.port)
    with websockets.sync.server.serve(handle_connection, args.host, args.port, compression=None, max_size=None) as server:
        while True:
            server.serve_forever()
            time.sleep(1.0)


if __name__ == "__main__":
    main()
