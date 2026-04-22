from __future__ import annotations

import argparse
import logging
import time

from .action_adapter import ActionAdapter, extract_action_chunk
from .config import AdapterConfig, ObservationConfig, PolicyServerConfig, SafetyConfig
from .mock_robot import MockRobot
from .observation import CameraSource, make_droid_observation
from .openpi_ws_client import OpenPIWebsocketClient
from .safety import SafetyLimiter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query openpi policy server and print action chunks without moving a robot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dummy-images", action="store_true")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--print-rows", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    robot = MockRobot()
    adapter = ActionAdapter(
        AdapterConfig(
            position_scale=args.position_scale,
            rotation_scale=args.rotation_scale,
            invert_gripper=args.invert_gripper,
        )
    )
    safety = SafetyLimiter(SafetyConfig())

    policy_config = PolicyServerConfig(host=args.host, port=args.port, api_key=args.api_key)
    obs_config = ObservationConfig(prompt=args.prompt, camera_index=args.camera_index, dummy_images=args.dummy_images)

    with OpenPIWebsocketClient(policy_config) as client, CameraSource(obs_config) as camera:
        for step in range(args.steps):
            state = robot.get_state()
            image = camera.read_rgb()
            obs = make_droid_observation(
                exterior_image=image,
                wrist_image=image,
                robot_state=state,
                prompt=args.prompt,
                image_size=obs_config.image_size,
            )
            started = time.monotonic()
            response = client.infer(obs)
            chunk = extract_action_chunk(response)
            actions = adapter.chunk_to_actions(response)
            safe_first = safety.filter(actions[0], state, action_timestamp=started)
            robot.apply_action(safe_first.action)

            print(f"\nstep={step} chunk_shape={tuple(chunk.shape)}")
            print(chunk[: args.print_rows])
            print(f"mapped_first={safe_first.action.as_tuple()} clipped={safe_first.clipped} reason={safe_first.reason}")


if __name__ == "__main__":
    main()
