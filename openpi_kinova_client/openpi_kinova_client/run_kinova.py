from __future__ import annotations

import argparse
import logging
import time

from .action_adapter import ActionAdapter, extract_action_chunk
from .config import AdapterConfig, KinovaConfig, ObservationConfig, PolicyServerConfig, SafetyConfig
from .kinova_kortex import KinovaKortexController
from .mock_robot import MockRobot
from .observation import CameraSource, make_droid_observation
from .openpi_ws_client import OpenPIWebsocketClient
from .safety import SafetyLimiter


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run openpi action chunks on a Kinova Gen3 through Kortex.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--robot-ip", default="192.168.1.10")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dummy-images", action="store_true")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--max-translation", type=float, default=0.010)
    parser.add_argument("--max-rotation", type=float, default=0.05)
    parser.add_argument("--command-dt", type=float, default=0.25)
    parser.add_argument("--reach-pose", action="store_true", help="Use ExecuteAction reach_pose instead of twist.")
    parser.add_argument("--execute", action="store_true", help="Actually connect to and move Kinova. Without this, dry-runs on MockRobot.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    adapter = ActionAdapter(
        AdapterConfig(
            position_scale=args.position_scale,
            rotation_scale=args.rotation_scale,
            invert_gripper=args.invert_gripper,
        )
    )
    safety = SafetyLimiter(
        SafetyConfig(
            max_abs_translation_m=args.max_translation,
            max_abs_rotation_rad=args.max_rotation,
        )
    )
    policy_config = PolicyServerConfig(host=args.host, port=args.port, api_key=args.api_key)
    obs_config = ObservationConfig(prompt=args.prompt, camera_index=args.camera_index, dummy_images=args.dummy_images)

    robot = (
        KinovaKortexController(
            KinovaConfig(
                robot_ip=args.robot_ip,
                username=args.username,
                password=args.password,
                command_dt_s=args.command_dt,
                use_twist=not args.reach_pose,
            )
        )
        if args.execute
        else MockRobot()
    )

    if not args.execute:
        logger.warning("Dry-run mode: not connecting to Kinova. Add --execute to move the real robot.")

    with OpenPIWebsocketClient(policy_config) as client, CameraSource(obs_config) as camera:
        if args.execute:
            robot.connect()  # type: ignore[attr-defined]
        try:
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
                logger.info("step=%s received action chunk shape=%s", step, tuple(chunk.shape))

                for chunk_index, action in enumerate(actions[: args.chunk_steps]):
                    current_state = robot.get_state()
                    safe = safety.filter(action, current_state, action_timestamp=started)
                    if safe.stop:
                        logger.error("Safety stop: %s", safe.reason)
                        robot.stop()
                        return
                    logger.info(
                        "execute step=%s chunk=%s action=%s clipped=%s",
                        step,
                        chunk_index,
                        safe.action.as_tuple(),
                        safe.clipped,
                    )
                    robot.apply_action(safe.action)
        finally:
            if args.execute:
                robot.stop()
                robot.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
