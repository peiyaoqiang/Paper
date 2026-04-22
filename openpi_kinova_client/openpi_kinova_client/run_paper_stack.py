from __future__ import annotations

import argparse
import logging
import math
import time

from .action_adapter import ActionAdapter, extract_action_chunk
from .config import AdapterConfig, PolicyServerConfig, SafetyConfig
from .mock_robot import MockRobot
from .observation import make_droid_observation
from .openpi_ws_client import OpenPIWebsocketClient
from .paper_stack_bridge import (
    action7_to_kinova_delta,
    build_camera,
    build_gripper,
    build_robot,
    convert_robot_state,
    load_paper_config,
    rgb_from_camera_frame,
)
from .safety import SafetyLimiter
from .types import Action7, RobotState


logger = logging.getLogger(__name__)


def _normalize_sign(value: float) -> float:
    return -1.0 if value < 0.0 else 1.0


def _validate_axis_order(axis_order: str) -> tuple[int, int, int]:
    normalized = axis_order.strip().lower()
    if len(normalized) != 3 or set(normalized) != {"x", "y", "z"}:
        raise ValueError("--policy-axis-order must be a permutation of xyz, for example xyz or xzy.")
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    return tuple(axis_to_index[axis_name] for axis_name in normalized)


def _apply_axis_mapping(
    action: Action7,
    *,
    axis_order: tuple[int, int, int],
    axis_signs: tuple[float, float, float],
    axis_scales: tuple[float, float, float],
    yaw_sign: float,
    yaw_scale: float,
    zero_z: bool,
) -> Action7:
    raw_xyz = (action.dx, action.dy, action.dz)
    mapped_xyz = tuple(
        raw_xyz[source_idx] * axis_sign * axis_scale
        for source_idx, axis_sign, axis_scale in zip(axis_order, axis_signs, axis_scales)
    )
    if zero_z:
        mapped_xyz = (mapped_xyz[0], mapped_xyz[1], 0.0)
    return Action7(
        dx=mapped_xyz[0],
        dy=mapped_xyz[1],
        dz=mapped_xyz[2],
        droll=action.droll,
        dpitch=action.dpitch,
        dyaw=action.dyaw * yaw_sign * yaw_scale,
        gripper=action.gripper,
    )


ZERO_POLICY_STATE = RobotState(
    ee_xyz_m=(0.0, 0.0, 0.0),
    rpy_rad=(0.0, 0.0, 0.0),
    joint_position=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    gripper_position=0.0,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use existing Paper RealSense/Kinova ROS2 stack with remote openpi policy server."
    )
    parser.add_argument("--config", default=None, help="Path to configs/default_config.json override.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--instruction", default="", help="Override config task instruction.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--policy-state",
        choices=("real", "mock", "zero"),
        default="real",
        help="Robot state sent to openpi. Execution still uses the real robot when --execute is set.",
    )
    parser.add_argument("--position-scale", type=float, default=0.02)
    parser.add_argument("--rotation-scale", type=float, default=0.05)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--policy-axis-order", default="xyz")
    parser.add_argument("--policy-x-sign", type=float, default=1.0)
    parser.add_argument("--policy-y-sign", type=float, default=1.0)
    parser.add_argument("--policy-z-sign", type=float, default=1.0)
    parser.add_argument("--policy-x-scale", type=float, default=1.0)
    parser.add_argument("--policy-y-scale", type=float, default=1.0)
    parser.add_argument("--policy-z-scale", type=float, default=1.0)
    parser.add_argument("--policy-yaw-sign", type=float, default=1.0)
    parser.add_argument("--policy-yaw-scale", type=float, default=1.0)
    parser.add_argument("--zero-policy-z", action="store_true")
    parser.add_argument("--max-translation", type=float, default=0.005)
    parser.add_argument("--max-rotation", type=float, default=0.03, help="Radians.")
    parser.add_argument("--max-action-age", type=float, default=30.0)
    parser.add_argument(
        "--enforce-workspace",
        action="store_true",
        help="Force workspace clipping on even if configs/default_config.json disables it.",
    )
    parser.add_argument(
        "--disable-workspace",
        action="store_true",
        help="Force workspace clipping off; per-step limits still apply.",
    )
    parser.add_argument("--max-translation-step-m", type=float, default=None)
    parser.add_argument("--max-rotation-step-deg", type=float, default=None)
    parser.add_argument("--twist-command-duration-s", type=float, default=None)
    parser.add_argument("--twist-stop-duration-s", type=float, default=None)
    parser.add_argument("--combined-axis-commands", action="store_true")
    parser.add_argument("--twist-command-frame", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually move Kinova through the existing ROS2 driver.")
    parser.add_argument(
        "--read-robot-state",
        action="store_true",
        help="Read real Kinova joint/TF state without executing motion. Useful for full observation dry-runs.",
    )
    parser.add_argument("--skip-initial-open", action="store_true")
    parser.add_argument(
        "--gripper-mode",
        choices=("ignore", "close_only", "open_close"),
        default="ignore",
        help="How to apply openpi gripper bit. Default ignore until action semantics are confirmed.",
    )
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-steps-before-close",
        type=int,
        default=5,
        help="Ignore close commands before this many policy steps to avoid early air grasps.",
    )
    parser.add_argument(
        "--lift-after-close-m",
        type=float,
        default=0.08,
        help="Lift distance after a close_only grasp succeeds.",
    )
    parser.add_argument(
        "--settle-before-close-s",
        type=float,
        default=0.15,
        help="Short pause before closing the gripper.",
    )
    parser.add_argument("--print-rows", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_paper_config(args.config)
    instruction = args.instruction.strip() or config["task"]["instruction"]
    axis_order = _validate_axis_order(args.policy_axis_order)
    axis_signs = (
        _normalize_sign(args.policy_x_sign),
        _normalize_sign(args.policy_y_sign),
        _normalize_sign(args.policy_z_sign),
    )
    axis_scales = (args.policy_x_scale, args.policy_y_scale, args.policy_z_scale)
    yaw_sign = _normalize_sign(args.policy_yaw_sign)

    adapter = ActionAdapter(
        AdapterConfig(
            position_scale=args.position_scale,
            rotation_scale=args.rotation_scale,
            invert_gripper=args.invert_gripper,
        )
    )
    workspace_enforced = bool(config["robot"].get("workspace_enforced", True))
    if args.enforce_workspace:
        workspace_enforced = True
    if args.disable_workspace:
        workspace_enforced = False

    safety = SafetyLimiter(
        SafetyConfig(
            max_abs_translation_m=args.max_translation,
            max_abs_rotation_rad=args.max_rotation,
            workspace_min_xyz_m=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_max_xyz_m=tuple(config["robot"]["workspace_xyz_max"]),
            workspace_enforced=workspace_enforced,
            max_action_age_s=args.max_action_age,
        )
    )

    camera = build_camera(config, node_suffix="openpi")
    use_real_robot_driver = args.execute or args.read_robot_state
    robot = (
        build_robot(
            config,
            max_translation_step_m=args.max_translation_step_m,
            max_rotation_step_deg=args.max_rotation_step_deg,
            twist_command_duration_s=args.twist_command_duration_s,
            twist_stop_duration_s=args.twist_stop_duration_s,
            combined_axis_commands=args.combined_axis_commands,
            twist_command_frame=args.twist_command_frame,
            node_suffix="openpi",
        )
        if use_real_robot_driver
        else MockRobot()
    )
    policy_mock_robot = MockRobot()
    gripper = build_gripper(config, robot) if args.execute and args.gripper_mode != "ignore" else None

    print("Instruction:", instruction)
    print("Real camera mode:", config["camera"]["mode"])
    print("Execute robot:", args.execute)
    print("Read real robot state:", args.read_robot_state)
    print("Policy state:", args.policy_state)
    print("Gripper mode:", args.gripper_mode)
    print("Min steps before close:", args.min_steps_before_close)
    print("Lift after close m:", args.lift_after_close_m)
    print("Position scale:", args.position_scale)
    print("Rotation scale:", args.rotation_scale)
    print("Safety max translation m:", args.max_translation)
    print("Safety max rotation rad:", args.max_rotation)
    print("Workspace enforced:", workspace_enforced)
    print("Policy axis order:", args.policy_axis_order)
    print("Policy axis signs:", axis_signs)
    print("Policy axis scales:", axis_scales)
    print("Zero policy z:", args.zero_policy_z)
    if not args.execute:
        print("Dry-run: RealSense is used, but Kinova will not move. Add --execute to move.")

    with OpenPIWebsocketClient(PolicyServerConfig(args.host, args.port, args.api_key)) as client:
        try:
            if gripper is not None and not args.skip_initial_open:
                open_ok = gripper.open()
                print("Initial gripper open success:", open_ok)
                if not open_ok:
                    raise RuntimeError("Failed to open gripper before rollout.")

            for step_idx in range(args.steps):
                paper_state = robot.get_state()
                real_openpi_state = convert_robot_state(
                    paper_state,
                    gripper_open_width_m=config["gripper"]["open_width_m"],
                )
                if args.policy_state == "real":
                    policy_openpi_state = real_openpi_state
                elif args.policy_state == "mock":
                    policy_openpi_state = policy_mock_robot.get_state()
                else:
                    policy_openpi_state = ZERO_POLICY_STATE
                print(
                    f"Current ee_xyz_m=({real_openpi_state.ee_xyz_m[0]:+.4f}, "
                    f"{real_openpi_state.ee_xyz_m[1]:+.4f}, {real_openpi_state.ee_xyz_m[2]:+.4f})"
                    f" policy_ee_xyz_m=({policy_openpi_state.ee_xyz_m[0]:+.4f}, "
                    f"{policy_openpi_state.ee_xyz_m[1]:+.4f}, {policy_openpi_state.ee_xyz_m[2]:+.4f})"
                )
                frame = camera.capture_frame()
                rgb = rgb_from_camera_frame(frame.rgb_path_hint)
                obs = make_droid_observation(
                    exterior_image=rgb,
                    wrist_image=rgb,
                    robot_state=policy_openpi_state,
                    prompt=instruction,
                    image_size=args.image_size,
                )

                started = time.monotonic()
                response = client.infer(obs)
                chunk = extract_action_chunk(response)
                raw_actions = adapter.chunk_to_actions(response)
                print(f"\n[step {step_idx + 1}/{args.steps}] rgb={frame.rgb_path_hint} chunk_shape={tuple(chunk.shape)}")
                print(chunk[: args.print_rows])

                for chunk_idx, raw_action in enumerate(raw_actions[: args.chunk_steps]):
                    mapped_action = _apply_axis_mapping(
                        raw_action,
                        axis_order=axis_order,
                        axis_signs=axis_signs,
                        axis_scales=axis_scales,
                        yaw_sign=yaw_sign,
                        yaw_scale=args.policy_yaw_scale,
                        zero_z=args.zero_policy_z,
                    )
                    current_paper_state = robot.get_state()
                    current_openpi_state = convert_robot_state(
                        current_paper_state,
                        gripper_open_width_m=config["gripper"]["open_width_m"],
                    )
                    safe = safety.filter(mapped_action, current_openpi_state, action_timestamp=started)
                    dx, dy, dz = safe.action.dx, safe.action.dy, safe.action.dz
                    _, yaw_deg = action7_to_kinova_delta(safe.action)

                    print(
                        f"  chunk={chunk_idx}"
                        f" raw7={tuple(round(v, 6) for v in raw_action.as_tuple())}"
                        f" mapped7={tuple(round(v, 6) for v in mapped_action.as_tuple())}"
                    )
                    print(
                        f"  safe delta_xyz_m=({dx:+.4f}, {dy:+.4f}, {dz:+.4f})"
                        f" dyaw_deg={yaw_deg:+.3f}"
                        f" gripper={safe.action.gripper:.1f}"
                        f" clipped={safe.clipped}"
                        f" stop={safe.stop}"
                    )
                    if safe.reason:
                        print("  safety:", safe.reason)
                    if safe.stop:
                        robot.stop() if hasattr(robot, "stop") else None
                        return

                    if args.execute:
                        robot.move_cartesian_delta((dx, dy, dz), yaw_deg)
                        if gripper is not None:
                            if args.gripper_mode == "open_close":
                                if safe.action.gripper >= args.close_threshold:
                                    gripper.close()
                                else:
                                    gripper.open()
                            elif args.gripper_mode == "close_only" and safe.action.gripper >= args.close_threshold:
                                if step_idx + 1 < args.min_steps_before_close:
                                    print(
                                        "  close signal ignored:"
                                        f" step {step_idx + 1} < min_steps_before_close {args.min_steps_before_close}"
                                    )
                                else:
                                    if args.settle_before_close_s > 0.0:
                                        time.sleep(args.settle_before_close_s)
                                    close_ok = gripper.close()
                                    print("  close_only gripper close executed:", close_ok)
                                    if close_ok and args.lift_after_close_m > 0.0:
                                        robot.move_cartesian_delta((0.0, 0.0, args.lift_after_close_m), 0.0)
                                        print(f"  lift executed: {args.lift_after_close_m:.3f} m")
                                    print("  grasp rollout finished.")
                                    return
                    elif hasattr(robot, "apply_action"):
                        robot.apply_action(safe.action)
                    if args.policy_state == "mock":
                        policy_mock_robot.apply_action(safe.action)
        finally:
            if gripper is not None:
                gripper.shutdown()


if __name__ == "__main__":
    main()
