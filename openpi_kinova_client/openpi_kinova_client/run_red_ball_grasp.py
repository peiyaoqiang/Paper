from __future__ import annotations

import argparse
import logging
import math
import time

from .paper_stack_bridge import build_camera, build_gripper, build_robot, load_paper_config, rgb_from_camera_frame
from .red_ball_detector import detect_red_ball


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _rotate_ee_vector_to_base(
    vector_ee: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector_ee
    # q * v * q^-1, expanded for a unit quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct visual-servo grasp for the red sponge ball.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-initial-open", action="store_true")
    parser.add_argument("--target-x-frac", type=float, default=0.50)
    parser.add_argument("--target-y-frac", type=float, default=0.56)
    parser.add_argument("--center-threshold-px", type=float, default=55.0)
    parser.add_argument("--stable-steps-before-descend", type=int, default=2)
    parser.add_argument(
        "--min-descend-steps-before-close",
        type=int,
        default=2,
        help="Require at least this many descending steps before close, even if radius is large.",
    )
    parser.add_argument(
        "--close-min-radius-px",
        type=float,
        default=95.0,
        help="Only close when the detected red ball radius reaches this size.",
    )
    parser.add_argument(
        "--min-ee-z-m",
        type=float,
        default=0.58,
        help="Emergency stop if the measured end-effector z is below this height.",
    )
    parser.add_argument("--x-gain-m-per-px", type=float, default=-0.00035)
    parser.add_argument("--y-gain-m-per-px", type=float, default=0.00035)
    parser.add_argument(
        "--servo-mode",
        choices=("diagonal", "matrix"),
        default="diagonal",
        help="diagonal uses independent x/y gains; matrix uses the calibrated image-to-base coupling.",
    )
    parser.add_argument("--matrix-gain", type=float, default=0.20)
    parser.add_argument("--max-planar-step-m", type=float, default=0.010)
    parser.add_argument("--descend-step-m", type=float, default=0.006)
    parser.add_argument(
        "--approach-dx-m",
        type=float,
        default=0.0,
        help="Base-frame x approach delta applied each centered step.",
    )
    parser.add_argument(
        "--approach-dy-m",
        type=float,
        default=0.0,
        help="Base-frame y approach delta applied each centered step.",
    )
    parser.add_argument(
        "--approach-dz-m",
        type=float,
        default=None,
        help="Base-frame z approach delta applied each centered step. Defaults to -descend-step-m.",
    )
    parser.add_argument(
        "--max-approach-step-m",
        type=float,
        default=0.008,
        help="Per-axis limit for configured approach deltas.",
    )
    parser.add_argument(
        "--approach-tool-z-m",
        type=float,
        default=0.0,
        help="Additional approach delta along the current EE local +Z axis, converted to base frame.",
    )
    parser.add_argument("--lift-after-close-m", type=float, default=0.08)
    parser.add_argument("--settle-before-close-s", type=float, default=0.2)
    parser.add_argument("--twist-command-duration-s", type=float, default=None)
    parser.add_argument("--twist-stop-duration-s", type=float, default=None)
    parser.add_argument("--combined-axis-commands", action="store_true")
    parser.add_argument("--twist-command-frame", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_paper_config(args.config)
    camera = build_camera(config, node_suffix="red_ball_grasp")
    robot = build_robot(
        config,
        max_translation_step_m=args.max_planar_step_m,
        max_rotation_step_deg=0.0,
        twist_command_duration_s=args.twist_command_duration_s,
        twist_stop_duration_s=args.twist_stop_duration_s,
        combined_axis_commands=args.combined_axis_commands,
        twist_command_frame=args.twist_command_frame,
        node_suffix="red_ball_grasp",
    )
    gripper = build_gripper(config, robot)

    print("Mode:", "EXECUTE" if args.execute else "DRY-RUN")
    print("Target pixel fraction:", (args.target_x_frac, args.target_y_frac))
    print("Center threshold px:", args.center_threshold_px)
    print("Planar gains m/px:", (args.x_gain_m_per_px, args.y_gain_m_per_px))
    print("Servo mode:", args.servo_mode)
    print("Matrix gain:", args.matrix_gain)
    print("Max planar step m:", args.max_planar_step_m)
    approach_dz_m = -abs(args.descend_step_m) if args.approach_dz_m is None else args.approach_dz_m
    approach_delta_m = (
        _clip(args.approach_dx_m, args.max_approach_step_m),
        _clip(args.approach_dy_m, args.max_approach_step_m),
        _clip(approach_dz_m, args.max_approach_step_m),
    )
    print("Descend step m:", args.descend_step_m)
    print("Approach delta m:", approach_delta_m)
    print("Approach tool z m:", args.approach_tool_z_m)
    print("Max approach step m:", args.max_approach_step_m)
    print("Min descend steps before close:", args.min_descend_steps_before_close)
    print("Close min radius px:", args.close_min_radius_px)
    print("Min EE z m:", args.min_ee_z_m)
    print("Lift after close m:", args.lift_after_close_m)

    centered_streak = 0
    descend_count = 0

    try:
        if args.execute and not args.skip_initial_open:
            open_ok = gripper.open()
            print("Initial gripper open success:", open_ok)
            if not open_ok:
                raise RuntimeError("Failed to open gripper.")

        for step_idx in range(args.steps):
            state = robot.get_state()
            if state.ee_position_m[2] <= args.min_ee_z_m:
                print(
                    "Safety stop:"
                    f" ee z {state.ee_position_m[2]:.3f} m <= min-ee-z-m {args.min_ee_z_m:.3f} m"
                )
                return

            frame = camera.capture_frame()
            rgb = rgb_from_camera_frame(frame.rgb_path_hint)
            detection = detect_red_ball(rgb)

            if detection is None:
                print(f"[step {step_idx + 1}/{args.steps}] rgb={frame.rgb_path_hint} red ball not found")
                time.sleep(0.1)
                continue

            width, height = detection.image_size
            target_x = args.target_x_frac * width
            target_y = args.target_y_frac * height
            error_x = detection.center_xy[0] - target_x
            error_y = detection.center_xy[1] - target_y
            center_error = math.hypot(error_x, error_y)

            if args.servo_mode == "matrix":
                # Empirical image Jacobian from single-axis probe:
                # +base_x moves the ball roughly left/down; +base_y moves it right/down.
                # error is ball_px - target_px, so this inverse-Jacobian style mapping
                # moves the ball back toward the target point.
                dx = args.matrix_gain * (0.00060 * error_x - 0.00075 * error_y)
                dy = args.matrix_gain * (-0.00060 * error_x - 0.00075 * error_y)
            else:
                dx = error_x * args.x_gain_m_per_px
                dy = error_y * args.y_gain_m_per_px
            dx = _clip(dx, args.max_planar_step_m)
            dy = _clip(dy, args.max_planar_step_m)
            dz = 0.0

            if center_error <= args.center_threshold_px:
                centered_streak += 1
            else:
                centered_streak = 0
                descend_count = 0

            if centered_streak >= args.stable_steps_before_descend:
                approach_delta = approach_delta_m
                if abs(args.approach_tool_z_m) > 1e-9:
                    tool_z_base = _rotate_ee_vector_to_base(
                        (0.0, 0.0, 1.0),
                        state.ee_quaternion_xyzw,
                    )
                    tool_approach = tuple(args.approach_tool_z_m * component for component in tool_z_base)
                    approach_delta = (
                        approach_delta[0] + tool_approach[0],
                        approach_delta[1] + tool_approach[1],
                        approach_delta[2] + tool_approach[2],
                    )
                dx += _clip(approach_delta[0], args.max_approach_step_m)
                dy += _clip(approach_delta[1], args.max_approach_step_m)
                dz = _clip(approach_delta[2], args.max_approach_step_m)
                dx = _clip(dx, args.max_planar_step_m)
                dy = _clip(dy, args.max_planar_step_m)
                descend_count += 1

            if dz < 0.0 and state.ee_position_m[2] + dz <= args.min_ee_z_m:
                dz = max(0.0, args.min_ee_z_m - state.ee_position_m[2])
                print(
                    "  descend limited by min-ee-z-m:"
                    f" current_z={state.ee_position_m[2]:.3f} min_z={args.min_ee_z_m:.3f}"
                )

            radius_ready = detection.radius_px >= args.close_min_radius_px
            descend_ready = descend_count >= args.min_descend_steps_before_close
            close_ready = (
                centered_streak >= args.stable_steps_before_descend
                and radius_ready
                and descend_ready
            )

            print(
                f"[step {step_idx + 1}/{args.steps}] rgb={frame.rgb_path_hint}"
                f" ball={detection.center_xy} r={detection.radius_px:.1f}px"
                f" err=({error_x:+.1f}, {error_y:+.1f}) |err|={center_error:.1f}px"
                f" ee=({state.ee_position_m[0]:+.3f}, {state.ee_position_m[1]:+.3f}, {state.ee_position_m[2]:+.3f})"
            )
            print(
                f"  command delta_xyz_m=({dx:+.4f}, {dy:+.4f}, {dz:+.4f})"
                f" centered_streak={centered_streak} descend_count={descend_count}"
                f" radius_ready={radius_ready} close_ready={close_ready}"
            )

            if args.execute:
                robot.move_cartesian_delta((dx, dy, dz), 0.0)

            if close_ready:
                print(
                    "  close trigger reached:"
                    f" radius {detection.radius_px:.1f}px >= {args.close_min_radius_px:.1f}px"
                )
                if args.execute:
                    if args.settle_before_close_s > 0.0:
                        time.sleep(args.settle_before_close_s)
                    close_ok = gripper.close()
                    print("  gripper close success:", close_ok)
                    if close_ok and args.lift_after_close_m > 0.0:
                        robot.move_cartesian_delta((0.0, 0.0, args.lift_after_close_m), 0.0)
                        print(f"  lift executed: {args.lift_after_close_m:.3f} m")
                return
    finally:
        gripper.shutdown()


if __name__ == "__main__":
    main()
