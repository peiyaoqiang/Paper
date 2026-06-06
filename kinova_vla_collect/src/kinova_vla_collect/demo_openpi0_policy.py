from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinova_vla_collect.config import load_config
from kinova_vla_collect.deploy_openpi0_policy import OpenPI0DeploymentRunner, _confirm_real_run


RESET_COMMANDS = {"reset", "r", "复位", "回零"}
SET_PRESET_COMMANDS = {"set-preset", "preset", "记录预设", "设置预设"}
STATE_COMMANDS = {"state", "status", "pose", "当前位置"}
QUIT_COMMANDS = {"q", "quit", "exit", "退出"}
HELP_COMMANDS = {"h", "help", "?", "帮助"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive OpenPI0 demo: type reset to return to a preset pose, or type a task prompt to run it."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/collect_place_red_ball_on_black_x.yaml"))
    parser.add_argument("--server-uri", type=str, default="ws://127.0.0.1:8000")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--default-task", type=str, default="put the red ball on the black X")
    parser.add_argument("--dry-run", action="store_true", help="Use fake camera/robot/gripper.")
    parser.add_argument("--real", action="store_true", help="Force real camera/robot/gripper.")
    parser.add_argument("--policy-dry-run", action="store_true", help="Do not call the GPU policy server.")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--max-delta-m", type=float, default=0.003)
    parser.add_argument("--max-delta-rad", type=float, default=None)
    parser.add_argument("--xyz-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--policy-state-mode", choices=("real", "zero"), default="real")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument(
        "--gripper-mode",
        choices=("passthrough", "ignore", "close_only", "open_close"),
        default="passthrough",
    )
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--min-steps-before-close", type=int, default=5)
    parser.add_argument("--lift-after-close-m", type=float, default=0.08)
    parser.add_argument("--startup-open-s", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/deploy_runs"))
    parser.add_argument("--save-image-every", type=int, default=10)
    parser.add_argument("--connect-retry-s", type=float, default=2.0)
    parser.add_argument("--camera-color-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument(
        "--reset-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help="Preset end-effector pose in base frame. If omitted, the startup pose is captured as the preset.",
    )
    parser.add_argument("--reset-hz", type=float, default=10.0)
    parser.add_argument("--reset-timeout-s", type=float, default=45.0)
    parser.add_argument("--reset-position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--reset-rotation-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--reset-max-delta-m", type=float, default=0.006)
    parser.add_argument("--reset-max-delta-rad", type=float, default=0.05)
    parser.add_argument("--no-reset-open-gripper", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run and args.real:
        raise ValueError("Use at most one of --dry-run or --real")

    config = load_config(args.config)
    hardware_dry_run = True if args.dry_run else False if args.real else None
    runner = OpenPI0DeploymentRunner(
        config=config,
        server_uri=args.server_uri,
        task_prompt=args.default_task,
        hardware_dry_run=hardware_dry_run,
        policy_dry_run=args.policy_dry_run,
        api_key=args.api_key,
        hz=args.hz,
        max_steps=args.max_steps,
        chunk_steps=args.chunk_steps,
        max_delta_m=args.max_delta_m,
        max_delta_rad=args.max_delta_rad,
        xyz_scale=args.xyz_scale,
        rotation_scale=args.rotation_scale,
        gripper_mode=args.gripper_mode,
        invert_gripper=args.invert_gripper,
        min_steps_before_close=args.min_steps_before_close,
        lift_after_close_m=args.lift_after_close_m,
        open_gripper_on_start=True,
        startup_open_s=args.startup_open_s,
        image_size=args.image_size,
        policy_state_mode=args.policy_state_mode,
        preview=args.preview,
        preview_scale=args.preview_scale,
        save_image_every=args.save_image_every,
        log_dir=args.log_dir,
        connect_retry_s=args.connect_retry_s,
        camera_color_order=args.camera_color_order,
    )

    _confirm_real_run(argparse.Namespace(dry_run=runner.use_dry_run))
    print("Starting interactive demo session...")
    try:
        runner._connect_hardware()
        metadata = runner.policy.connect()
        print(f"OpenPI0 server connected. metadata keys: {sorted(str(key) for key in metadata.keys())}")
        if args.reset_pose is None:
            preset_pose = runner.robot.get_state().astype(np.float32)[:6].copy()
            print("Captured startup pose as reset preset.")
        else:
            preset_pose = np.asarray(args.reset_pose, dtype=np.float32)
            print("Using reset pose from --reset-pose.")
        print(f"Reset preset xyz/rpy: {preset_pose.tolist()}")
        _print_help(args.default_task)

        while True:
            command = input("\ndemo> ").strip()
            normalized = command.lower()
            if not command:
                continue
            if normalized in QUIT_COMMANDS:
                break
            if normalized in HELP_COMMANDS:
                _print_help(args.default_task)
                continue
            if normalized in SET_PRESET_COMMANDS:
                preset_pose = runner.robot.get_state().astype(np.float32)[:6].copy()
                print(f"Updated reset preset xyz/rpy: {preset_pose.tolist()}")
                continue
            if normalized in STATE_COMMANDS:
                state = runner.robot.get_state().astype(np.float32)
                state[6] = runner.gripper.get_position()
                print(f"state xyz/rpy/gripper: {state[:7].tolist()}")
                continue
            if normalized in RESET_COMMANDS:
                runner.reset_to_pose(
                    preset_pose,
                    hz=args.reset_hz,
                    timeout_s=args.reset_timeout_s,
                    position_tolerance_m=args.reset_position_tolerance_m,
                    rotation_tolerance_rad=args.reset_rotation_tolerance_rad,
                    max_delta_m=args.reset_max_delta_m,
                    max_delta_rad=args.reset_max_delta_rad,
                    open_gripper=not args.no_reset_open_gripper,
                )
                continue

            prompt = args.default_task if normalized in {"task", "run", "执行"} else command
            runner.run_prompt_once(prompt, prepare_gripper=True)
    finally:
        runner._shutdown()
        print("Interactive demo session closed.")


def _print_help(default_task: str) -> None:
    print("Commands:")
    print("  reset / 复位                 move to the preset pose")
    print("  put the red ball on the black X  run that task prompt")
    print(f"  task / run / 执行             run default prompt: {default_task!r}")
    print("  set-preset / 记录预设         capture current pose as the new reset preset")
    print("  state / pose                 print current xyz/rpy/gripper")
    print("  q / quit                     exit")


if __name__ == "__main__":
    main()
