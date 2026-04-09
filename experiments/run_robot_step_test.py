from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from common.types import ExecutionResult, Observation, RefinedGrasp
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe real-robot OpenVLA single-step test.")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Override the instruction from configs/default_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]

    camera = RealSenseDriver(
        RealSenseConfig(
            width=config["camera"]["width"],
            height=config["camera"]["height"],
            mode=config["camera"]["mode"],
            color_topic=config["camera"]["color_topic"],
            aligned_depth_topic=config["camera"]["aligned_depth_topic"],
            camera_info_topic=config["camera"].get("camera_info_topic", "/camera/camera/color/camera_info"),
            capture_timeout_s=config["camera"]["capture_timeout_s"],
            output_dir=config["camera"]["output_dir"],
            ros_node_name=f"{config['camera']['ros_node_name']}_robot_step_test",
        )
    )
    robot = KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=config["robot"].get("twist_command_frame", "tool_frame"),
            ros_node_name=f"{config['robot']['ros_node_name']}_robot_step_test",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=config["robot"]["twist_stop_duration_s"],
        )
    )
    policy = OpenVLAWrapper(
        OpenVLAConfig(
            model_name=config["policy"]["model_name"],
            mode=config["policy"]["mode"],
            remote_url=config["policy"]["remote_url"],
            remote_timeout_s=config["policy"]["remote_timeout_s"],
            unnorm_key=config["policy"]["unnorm_key"],
            image_input_key=config["policy"]["image_input_key"],
        )
    )
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            workspace_xyz_min=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_xyz_max=tuple(config["robot"]["workspace_xyz_max"]),
            workspace_enforced=config["robot"].get("workspace_enforced", True),
        )
    )

    before_state = robot.get_state()
    frame = camera.capture_frame()
    observation = Observation(
        instruction=instruction,
        frame=frame,
        robot_state=before_state,
    )
    policy_action = policy.predict_action(observation)
    safe_action = action_adapter.adapt(policy_action, before_state)

    print("Instruction:", instruction)
    print("RGB path:", frame.rgb_path_hint)
    print("Depth path:", frame.depth_path_hint)
    print("Before ee_position_m:", before_state.ee_position_m)
    print("Before ee_yaw_deg:", before_state.ee_yaw_deg)
    print("Policy action delta_xyz_m:", policy_action.delta_xyz_m)
    print("Policy action delta_yaw_deg:", policy_action.delta_yaw_deg)
    print("Policy action gripper_command:", policy_action.gripper_command)
    print("Policy metadata:", policy_action.metadata)
    print("Safe action delta_xyz_m:", safe_action.delta_xyz_m)
    print("Safe action delta_yaw_deg:", safe_action.delta_yaw_deg)
    print("Safe action clipped:", safe_action.clipped)
    if safe_action.rejection_reason:
        print("Safe action note:", safe_action.rejection_reason)

    robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

    after_state = robot.get_state()
    observed_delta = tuple(after - before for after, before in zip(after_state.ee_position_m, before_state.ee_position_m))
    observed_yaw_delta = after_state.ee_yaw_deg - before_state.ee_yaw_deg

    print("After ee_position_m:", after_state.ee_position_m)
    print("After ee_yaw_deg:", after_state.ee_yaw_deg)
    print("Observed ee delta:", observed_delta)
    print("Observed yaw delta:", observed_yaw_delta)

    # Keep the shape similar to other scripts for future logging extension.
    result = ExecutionResult(
        success=True,
        state_trace=["observe", "policy_predict", "action_adapt", "coarse_approach_only"],
        message="Robot step test completed",
        grasp=RefinedGrasp(
            target_xyz_m=after_state.ee_position_m,
            target_yaw_deg=after_state.ee_yaw_deg,
            grasp_width_m=after_state.gripper_opening_m,
            quality=1.0,
            source="robot_step_test",
        ),
    )
    print("Result message:", result.message)
    if trial_logger is not None:
        trial_logger.log_trial(
            instruction=instruction,
            observation=observation,
            policy_action=policy_action,
            safe_action=safe_action,
            refined_grasp=result.grasp,
            result=result,
            final_robot_state=after_state,
            metadata={
                "test_type": "robot_step_test",
                "observed_ee_delta": observed_delta,
                "observed_yaw_delta": observed_yaw_delta,
            },
        )
        print("Trial log:", trial_logger.log_path)


if __name__ == "__main__":
    main()
