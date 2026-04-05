from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from calibration.tf_manager import TFConfig, TFManager
from drivers.gripper_driver import GripperConfig, GripperDriver
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from executor.task_state_machine import TaskStateMachine, TaskStateMachineConfig
from geometry.depth_filter import DepthFilter
from geometry.grasp_refiner import GraspRefiner, GraspRefinerConfig
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    config = load_config()

    camera = RealSenseDriver(
        RealSenseConfig(
            width=config["camera"]["width"],
            height=config["camera"]["height"],
        )
    )
    robot = KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
        )
    )
    gripper = GripperDriver(
        robot,
        GripperConfig(
            open_width_m=config["gripper"]["open_width_m"],
            close_width_m=config["gripper"]["close_width_m"],
        ),
    )
    policy = OpenVLAWrapper(OpenVLAConfig())
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            workspace_xyz_min=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_xyz_max=tuple(config["robot"]["workspace_xyz_max"]),
        )
    )
    tf_manager = TFManager(
        TFConfig(
            camera_to_ee_translation_m=tuple(config["calibration"]["camera_to_ee_translation_m"]),
            fx=config["camera"]["fx"],
            fy=config["camera"]["fy"],
            cx=config["camera"]["cx"],
            cy=config["camera"]["cy"],
        )
    )
    grasp_refiner = GraspRefiner(
        depth_filter=DepthFilter(),
        tf_manager=tf_manager,
        config=GraspRefinerConfig(
            approach_height_m=config["task"]["approach_height_m"],
            refine_height_m=config["task"]["refine_height_m"],
        ),
    )
    executor = TaskStateMachine(
        camera=camera,
        robot=robot,
        gripper=gripper,
        policy=policy,
        action_adapter=action_adapter,
        grasp_refiner=grasp_refiner,
        config=TaskStateMachineConfig(
            max_steps=config["task"]["max_steps"],
            lift_height_m=config["task"]["lift_height_m"],
        ),
    )

    result = executor.run_once(config["task"]["instruction"])
    print("Instruction:", config["task"]["instruction"])
    print("Success:", result.success)
    print("Trace:", " -> ".join(result.state_trace))
    if result.grasp:
        print("Refined grasp target xyz:", result.grasp.target_xyz_m)
        print("Refined grasp quality:", result.grasp.quality)
    if result.failure_reason:
        print("Failure reason:", result.failure_reason)


if __name__ == "__main__":
    main()
