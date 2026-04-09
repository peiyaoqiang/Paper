from __future__ import annotations

from dataclasses import dataclass
from typing import List

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter
from common.types import ExecutionResult, Observation
from drivers.gripper_driver import GripperDriver
from drivers.kinova_driver import KinovaDriver
from drivers.realsense_driver import RealSenseDriver
from geometry.grasp_refiner import GraspRefiner
from policy.openvla_wrapper import OpenVLAWrapper


@dataclass
class TaskStateMachineConfig:
    max_steps: int
    lift_height_m: float


class TaskStateMachine:
    def __init__(
        self,
        camera: RealSenseDriver,
        robot: KinovaDriver,
        gripper: GripperDriver,
        policy: OpenVLAWrapper,
        action_adapter: ActionAdapter,
        grasp_refiner: GraspRefiner,
        config: TaskStateMachineConfig,
        trial_logger: TrialLogger | None = None,
    ) -> None:
        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.policy = policy
        self.action_adapter = action_adapter
        self.grasp_refiner = grasp_refiner
        self.config = config
        self.trial_logger = trial_logger

    def _build_observation(self, instruction: str) -> Observation:
        frame = self.camera.capture_frame()
        robot_state = self.robot.get_state()
        return Observation(instruction=instruction, frame=frame, robot_state=robot_state)

    def run_once(self, instruction: str) -> ExecutionResult:
        trace: List[str] = []

        observation = self._build_observation(instruction)
        trace.append("observe")

        policy_action = self.policy.predict_action(observation)
        trace.append("policy_predict")

        safe_action = self.action_adapter.adapt(policy_action, observation.robot_state)
        trace.append("action_adapt")

        if safe_action.gripper_command == "open":
            self.gripper.open()
            trace.append("gripper_open")

        self.robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)
        trace.append("coarse_approach")

        refined_grasp = self.grasp_refiner.refine(policy_action, observation)
        trace.append("rgbd_refine")

        current_xyz = self.robot.get_state().ee_position_m
        delta_to_refined = tuple(target - current for target, current in zip(refined_grasp.target_xyz_m, current_xyz))
        self.robot.move_cartesian_delta(delta_to_refined, 0.0)
        trace.append("final_approach")

        self.gripper.close()
        trace.append("grasp_close")

        self.robot.move_cartesian_delta((0.0, 0.0, self.config.lift_height_m), 0.0)
        trace.append("lift")

        success = refined_grasp.quality >= 0.5
        result = ExecutionResult(
            success=success,
            state_trace=trace,
            message="Closed-loop grasp completed" if success else "Closed-loop grasp failed",
            failure_reason="" if success else "Low refined grasp quality",
            grasp=refined_grasp,
        )
        if self.trial_logger is not None:
            self.trial_logger.log_trial(
                instruction=instruction,
                observation=observation,
                policy_action=policy_action,
                safe_action=safe_action,
                refined_grasp=refined_grasp,
                result=result,
                final_robot_state=self.robot.get_state(),
                metadata={"max_steps": self.config.max_steps},
            )
        return result
