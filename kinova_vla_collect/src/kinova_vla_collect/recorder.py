from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]

IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TIMESTAMP_KEY = "timestamp"
FRAME_INDEX_KEY = "frame_index"
EPISODE_INDEX_KEY = "episode_index"
TASK_KEY = "task"

ACTION_DEFINITION = "[dx, dy, dz, droll, dpitch, dyaw, gripper]"
ACTION_SEMANTICS = {
    "dx": "end-effector x delta in meters per control step",
    "dy": "end-effector y delta in meters per control step",
    "dz": "end-effector z delta in meters per control step",
    "droll": "end-effector roll delta in radians per control step",
    "dpitch": "end-effector pitch delta in radians per control step",
    "dyaw": "end-effector yaw delta in radians per control step",
    "gripper": "-1 desired open state, +1 desired close state",
}


@dataclass
class EpisodeRecorder:
    dataset_root: Path
    task_name: str
    task_prompt: str
    robot_name: str
    camera_name: str
    control_hz: float
    action_space: str = "delta_ee_pose_rpy_with_gripper"
    action_dim: int = 7
    jpeg_quality: int = 95
    _episode_index: int | None = field(default=None, init=False)
    _episode_image_dir: Path | None = field(default=None, init=False)
    _image_paths: list[str] = field(default_factory=list, init=False)
    _states: list[FloatArray] = field(default_factory=list, init=False)
    _actions: list[FloatArray] = field(default_factory=list, init=False)
    _timestamps: list[float] = field(default_factory=list, init=False)
    _frame_indices: list[int] = field(default_factory=list, init=False)
    _created_at: str = field(default="", init=False)
    _episode_start_monotonic: float = field(default=0.0, init=False)
    _first_timestamp_ns: int | None = field(default=None, init=False)
    _last_saved_summary: dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_image_dir

    @property
    def num_steps(self) -> int:
        return len(self._frame_indices)

    def start_episode(self, episode_index: int) -> Path:
        if self._episode_index is not None:
            raise RuntimeError("An episode is already active")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.action_dim != 7:
            raise ValueError("VLA/OpenPI collection requires 7-D actions")

        task_dir = self.dataset_root / self.task_name
        data_dir = task_dir / "data"
        images_dir = task_dir / "images"
        meta_dir = task_dir / "meta"
        episode_name = f"episode_{episode_index:06d}"
        episode_image_dir = images_dir / episode_name
        shard_path = data_dir / f"{episode_name}.npz"
        if episode_image_dir.exists() or shard_path.exists():
            raise FileExistsError(f"Episode already exists: {episode_name}")

        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        episode_image_dir.mkdir(parents=True, exist_ok=False)
        self._episode_index = episode_index
        self._episode_image_dir = episode_image_dir
        self._clear_buffers()
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._episode_start_monotonic = time.monotonic()
        self._first_timestamp_ns = None
        return episode_image_dir

    def append(
        self,
        image: ImageArray,
        state: FloatArray,
        action: FloatArray,
        timestamp: float | None = None,
    ) -> None:
        if self._episode_index is None or self._episode_image_dir is None:
            raise RuntimeError("No active episode. Call start_episode(episode_index) first.")
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")

        state_array = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if state_array.shape != (14,):
            raise ValueError(f"Expected 14-D state vector, got {state_array.shape}")
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action_array.shape}")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("State contains NaN or Inf")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Action contains NaN or Inf")
        if float(action_array[-1]) not in {-1.0, 1.0}:
            raise ValueError(
                f"Invalid gripper target {float(action_array[-1]):+.3f}; "
                "VLA/OpenPI collection requires action[-1] to be exactly -1 or +1."
            )

        frame_index = len(self._frame_indices)
        image_path = f"images/episode_{self._episode_index:06d}/{frame_index:06d}.jpg"
        Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(
            self.dataset_root / self.task_name / image_path,
            quality=self.jpeg_quality,
        )

        self._image_paths.append(image_path)
        self._states.append(state_array.copy())
        self._actions.append(action_array.copy())
        if timestamp is None:
            timestamp = time.monotonic() - self._episode_start_monotonic
        self._timestamps.append(float(timestamp))
        self._frame_indices.append(frame_index)

    def save_episode(self, success: bool, extra_meta: dict[str, Any] | None = None) -> Path:
        if self._episode_index is None:
            raise RuntimeError("No active episode")
        episode_index = self._episode_index
        shard_path = self._write_episode_npz(episode_index)
        self._write_info_json(
            latest_episode={
                "episode": f"episode_{episode_index:06d}",
                "episode_index": episode_index,
                "num_frames": self.num_steps,
                "shard": f"data/episode_{episode_index:06d}.npz",
                "success": success,
                **(extra_meta or {}),
            }
        )
        self._last_saved_summary = self.write_summary()
        self._reset_active()
        return shard_path

    def discard_episode(self) -> None:
        if self._episode_index is None:
            return
        episode_image_dir = self._episode_image_dir
        self._reset_active()
        if episode_image_dir is not None:
            shutil.rmtree(episode_image_dir, ignore_errors=True)

    def save_step(
        self,
        image: ImageArray,
        state: FloatArray,
        action: FloatArray,
        timestamp_ns: int,
        gripper_command: float | None = None,
    ) -> None:
        del gripper_command
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = int(timestamp_ns)
        timestamp_s = float(int(timestamp_ns) - self._first_timestamp_ns) * 1e-9
        self.append(image=image, state=state, action=action, timestamp=timestamp_s)

    def finish_episode(self, success: bool) -> Path:
        return self.save_episode(success=success)

    def abort_episode(self, discard: bool = True) -> Path | None:
        if self._episode_index is None:
            return None
        if discard:
            self.discard_episode()
            return None
        return self.save_episode(success=False)

    def _write_episode_npz(self, episode_index: int) -> Path:
        states = (
            np.stack(self._states).astype(np.float32)
            if self._states
            else np.zeros((0, 14), dtype=np.float32)
        )
        actions = (
            np.stack(self._actions).astype(np.float32)
            if self._actions
            else np.zeros((0, self.action_dim), dtype=np.float32)
        )
        gripper = actions[:, -1] if actions.size else np.zeros((0,), dtype=np.float32)
        invalid_gripper = gripper[(gripper != -1.0) & (gripper != 1.0)]
        if np.any(gripper == 0.0):
            raise ValueError("Episode contains action[-1] == 0. VLA gripper labels must be target states.")
        if invalid_gripper.size:
            raise ValueError(f"Episode contains invalid gripper targets: {invalid_gripper.tolist()}")

        task = np.array([self.task_prompt] * len(self._frame_indices), dtype=str)
        shard_path = self.dataset_root / self.task_name / "data" / f"episode_{episode_index:06d}.npz"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shard_path,
            **{
                IMAGE_KEY: np.array(self._image_paths, dtype=str),
                STATE_KEY: states,
                ACTION_KEY: actions,
                TIMESTAMP_KEY: np.array(self._timestamps, dtype=np.float64),
                FRAME_INDEX_KEY: np.array(self._frame_indices, dtype=np.int32),
                EPISODE_INDEX_KEY: np.full((len(self._frame_indices),), episode_index, dtype=np.int32),
                TASK_KEY: task,
            },
        )
        return shard_path

    def _write_info_json(self, latest_episode: dict[str, Any] | None = None) -> Path:
        task_dir = self.dataset_root / self.task_name
        info_path = task_dir / "meta" / "info.json"
        existing_episodes: list[dict[str, Any]] = []
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as file:
                old_info = json.load(file)
            old_episodes = old_info.get("episodes", [])
            if isinstance(old_episodes, list):
                existing_episodes = [item for item in old_episodes if isinstance(item, dict)]

        if latest_episode is not None:
            existing_episodes = [
                item for item in existing_episodes if item.get("episode_index") != latest_episode["episode_index"]
            ]
            existing_episodes.append(latest_episode)
        existing_episodes.sort(key=lambda item: int(item.get("episode_index", -1)))

        num_frames = int(sum(int(item.get("num_frames", 0)) for item in existing_episodes))
        info: dict[str, Any] = {
            "format": "kinova_vla_collect_lerobot_intermediate_npz",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_name": self.task_name,
            "task": self.task_prompt,
            "robot": self.robot_name,
            "camera": self.camera_name,
            "control_hz": self.control_hz,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "num_episodes": len(existing_episodes),
            "num_frames": num_frames,
            "action_definition": ACTION_DEFINITION,
            "action_semantics": ACTION_SEMANTICS,
            "fields": {
                IMAGE_KEY: "string relative image path under the task directory",
                STATE_KEY: "float32 [T, 14]",
                ACTION_KEY: "float32 [T, 7]",
                TIMESTAMP_KEY: "float64 [T], seconds from episode start",
                FRAME_INDEX_KEY: "int32 [T]",
                EPISODE_INDEX_KEY: "int32 [T]",
                TASK_KEY: "string [T]",
            },
            "episodes": existing_episodes,
        }
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with info_path.open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2, ensure_ascii=False)
        return info_path

    def build_summary(self) -> dict[str, Any]:
        task_dir = self.dataset_root / self.task_name
        shard_paths = sorted((task_dir / "data").glob("episode_*.npz"))
        all_actions: list[FloatArray] = []
        episode_frame_counts: dict[str, int] = {}
        durations: list[float] = []
        has_nan_or_inf = False
        invalid_gripper_values: list[float] = []
        total_frames = 0

        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                actions = shard[ACTION_KEY].astype(np.float32)
                states = shard[STATE_KEY].astype(np.float32)
                timestamps = shard[TIMESTAMP_KEY].astype(np.float64)
                total_frames += int(actions.shape[0])
                episode_frame_counts[shard_path.stem] = int(actions.shape[0])
                all_actions.append(actions)
                has_nan_or_inf = has_nan_or_inf or not np.all(np.isfinite(actions))
                has_nan_or_inf = has_nan_or_inf or not np.all(np.isfinite(states))
                has_nan_or_inf = has_nan_or_inf or not np.all(np.isfinite(timestamps))
                if timestamps.size > 1:
                    durations.append(float(timestamps[-1] - timestamps[0]))
                if actions.size:
                    gripper = actions[:, -1]
                    invalid = gripper[(gripper != -1.0) & (gripper != 1.0)]
                    invalid_gripper_values.extend(float(value) for value in invalid.tolist())

        actions_all = (
            np.concatenate(all_actions, axis=0).astype(np.float32)
            if all_actions
            else np.zeros((0, self.action_dim), dtype=np.float32)
        )
        if actions_all.size:
            action_min = actions_all.min(axis=0).astype(float).tolist()
            action_max = actions_all.max(axis=0).astype(float).tolist()
            action_mean = actions_all.mean(axis=0).astype(float).tolist()
            action_std = actions_all.std(axis=0).astype(float).tolist()
            open_frames = int(np.sum(actions_all[:, -1] == -1.0))
            close_frames = int(np.sum(actions_all[:, -1] == 1.0))
        else:
            action_min = []
            action_max = []
            action_mean = []
            action_std = []
            open_frames = 0
            close_frames = 0

        if any(value == 0.0 for value in invalid_gripper_values):
            raise ValueError("Dataset contains action[-1] == 0. VLA gripper labels must be target states.")
        if invalid_gripper_values:
            raise ValueError(f"Dataset contains invalid gripper targets: {invalid_gripper_values}")

        total_duration = float(sum(max(0.0, duration) for duration in durations))
        stepped_frames = sum(max(0, count - 1) for count in episode_frame_counts.values())
        average_fps = float(stepped_frames / total_duration) if total_duration > 1e-9 else 0.0
        return {
            "task_name": self.task_name,
            "episode_count": len(shard_paths),
            "num_episodes": len(shard_paths),
            "total_frames": total_frames,
            "num_frames": total_frames,
            "average_fps": average_fps,
            "action_min": action_min,
            "action_max": action_max,
            "action_mean": action_mean,
            "action_std": action_std,
            "gripper_open_frames": open_frames,
            "gripper_close_frames": close_frames,
            "episode_frame_counts": episode_frame_counts,
            "has_nan_or_inf": bool(has_nan_or_inf),
            "invalid_gripper_value_count": len(invalid_gripper_values),
            "invalid_gripper_values": sorted(set(invalid_gripper_values)),
        }

    def write_summary(self) -> dict[str, Any]:
        summary = self.build_summary()
        summary_path = self.dataset_root / self.task_name / "meta" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        return summary

    def _reset_active(self) -> None:
        self._episode_index = None
        self._episode_image_dir = None
        self._created_at = ""
        self._episode_start_monotonic = 0.0
        self._first_timestamp_ns = None
        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self._image_paths.clear()
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._frame_indices.clear()


def _fake_image(height: int = 120, width: int = 160, frame_index: int = 0) -> ImageArray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image[:, :, 2] = np.uint8((frame_index * 30) % 255)
    return image


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a dry-run fake episode.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/data/lerobot"))
    parser.add_argument("--task-name", type=str, default="pick_up_the_red_ball")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args(argv)

    recorder = EpisodeRecorder(
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        task_prompt="pick up the red ball",
        robot_name="Kinova Gen3 dry-run",
        camera_name="RealSense D435i dry-run",
        control_hz=5.0,
        action_space="delta_ee_pose_rpy_with_gripper",
        action_dim=7,
    )
    episode_dir = recorder.start_episode(args.episode_index)
    for frame_index in range(args.num_steps):
        recorder.append(
            image=_fake_image(frame_index=frame_index),
            state=np.full((14,), frame_index, dtype=np.float32),
            action=np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            timestamp=frame_index / 5.0,
        )
    saved_dir = recorder.save_episode(
        success=True,
        extra_meta={"dry_run": True, "note": "recorder self-test episode"},
    )
    print(f"Saved fake episode: {saved_dir}")
    print(f"Frames: {len(list(episode_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
