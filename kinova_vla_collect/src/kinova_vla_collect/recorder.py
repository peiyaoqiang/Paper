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


@dataclass
class EpisodeRecorder:
    dataset_root: Path
    task_name: str
    task_prompt: str
    robot_name: str
    camera_name: str
    control_hz: float
    action_space: str = "delta_ee_position_with_gripper"
    action_dim: int = 4
    jpeg_quality: int = 95
    _episode_dir: Path | None = field(default=None, init=False)
    _frames_dir: Path | None = field(default=None, init=False)
    _image_paths: list[str] = field(default_factory=list, init=False)
    _states: list[FloatArray] = field(default_factory=list, init=False)
    _actions: list[FloatArray] = field(default_factory=list, init=False)
    _timestamps: list[float] = field(default_factory=list, init=False)
    _frame_indices: list[int] = field(default_factory=list, init=False)
    _created_at: str = field(default="", init=False)

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    @property
    def num_steps(self) -> int:
        return len(self._frame_indices)

    def start_episode(self, episode_index: int) -> Path:
        if self._episode_dir is not None:
            raise RuntimeError("An episode is already active")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        task_dir = self.dataset_root / self.task_name
        episode_dir = task_dir / f"episode_{episode_index:06d}"
        frames_dir = episode_dir / "frames"
        if episode_dir.exists():
            raise FileExistsError(f"Episode already exists: {episode_dir}")

        frames_dir.mkdir(parents=True, exist_ok=False)
        self._episode_dir = episode_dir
        self._frames_dir = frames_dir
        self._clear_buffers()
        self._created_at = datetime.now(timezone.utc).isoformat()
        return episode_dir

    def append(
        self,
        image: ImageArray,
        state: FloatArray,
        action: FloatArray,
        timestamp: float | None = None,
    ) -> None:
        if self._episode_dir is None or self._frames_dir is None:
            raise RuntimeError("No active episode. Call start_episode(episode_index) first.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")

        state_array = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if state_array.ndim != 1:
            raise ValueError(f"Expected 1D state vector, got {state_array.shape}")
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action_array.shape}")

        frame_index = len(self._frame_indices)
        image_path = f"frames/{frame_index:06d}.jpg"
        Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(
            self._episode_dir / image_path,
            quality=self.jpeg_quality,
        )

        self._image_paths.append(image_path)
        self._states.append(state_array.copy())
        self._actions.append(action_array.copy())
        self._timestamps.append(float(time.time() if timestamp is None else timestamp))
        self._frame_indices.append(frame_index)

    def save_episode(self, success: bool, extra_meta: dict[str, Any] | None = None) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("No active episode")
        episode_dir = self._episode_dir
        self._write_steps_npz(episode_dir)
        self._write_meta_json(episode_dir, success=success, extra_meta=extra_meta)
        self._reset_active()
        return episode_dir

    def discard_episode(self) -> None:
        if self._episode_dir is None:
            return
        episode_dir = self._episode_dir
        self._reset_active()
        shutil.rmtree(episode_dir, ignore_errors=True)

    def save_step(
        self,
        image: ImageArray,
        state: FloatArray,
        action: FloatArray,
        timestamp_ns: int,
        gripper_command: float | None = None,
    ) -> None:
        del gripper_command
        self.append(image=image, state=state, action=action, timestamp=float(timestamp_ns) * 1e-9)

    def finish_episode(self, success: bool) -> Path:
        return self.save_episode(success=success)

    def abort_episode(self, discard: bool = True) -> Path | None:
        if self._episode_dir is None:
            return None
        if discard:
            self.discard_episode()
            return None
        return self.save_episode(success=False)

    def _write_steps_npz(self, episode_dir: Path) -> None:
        states = (
            np.stack(self._states).astype(np.float32)
            if self._states
            else np.zeros((0, 0), dtype=np.float32)
        )
        actions = (
            np.stack(self._actions).astype(np.float32)
            if self._actions
            else np.zeros((0, self.action_dim), dtype=np.float32)
        )
        np.savez_compressed(
            episode_dir / "steps.npz",
            image_paths=np.array(self._image_paths, dtype=str),
            states=states,
            actions=actions,
            timestamps=np.array(self._timestamps, dtype=np.float64),
            frame_indices=np.array(self._frame_indices, dtype=np.int32),
        )

    def _write_meta_json(
        self,
        episode_dir: Path,
        success: bool,
        extra_meta: dict[str, Any] | None,
    ) -> None:
        meta: dict[str, Any] = {
            "task": self.task_prompt,
            "robot": self.robot_name,
            "camera": self.camera_name,
            "control_hz": self.control_hz,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "success": success,
            "num_steps": self.num_steps,
            "created_at": self._created_at,
        }
        if extra_meta:
            meta.update(extra_meta)
        with (episode_dir / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, ensure_ascii=False)

    def _reset_active(self) -> None:
        self._episode_dir = None
        self._frames_dir = None
        self._created_at = ""
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
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
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
    )
    episode_dir = recorder.start_episode(args.episode_index)
    for frame_index in range(args.num_steps):
        recorder.append(
            image=_fake_image(frame_index=frame_index),
            state=np.full((14,), frame_index, dtype=np.float32),
            action=np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32),
            timestamp=time.time(),
        )
    saved_dir = recorder.save_episode(
        success=True,
        extra_meta={"dry_run": True, "note": "recorder self-test episode"},
    )
    print(f"Saved fake episode: {saved_dir}")
    print(f"Frames: {len(list((episode_dir / 'frames').glob('*.jpg')))}")


if __name__ == "__main__":
    main()
