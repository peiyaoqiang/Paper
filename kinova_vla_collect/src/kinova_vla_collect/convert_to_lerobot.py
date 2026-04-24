from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


ACTION_DEFINITION = "[dx, dy, dz, gripper]"
IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"
TIMESTAMP_KEY = "timestamp"
EPISODE_INDEX_KEY = "episode_index"
FRAME_INDEX_KEY = "frame_index"


@dataclass(frozen=True)
class ConversionResult:
    output_task_dir: Path
    converted_episodes: list[str]
    skipped_episodes: list[str]
    num_frames: int
    report_path: Path


def convert_to_lerobot_intermediate(
    input_path: Path,
    lerobot_dataset_root: Path,
    task_name: str | None = None,
    overwrite: bool = False,
) -> ConversionResult:
    episodes, resolved_task_name = _resolve_input_episodes(input_path, task_name)
    output_task_dir = lerobot_dataset_root / resolved_task_name
    data_dir = output_task_dir / "data"
    images_dir = output_task_dir / "images"
    meta_dir = output_task_dir / "meta"

    if overwrite and output_task_dir.exists():
        shutil.rmtree(output_task_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    converted: list[str] = []
    skipped: list[str] = []
    total_frames = 0
    episode_summaries: list[dict[str, Any]] = []

    lerobot_available = _is_lerobot_available()
    for episode_dir in episodes:
        meta = _read_json(episode_dir / "meta.json")
        if meta.get("success") is not True:
            skipped.append(f"{episode_dir.name}: success is not true")
            continue

        episode_index = _episode_index_from_dir(episode_dir)
        shard_path, num_frames = _convert_one_episode(
            episode_dir=episode_dir,
            output_task_dir=output_task_dir,
            episode_index=episode_index,
            task_string=str(meta.get("task", resolved_task_name)),
        )
        converted.append(episode_dir.name)
        total_frames += num_frames
        episode_summaries.append(
            {
                "episode": episode_dir.name,
                "episode_index": episode_index,
                "num_frames": num_frames,
                "shard": str(shard_path.relative_to(output_task_dir)),
            }
        )

    info = {
        "format": "kinova_vla_collect_lerobot_intermediate_npz",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_name": resolved_task_name,
        "source": str(input_path),
        "num_episodes": len(converted),
        "num_frames": total_frames,
        "converted_episodes": converted,
        "skipped_episodes": skipped,
        "episodes": episode_summaries,
        "fields": {
            IMAGE_KEY: "relative image path under output task directory",
            STATE_KEY: "float32 [T, state_dim]",
            ACTION_KEY: "float32 [T, 4]",
            TASK_KEY: "string [T]",
            TIMESTAMP_KEY: "float64 [T]",
            EPISODE_INDEX_KEY: "int32 [T]",
            FRAME_INDEX_KEY: "int32 [T]",
        },
        "action_definition": ACTION_DEFINITION,
        "action_semantics": {
            "dx": "end-effector x delta in meters per control step",
            "dy": "end-effector y delta in meters per control step",
            "dz": "end-effector z delta in meters per control step",
            "gripper": "-1 open, 0 hold, +1 close",
        },
        "openpi_note": (
            "For OpenPI fine-tuning, keep the action definition unchanged: "
            "action = [dx, dy, dz, gripper]."
        ),
        "lerobot_available": lerobot_available,
        "todo_lerobot_dataset": (
            "TODO: If the `lerobot` library is installed, replace/extend this "
            "intermediate export with HuggingFace LeRobotDataset.create(...) and "
            "add_frame(...) calls using observation.images.wrist, observation.state, "
            "action, task, timestamp, episode_index, and frame_index."
        ),
    }
    report_path = meta_dir / "info.json"
    _write_json(report_path, info)

    validate_conversion(output_task_dir)
    return ConversionResult(
        output_task_dir=output_task_dir,
        converted_episodes=converted,
        skipped_episodes=skipped,
        num_frames=total_frames,
        report_path=report_path,
    )


def validate_conversion(output_task_dir: Path) -> dict[str, Any]:
    info_path = output_task_dir / "meta" / "info.json"
    data_dir = output_task_dir / "data"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing converted metadata: {info_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing converted data directory: {data_dir}")

    info = _read_json(info_path)
    shard_paths = sorted(data_dir.glob("episode_*.npz"))
    errors: list[str] = []
    num_frames = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            required_keys = [
                IMAGE_KEY,
                STATE_KEY,
                ACTION_KEY,
                TASK_KEY,
                TIMESTAMP_KEY,
                EPISODE_INDEX_KEY,
                FRAME_INDEX_KEY,
            ]
            for key in required_keys:
                if key not in shard.files:
                    errors.append(f"{shard_path.name}: missing key {key}")
            if errors:
                continue
            image_paths = shard[IMAGE_KEY]
            states = shard[STATE_KEY]
            actions = shard[ACTION_KEY]
            timestamps = shard[TIMESTAMP_KEY]
            episode_indices = shard[EPISODE_INDEX_KEY]
            frame_indices = shard[FRAME_INDEX_KEY]

            length = len(image_paths)
            num_frames += length
            if states.ndim != 2 or states.shape[0] != length:
                errors.append(f"{shard_path.name}: invalid state shape {states.shape}")
            if actions.shape != (length, 4):
                errors.append(f"{shard_path.name}: invalid action shape {actions.shape}")
            if timestamps.shape != (length,):
                errors.append(f"{shard_path.name}: invalid timestamp shape {timestamps.shape}")
            if episode_indices.shape != (length,):
                errors.append(f"{shard_path.name}: invalid episode_index shape {episode_indices.shape}")
            if frame_indices.shape != (length,):
                errors.append(f"{shard_path.name}: invalid frame_index shape {frame_indices.shape}")
            if length > 1 and not np.all(np.diff(timestamps.astype(np.float64)) > 0.0):
                errors.append(f"{shard_path.name}: timestamps are not strictly increasing")
            for image_path in image_paths:
                if not (output_task_dir / str(image_path)).exists():
                    errors.append(f"{shard_path.name}: missing image {image_path}")
                    break

    expected_frames = int(info.get("num_frames", -1))
    if expected_frames != num_frames:
        errors.append(f"info.json num_frames={expected_frames}, shards contain {num_frames}")
    if errors:
        raise ValueError("Converted dataset validation failed:\n" + "\n".join(errors))
    return {
        "output_task_dir": str(output_task_dir),
        "num_shards": len(shard_paths),
        "num_frames": num_frames,
        "valid": True,
    }


def _convert_one_episode(
    episode_dir: Path,
    output_task_dir: Path,
    episode_index: int,
    task_string: str,
) -> tuple[Path, int]:
    steps_path = episode_dir / "steps.npz"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing raw steps.npz: {steps_path}")

    with np.load(steps_path, allow_pickle=False) as steps:
        for key in ["image_paths", "states", "actions", "timestamps", "frame_indices"]:
            if key not in steps.files:
                raise KeyError(f"{steps_path} missing key {key}")
        raw_image_paths = steps["image_paths"]
        states = steps["states"].astype(np.float32)
        actions = steps["actions"].astype(np.float32)
        timestamps = steps["timestamps"].astype(np.float64)
        frame_indices = steps["frame_indices"].astype(np.int32)

    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"{episode_dir.name}: expected actions [T, 4], got {actions.shape}")
    num_frames = int(actions.shape[0])
    if states.ndim != 2 or states.shape[0] != num_frames:
        raise ValueError(f"{episode_dir.name}: states rows do not match actions")
    if len(raw_image_paths) != num_frames:
        raise ValueError(f"{episode_dir.name}: image_paths length does not match actions")

    output_episode_image_dir = output_task_dir / "images" / f"episode_{episode_index:06d}"
    output_episode_image_dir.mkdir(parents=True, exist_ok=True)
    converted_image_paths: list[str] = []
    for output_frame_index, raw_image_path in enumerate(raw_image_paths):
        source_image = episode_dir / str(raw_image_path)
        if not source_image.exists():
            raise FileNotFoundError(f"{episode_dir.name}: missing image {source_image}")
        destination_relative = Path("images") / f"episode_{episode_index:06d}" / f"{output_frame_index:06d}.jpg"
        destination = output_task_dir / destination_relative
        shutil.copy2(source_image, destination)
        converted_image_paths.append(destination_relative.as_posix())

    shard_path = output_task_dir / "data" / f"episode_{episode_index:06d}.npz"
    np.savez_compressed(
        shard_path,
        **{
            IMAGE_KEY: np.array(converted_image_paths, dtype=str),
            STATE_KEY: states,
            ACTION_KEY: actions,
            TASK_KEY: np.array([task_string] * num_frames, dtype=str),
            TIMESTAMP_KEY: timestamps,
            EPISODE_INDEX_KEY: np.full((num_frames,), episode_index, dtype=np.int32),
            FRAME_INDEX_KEY: frame_indices.astype(np.int32),
        },
    )
    return shard_path, num_frames


def _resolve_input_episodes(input_path: Path, task_name: str | None) -> tuple[list[Path], str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_dir() and input_path.name.startswith("episode_"):
        resolved_task_name = task_name or input_path.parent.name
        return [input_path], resolved_task_name
    if input_path.is_dir():
        resolved_task_name = task_name or input_path.name
        episodes = sorted(path for path in input_path.glob("episode_*") if path.is_dir())
        return episodes, resolved_task_name
    raise ValueError(f"Input path must be an episode directory or task directory: {input_path}")


def _episode_index_from_dir(episode_dir: Path) -> int:
    try:
        return int(episode_dir.name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Episode directory must look like episode_000000: {episode_dir}") from exc


def _is_lerobot_available() -> bool:
    try:
        import lerobot  # noqa: F401
    except Exception:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert kinova_vla_collect raw episodes to a LeRobot-style intermediate dataset."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Raw episode directory or task directory, e.g. data/raw/pick_up_the_red_ball/episode_000000.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output LeRobot-style dataset root. The task directory is created under this root.",
    )
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = convert_to_lerobot_intermediate(
        input_path=args.input_path,
        lerobot_dataset_root=args.output_root,
        task_name=args.task_name,
        overwrite=args.overwrite,
    )
    print(f"Converted episodes: {len(result.converted_episodes)}")
    print(f"Skipped episodes: {len(result.skipped_episodes)}")
    print(f"Frames: {result.num_frames}")
    print(f"Output: {result.output_task_dir}")
    print(f"Info: {result.report_path}")


if __name__ == "__main__":
    main()
