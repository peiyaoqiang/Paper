from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kinova_vla_collect.recorder import (
    ACTION_KEY,
    EPISODE_INDEX_KEY,
    FRAME_INDEX_KEY,
    IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TIMESTAMP_KEY,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect kinova_vla_collect intermediate npz dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/data/lerobot"))
    parser.add_argument("--task-name", "--task", dest="task_name", type=str, default="pick_up_the_red_ball")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--max-image-checks", type=int, default=5)
    args = parser.parse_args(argv)

    report = inspect_intermediate_dataset(
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        episodes=args.episodes,
        max_image_checks=args.max_image_checks,
    )
    _print_report(report)
    output = args.output or Path(args.dataset_root) / args.task_name / "inspection" / "intermediate_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"Report: {output}")
    if report["num_errors"] > 0:
        raise SystemExit(2)


def inspect_intermediate_dataset(
    *,
    dataset_root: Path,
    task_name: str,
    episodes: list[int] | None,
    max_image_checks: int,
) -> dict[str, Any]:
    task_dir = dataset_root / task_name
    data_dir = task_dir / "data"
    if episodes is None:
        shard_paths = sorted(data_dir.glob("episode_*.npz"))
    else:
        shard_paths = [data_dir / f"episode_{index:06d}.npz" for index in episodes]

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "task_name": task_name,
        "task_dir": str(task_dir),
        "num_episodes": len(shard_paths),
        "episodes": [],
        "aggregate": {},
        "errors": [],
        "warnings": [],
    }
    if not task_dir.exists():
        report["errors"].append(f"task directory not found: {task_dir}")
        report["num_errors"] = len(report["errors"])
        report["num_warnings"] = len(report["warnings"])
        return report

    all_actions: list[np.ndarray] = []
    for shard_path in shard_paths:
        episode_report = _inspect_shard(task_dir, shard_path, max_image_checks)
        report["episodes"].append(episode_report)
        if not episode_report["errors"] and episode_report["num_steps"] > 0:
            all_actions.append(np.asarray(episode_report.pop("_actions"), dtype=np.float32))
        else:
            episode_report.pop("_actions", None)

    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
        report["aggregate"] = _summarize_actions(actions)
    else:
        report["warnings"].append("no valid actions found")

    report["num_errors"] = len(report["errors"]) + sum(len(item["errors"]) for item in report["episodes"])
    report["num_warnings"] = len(report["warnings"]) + sum(len(item["warnings"]) for item in report["episodes"])
    return report


def _inspect_shard(task_dir: Path, shard_path: Path, max_image_checks: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    episode = shard_path.stem
    result: dict[str, Any] = {
        "episode": episode,
        "path": str(shard_path),
        "num_steps": 0,
        "action_stats": {},
        "state_stats": {},
        "timestamp_dt_mean": None,
        "timestamp_dt_std": None,
        "image_checks": [],
        "errors": errors,
        "warnings": warnings,
    }
    if not shard_path.exists():
        errors.append(f"missing shard: {shard_path}")
        result["_actions"] = np.zeros((0, 7), dtype=np.float32)
        return result

    try:
        shard = np.load(shard_path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"failed to open npz: {exc}")
        result["_actions"] = np.zeros((0, 7), dtype=np.float32)
        return result

    with shard:
        required = [IMAGE_KEY, STATE_KEY, ACTION_KEY, TIMESTAMP_KEY, FRAME_INDEX_KEY, EPISODE_INDEX_KEY, TASK_KEY]
        missing = [key for key in required if key not in shard.files]
        if missing:
            errors.extend(f"missing key: {key}" for key in missing)
            result["_actions"] = np.zeros((0, 7), dtype=np.float32)
            return result

        image_paths = shard[IMAGE_KEY]
        states = shard[STATE_KEY].astype(np.float32)
        actions = shard[ACTION_KEY].astype(np.float32)
        timestamps = shard[TIMESTAMP_KEY].astype(np.float64)
        frame_indices = shard[FRAME_INDEX_KEY]
        episode_indices = shard[EPISODE_INDEX_KEY]
        tasks = shard[TASK_KEY]

        n = int(actions.shape[0]) if actions.ndim == 2 else 0
        result["num_steps"] = n
        result["_actions"] = actions

        if image_paths.shape != (n,):
            errors.append(f"image path shape mismatch: {image_paths.shape} vs ({n},)")
        if states.shape != (n, 14):
            errors.append(f"state shape mismatch: {states.shape}, expected ({n}, 14)")
        if actions.shape != (n, 7):
            errors.append(f"action shape mismatch: {actions.shape}, expected ({n}, 7)")
        if timestamps.shape != (n,):
            errors.append(f"timestamp shape mismatch: {timestamps.shape}, expected ({n},)")
        elif n > 1 and not np.all(np.diff(timestamps) > 0.0):
            errors.append("timestamps are not strictly increasing")
        if frame_indices.shape != (n,):
            errors.append(f"frame index shape mismatch: {frame_indices.shape}, expected ({n},)")
        elif n > 0 and not np.array_equal(frame_indices.astype(np.int64), np.arange(n, dtype=np.int64)):
            warnings.append("frame indices are not 0..N-1")
        if episode_indices.shape != (n,):
            errors.append(f"episode index shape mismatch: {episode_indices.shape}, expected ({n},)")
        if tasks.shape != (n,):
            errors.append(f"task shape mismatch: {tasks.shape}, expected ({n},)")

        if not np.all(np.isfinite(states)):
            errors.append("states contain NaN/Inf")
        if not np.all(np.isfinite(actions)):
            errors.append("actions contain NaN/Inf")
        if not np.all(np.isfinite(timestamps)):
            errors.append("timestamps contain NaN/Inf")

        if n > 0:
            result["action_stats"] = _summarize_actions(actions)
            result["state_stats"] = {
                "min": states.min(axis=0).astype(float).tolist(),
                "max": states.max(axis=0).astype(float).tolist(),
            }
            gripper = actions[:, 6]
            invalid_gripper = gripper[(gripper != -1.0) & (gripper != 1.0)]
            if invalid_gripper.size:
                errors.append(f"invalid gripper values: {sorted(set(float(v) for v in invalid_gripper.tolist()))}")
            zero_motion_ratio = float(np.mean(np.linalg.norm(actions[:, :6], axis=1) < 1e-6))
            result["zero_motion_ratio"] = zero_motion_ratio
            if zero_motion_ratio > 0.8:
                warnings.append(f"high zero-motion ratio: {zero_motion_ratio:.3f}")
            if timestamps.shape == (n,) and n > 1:
                dt = np.diff(timestamps)
                result["timestamp_dt_mean"] = float(np.mean(dt))
                result["timestamp_dt_std"] = float(np.std(dt))

        for image_path in list(image_paths[: max(0, max_image_checks)]):
            rel_path = str(image_path)
            abs_path = task_dir / rel_path
            check: dict[str, Any] = {"path": rel_path, "exists": abs_path.exists()}
            if abs_path.exists():
                try:
                    with Image.open(abs_path) as image:
                        check["size"] = list(image.size)
                        check["mode"] = image.mode
                except Exception as exc:
                    check["error"] = str(exc)
                    errors.append(f"failed to open image {rel_path}: {exc}")
            else:
                errors.append(f"missing image: {rel_path}")
            result["image_checks"].append(check)

    return result


def _summarize_actions(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {}
    gripper = actions[:, 6]
    return {
        "min": actions.min(axis=0).astype(float).tolist(),
        "max": actions.max(axis=0).astype(float).tolist(),
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0).astype(float).tolist(),
        "gripper_open_ratio": float(np.mean(gripper == -1.0)),
        "gripper_close_ratio": float(np.mean(gripper == 1.0)),
        "motion_abs_mean": np.mean(np.abs(actions[:, :6]), axis=0).astype(float).tolist(),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Dataset: {report['task_dir']}")
    for item in report["episodes"]:
        stats = item.get("action_stats", {})
        print(
            f"- {item['episode']}: steps={item['num_steps']} "
            f"zero_motion={item.get('zero_motion_ratio', 0.0):.3f} "
            f"dt={item.get('timestamp_dt_mean')} "
            f"gripper(open/close)="
            f"{stats.get('gripper_open_ratio', 0.0):.2f}/{stats.get('gripper_close_ratio', 0.0):.2f} "
            f"errors={len(item['errors'])} warnings={len(item['warnings'])}"
        )
        if item["errors"]:
            for error in item["errors"][:5]:
                print(f"  ERROR: {error}")
        if item["warnings"]:
            for warning in item["warnings"][:5]:
                print(f"  WARN: {warning}")
    aggregate = report.get("aggregate", {})
    if aggregate:
        labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        print("Aggregate action range:")
        for label, min_value, max_value, mean_abs in zip(
            labels,
            aggregate.get("min", []),
            aggregate.get("max", []),
            aggregate.get("motion_abs_mean", []) + [abs(aggregate.get("mean", [0] * 7)[6])],
        ):
            print(f"  {label}: min={min_value:+.5f} max={max_value:+.5f} mean_abs={mean_abs:.5f}")
    print(f"Inspection complete: errors={report['num_errors']} warnings={report['num_warnings']}")


if __name__ == "__main__":
    main()
