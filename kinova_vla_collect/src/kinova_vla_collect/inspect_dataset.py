from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class InspectConfig:
    dataset_root: Path
    task_name: str
    output_dir: Path | None
    expected_action_dim: int = 4
    expected_state_dim: int = 14
    noop_threshold: float = 1e-6
    excessive_noop_ratio: float = 0.8
    random_seed: int = 0
    max_preview_frames: int = 120


def inspect_dataset(config: InspectConfig) -> int:
    task_dir = config.dataset_root / config.task_name
    output_dir = config.output_dir or task_dir / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(path for path in task_dir.glob("episode_*") if path.is_dir()) if task_dir.exists() else []
    report: dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "task_name": config.task_name,
        "task_dir": str(task_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(episode_dirs),
        "num_success": 0,
        "episodes": [],
        "aggregate": {},
        "artifacts": {},
        "errors": [],
        "warnings": [],
    }

    if not task_dir.exists():
        report["errors"].append(f"Task directory not found: {task_dir}")
        _write_report(output_dir / "report.json", report)
        print(f"Task directory not found: {task_dir}")
        return 1

    all_actions: list[FloatArray] = []
    valid_episode_dirs: list[Path] = []
    print(f"Dataset: {task_dir}")
    print(f"Episodes: {len(episode_dirs)}")

    for episode_dir in episode_dirs:
        episode_report = inspect_episode(episode_dir, config)
        report["episodes"].append(episode_report)
        if episode_report.get("success") is True:
            report["num_success"] += 1
        if not episode_report["errors"] and episode_report["num_steps"] > 0:
            valid_episode_dirs.append(episode_dir)
            actions = np.asarray(episode_report.pop("_actions_for_aggregate"), dtype=np.float32)
            all_actions.append(actions)
        else:
            episode_report.pop("_actions_for_aggregate", None)

        gripper_ratios = episode_report["gripper_ratios"]
        print(
            f"- {episode_dir.name}: steps={episode_report['num_steps']} "
            f"success={episode_report.get('success')} "
            f"noop={episode_report['noop_ratio']:.3f} "
            f"gripper(open/hold/close)="
            f"{gripper_ratios['open']:.2f}/{gripper_ratios['hold']:.2f}/{gripper_ratios['close']:.2f} "
            f"errors={len(episode_report['errors'])} warnings={len(episode_report['warnings'])}"
        )

    if all_actions:
        concatenated_actions = np.concatenate(all_actions, axis=0)
        report["aggregate"] = summarize_actions(concatenated_actions, config.noop_threshold)
        print_action_summary("Aggregate action", report["aggregate"])
    else:
        report["warnings"].append("No valid actions found for aggregate statistics")

    print(f"Success episodes: {report['num_success']}/{report['num_episodes']}")

    sampled_episode = _sample_episode(valid_episode_dirs, config.random_seed)
    if sampled_episode is not None:
        gif_path = output_dir / f"{sampled_episode.name}_preview.gif"
        plot_path = output_dir / f"{sampled_episode.name}_actions.png"
        preview_ok = make_episode_gif(sampled_episode, gif_path, config.max_preview_frames)
        plot_ok = plot_episode_actions(sampled_episode, plot_path)
        report["artifacts"] = {
            "sampled_episode": sampled_episode.name,
            "preview_gif": str(gif_path) if preview_ok else None,
            "action_plot": str(plot_path) if plot_ok else None,
        }
        if preview_ok:
            print(f"Preview GIF: {gif_path}")
        if plot_ok:
            print(f"Action plot: {plot_path}")
        else:
            report["warnings"].append("Action plot was not generated; matplotlib may be unavailable")
    else:
        report["warnings"].append("No valid episode available for preview/plot artifacts")

    total_errors = sum(len(item["errors"]) for item in report["episodes"]) + len(report["errors"])
    total_warnings = sum(len(item["warnings"]) for item in report["episodes"]) + len(report["warnings"])
    report["num_errors"] = total_errors
    report["num_warnings"] = total_warnings
    _write_report(output_dir / "report.json", report)
    print(f"Report: {output_dir / 'report.json'}")
    print(f"Inspection complete. errors={total_errors} warnings={total_warnings}")
    return 0 if total_errors == 0 else 2


def inspect_episode(episode_dir: Path, config: InspectConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    meta_path = episode_dir / "meta.json"
    steps_path = episode_dir / "steps.npz"
    frames_dir = episode_dir / "frames"

    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
        except Exception as exc:
            errors.append(f"Failed to read meta.json: {exc}")
    else:
        errors.append("Missing meta.json")

    episode_report: dict[str, Any] = {
        "episode": episode_dir.name,
        "path": str(episode_dir),
        "success": meta.get("success"),
        "num_steps": 0,
        "meta_num_steps": meta.get("num_steps"),
        "action_stats": {},
        "noop_ratio": 0.0,
        "gripper_ratios": {"open": 0.0, "hold": 0.0, "close": 0.0},
        "errors": errors,
        "warnings": warnings,
    }

    if not frames_dir.exists():
        errors.append("Missing frames directory")
    if not steps_path.exists():
        errors.append("Missing steps.npz")
        episode_report["_actions_for_aggregate"] = np.zeros((0, config.expected_action_dim), dtype=np.float32)
        return episode_report

    try:
        steps = np.load(steps_path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"Failed to read steps.npz: {exc}")
        episode_report["_actions_for_aggregate"] = np.zeros((0, config.expected_action_dim), dtype=np.float32)
        return episode_report

    try:
        required_keys = ["image_paths", "states", "actions", "timestamps", "frame_indices"]
        missing_keys = []
        for key in required_keys:
            if key not in steps.files:
                missing_keys.append(key)
        if missing_keys:
            errors.extend(f"steps.npz missing key: {key}" for key in missing_keys)
            episode_report["_actions_for_aggregate"] = np.zeros((0, config.expected_action_dim), dtype=np.float32)
            return episode_report

        image_paths = steps["image_paths"]
        states = steps["states"]
        actions = steps["actions"]
        timestamps = steps["timestamps"]
        frame_indices = steps["frame_indices"]

        num_steps = int(len(image_paths))
        episode_report["num_steps"] = num_steps
        if meta.get("num_steps") is not None and int(meta["num_steps"]) != num_steps:
            errors.append(f"meta num_steps={meta['num_steps']} but steps.npz has {num_steps}")
        if states.ndim != 2 or states.shape != (num_steps, config.expected_state_dim):
            errors.append(
                f"state dimension error: got {states.shape}, expected ({num_steps}, {config.expected_state_dim})"
            )
        if actions.ndim != 2 or actions.shape != (num_steps, config.expected_action_dim):
            errors.append(
                f"action dimension error: got {actions.shape}, expected ({num_steps}, {config.expected_action_dim})"
            )
        if timestamps.shape != (num_steps,):
            errors.append(f"timestamp shape error: got {timestamps.shape}, expected ({num_steps},)")
        elif num_steps > 1 and not np.all(np.diff(timestamps.astype(np.float64)) > 0.0):
            errors.append("timestamps are not strictly increasing")
        if frame_indices.shape != (num_steps,):
            errors.append(f"frame_indices shape error: got {frame_indices.shape}, expected ({num_steps},)")

        missing_images = []
        for image_path in image_paths:
            resolved_path = episode_dir / str(image_path)
            if not resolved_path.exists():
                missing_images.append(str(image_path))
                if len(missing_images) >= 10:
                    break
        if missing_images:
            errors.append(f"missing image paths: {missing_images}")

        if actions.ndim == 2 and actions.shape[1] == config.expected_action_dim:
            action_stats = summarize_actions(actions.astype(np.float32), config.noop_threshold)
            episode_report["action_stats"] = action_stats
            episode_report["noop_ratio"] = action_stats["noop_ratio"]
            episode_report["gripper_ratios"] = action_stats["gripper_ratios"]
            episode_report["_actions_for_aggregate"] = actions.astype(np.float32)
            if action_stats["noop_ratio"] >= config.excessive_noop_ratio and num_steps > 0:
                warnings.append(
                    f"large no-op ratio: {action_stats['noop_ratio']:.3f} >= {config.excessive_noop_ratio:.3f}"
                )
        else:
            episode_report["_actions_for_aggregate"] = np.zeros((0, config.expected_action_dim), dtype=np.float32)
    finally:
        steps.close()

    return episode_report


def summarize_actions(actions: FloatArray, noop_threshold: float) -> dict[str, Any]:
    if actions.size == 0:
        return {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "noop_ratio": 0.0,
            "gripper_ratios": {"open": 0.0, "hold": 0.0, "close": 0.0},
        }
    gripper = actions[:, 3]
    noops = (np.linalg.norm(actions[:, :3], axis=1) <= noop_threshold) & (np.abs(gripper) <= 0.5)
    total = max(1, actions.shape[0])
    return {
        "min": actions.min(axis=0).astype(float).tolist(),
        "max": actions.max(axis=0).astype(float).tolist(),
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0).astype(float).tolist(),
        "noop_ratio": float(np.mean(noops)),
        "gripper_ratios": {
            "open": float(np.sum(gripper < -0.5) / total),
            "hold": float(np.sum(np.abs(gripper) <= 0.5) / total),
            "close": float(np.sum(gripper > 0.5) / total),
        },
    }


def print_action_summary(title: str, stats: dict[str, Any]) -> None:
    labels = ["dx", "dy", "dz", "gripper"]
    print(title)
    for name, min_value, max_value, mean_value, std_value in zip(
        labels,
        stats.get("min", []),
        stats.get("max", []),
        stats.get("mean", []),
        stats.get("std", []),
    ):
        print(
            f"  {name}: min={min_value:+.6f} max={max_value:+.6f} "
            f"mean={mean_value:+.6f} std={std_value:.6f}"
        )
    ratios = stats.get("gripper_ratios", {})
    print(f"  no-op ratio: {stats.get('noop_ratio', 0.0):.3f}")
    print(
        "  gripper ratios: "
        f"open={ratios.get('open', 0.0):.3f} "
        f"hold={ratios.get('hold', 0.0):.3f} "
        f"close={ratios.get('close', 0.0):.3f}"
    )


def make_episode_gif(episode_dir: Path, output_path: Path, max_frames: int) -> bool:
    steps_path = episode_dir / "steps.npz"
    try:
        steps = np.load(steps_path, allow_pickle=False)
        image_paths = [episode_dir / str(path) for path in steps["image_paths"]]
        timestamps = steps["timestamps"].astype(np.float64) if "timestamps" in steps.files else np.array([])
    except Exception as exc:
        print(f"Warning: failed to create GIF for {episode_dir.name}: {exc}")
        return False
    finally:
        if "steps" in locals():
            steps.close()

    existing_paths = [path for path in image_paths if path.exists()]
    if not existing_paths:
        return False
    stride = max(1, len(existing_paths) // max(1, max_frames))
    sampled_paths = existing_paths[::stride][:max_frames]
    frames = [Image.open(path).convert("RGB") for path in sampled_paths]
    first_size = frames[0].size
    frames = [frame.resize(first_size) if frame.size != first_size else frame for frame in frames]
    duration_ms = _estimate_frame_duration_ms(timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()
    return True


def plot_episode_actions(episode_dir: Path, output_path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib is unavailable; skipping action plot: {exc}")
        return False

    steps_path = episode_dir / "steps.npz"
    try:
        steps = np.load(steps_path, allow_pickle=False)
        actions = steps["actions"].astype(np.float32)
        timestamps = steps["timestamps"].astype(np.float64)
    except Exception as exc:
        print(f"Warning: failed to plot actions for {episode_dir.name}: {exc}")
        return False
    finally:
        if "steps" in locals():
            steps.close()

    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 4:
        return False
    if timestamps.shape == (actions.shape[0],):
        x_values = timestamps - timestamps[0]
        x_label = "time (s)"
    else:
        x_values = np.arange(actions.shape[0])
        x_label = "frame index"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ["dx", "dy", "dz", "gripper"]
    for index, axis in enumerate(axes):
        axis.plot(x_values, actions[:, index], linewidth=1.5)
        axis.set_ylabel(labels[index])
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"Action curves: {episode_dir.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _sample_episode(episode_dirs: list[Path], random_seed: int) -> Path | None:
    if not episode_dirs:
        return None
    rng = random.Random(random_seed)
    return rng.choice(episode_dirs)


def _estimate_frame_duration_ms(timestamps: NDArray[np.float64]) -> int:
    if timestamps.shape[0] > 1:
        diffs = np.diff(timestamps)
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size > 0:
            return int(max(20, min(1000, float(np.median(positive_diffs) * 1000.0))))
    return 200


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect raw Kinova VLA dataset quality.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--task-name", "--task", dest="task_name", type=str, default="pick_up_the_red_ball")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=4)
    parser.add_argument("--expected-state-dim", type=int, default=14)
    parser.add_argument("--noop-threshold", type=float, default=1e-6)
    parser.add_argument("--excessive-noop-ratio", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-preview-frames", type=int, default=120)
    args = parser.parse_args(argv)
    config = InspectConfig(
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        output_dir=args.output_dir,
        expected_action_dim=args.expected_action_dim,
        expected_state_dim=args.expected_state_dim,
        noop_threshold=args.noop_threshold,
        excessive_noop_ratio=args.excessive_noop_ratio,
        random_seed=args.random_seed,
        max_preview_frames=args.max_preview_frames,
    )
    raise SystemExit(inspect_dataset(config))


if __name__ == "__main__":
    main()
