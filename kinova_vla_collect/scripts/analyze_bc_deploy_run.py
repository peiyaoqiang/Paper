#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing deployment log: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No steps found in {path}")
    return rows


def summarize(run_dir: Path) -> None:
    rows = load_rows(run_dir)
    states = np.asarray([row["state"] for row in rows], dtype=np.float32)
    raw = np.asarray([row["raw_action"] for row in rows], dtype=np.float32)
    safe = np.asarray([row["safe_action"] for row in rows], dtype=np.float32)
    elapsed = np.asarray([row["elapsed_ms"] for row in rows], dtype=np.float32)
    gripper_scores = [_gripper_score(row) for row in rows]
    gripper_score = np.asarray(
        [np.nan if score is None else float(score) for score in gripper_scores],
        dtype=np.float32,
    )

    print(f"Run: {run_dir}")
    print(f"Steps: {len(rows)}")
    print(
        "EEF xyz start -> end: "
        f"{_fmt(states[0, :3])} -> {_fmt(states[-1, :3])}; "
        f"delta={_fmt(states[-1, :3] - states[0, :3])}"
    )
    print(f"EEF z min/max: {float(states[:, 2].min()):.4f}/{float(states[:, 2].max()):.4f}")
    print(f"Loop ms median/max: {float(np.median(elapsed)):.1f}/{float(elapsed.max()):.1f}")
    print_action_stats("Raw action", raw)
    print_action_stats("Safe action", safe)
    if np.all(np.isnan(gripper_score)):
        print("Gripper close score: unavailable")
    else:
        finite = gripper_score[np.isfinite(gripper_score)]
        print(
            "Gripper close score min/mean/max: "
            f"{float(finite.min()):.3f}/{float(finite.mean()):.3f}/{float(finite.max()):.3f}"
        )
        close_steps = np.flatnonzero(finite >= 0.5)
        if close_steps.size:
            print(f"First close-threshold step: {int(close_steps[0])}")
        else:
            print("First close-threshold step: never reached >= 0.5")
    print_recommendations(states, raw, safe, gripper_score)


def print_action_stats(title: str, actions: np.ndarray) -> None:
    print(title)
    labels = _action_labels(actions.shape[1])
    gripper_index = _gripper_index(actions.shape[1])
    mins = actions.min(axis=0)
    maxs = actions.max(axis=0)
    means = actions.mean(axis=0)
    for index, label in enumerate(labels):
        print(
            f"  {label}: min={float(mins[index]):+.5f} "
            f"mean={float(means[index]):+.5f} max={float(maxs[index]):+.5f}"
        )
    print(f"  gripper unique: {sorted(set(float(round(v, 3)) for v in actions[:, gripper_index]))}")


def print_recommendations(
    states: np.ndarray,
    raw: np.ndarray,
    safe: np.ndarray,
    gripper_score: np.ndarray,
) -> None:
    print("Diagnosis")
    raw_gripper = raw[:, _gripper_index(raw.shape[1])]
    if np.all(raw_gripper < -0.5):
        print("- Policy always commanded open gripper; this is a policy/data/deployment-distribution issue.")
    if np.isfinite(gripper_score).any() and float(np.nanmax(gripper_score)) < 0.5:
        print("- Gripper close score never crossed 0.5, so lowering client safety will not make it close.")
    if float(np.mean(raw[:, 2] < 0.0)) > 0.7:
        print("- Policy mostly commanded downward dz; use --ee-z-min during real tests.")
    sat_ratio = float(np.mean(np.isclose(np.abs(safe[:, :3]), np.max(np.abs(safe[:, :3]), axis=0), atol=1e-6)))
    if sat_ratio > 0.5:
        print("- Many safe actions are saturated; keep max_delta_m conservative until behavior is stable.")


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"


def _gripper_score(row: dict) -> float | None:
    probs = row.get("gripper_probs")
    if isinstance(probs, list) and len(probs) == 3:
        return float(probs[2])
    if row.get("gripper_value") is not None:
        return max(0.0, float(row["gripper_value"]))
    if row.get("gripper_prob") is not None:
        return float(row["gripper_prob"])
    return None


def _gripper_index(action_dim: int) -> int:
    if action_dim == 7:
        return 6
    if action_dim == 4:
        return 3
    raise ValueError(f"Unsupported action dimension: {action_dim}")


def _action_labels(action_dim: int) -> list[str]:
    if action_dim == 7:
        return ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    if action_dim == 4:
        return ["dx", "dy", "dz", "gripper"]
    return [f"action_{index}" for index in range(action_dim)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Kinova BC deployment run log.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir)


if __name__ == "__main__":
    main()
