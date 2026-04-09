from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "analysis" / "logs" / "trial_log.jsonl"
CSV_PATH = PROJECT_ROOT / "analysis" / "logs" / "trial_summary.csv"


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def load_trials() -> list[dict]:
    if not LOG_PATH.is_file():
        return []
    trials = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    return trials


def bool_as_int(value: bool) -> int:
    return 1 if value else 0


def build_rows(trials: list[dict]) -> list[dict]:
    rows = []
    for trial in trials:
        observation = trial["observation"]
        frame = observation["frame"]
        policy_action = trial["policy_action"]
        safe_action = trial["safe_action"]
        result = trial["result"]
        metadata = policy_action.get("metadata", {})
        trial_metadata = trial.get("metadata", {})

        raw_delta = [float(v) for v in policy_action["delta_xyz_m"]]
        safe_delta = [float(v) for v in safe_action["delta_xyz_m"]]
        target_pixel = policy_action.get("target_pixel") or [None, None]

        used_real_capture = "analysis/captures/" in frame["rgb_path_hint"]
        row = {
            "timestamp_utc": trial["timestamp_utc"],
            "instruction": trial["instruction"],
            "rgb_path_hint": frame["rgb_path_hint"],
            "depth_path_hint": frame["depth_path_hint"],
            "image_width": frame["width"],
            "image_height": frame["height"],
            "used_real_capture": bool_as_int(used_real_capture),
            "camera_mode_label": "real_camera" if used_real_capture else "mock_camera",
            "test_type_label": str(trial_metadata.get("test_type", "run_demo")),
            "success": bool_as_int(bool(result["success"])),
            "safe_action_clipped": bool_as_int(bool(safe_action["clipped"])),
            "policy_confidence": float(policy_action["confidence"]),
            "policy_gripper_command": policy_action["gripper_command"],
            "raw_dx": raw_delta[0],
            "raw_dy": raw_delta[1],
            "raw_dz": raw_delta[2],
            "raw_delta_xyz_norm": vector_norm(raw_delta),
            "safe_dx": safe_delta[0],
            "safe_dy": safe_delta[1],
            "safe_dz": safe_delta[2],
            "safe_delta_xyz_norm": vector_norm(safe_delta),
            "delta_yaw_deg": float(policy_action["delta_yaw_deg"]),
            "target_pixel_x": target_pixel[0],
            "target_pixel_y": target_pixel[1],
            "refined_grasp_quality": float(trial["refined_grasp"]["quality"]),
            "preprocess_ms": metadata.get("preprocess_ms"),
            "infer_ms": metadata.get("infer_ms"),
            "total_ms": metadata.get("total_ms"),
            "model_name": metadata.get("model_name", ""),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    successes = sum(row["success"] for row in rows)
    clipped = sum(row["safe_action_clipped"] for row in rows)
    real_capture = sum(row["used_real_capture"] for row in rows)

    raw_norms = [row["raw_delta_xyz_norm"] for row in rows]
    safe_norms = [row["safe_delta_xyz_norm"] for row in rows]
    confidences = [row["policy_confidence"] for row in rows]
    qualities = [row["refined_grasp_quality"] for row in rows]
    preprocess_values = [float(row["preprocess_ms"]) for row in rows if row["preprocess_ms"] is not None]
    infer_values = [float(row["infer_ms"]) for row in rows if row["infer_ms"] is not None]
    total_values = [float(row["total_ms"]) for row in rows if row["total_ms"] is not None]

    return {
        "total": total,
        "successes": successes,
        "clipped": clipped,
        "real_capture": real_capture,
        "mean_confidence": mean(confidences),
        "mean_raw_norm": mean(raw_norms),
        "mean_safe_norm": mean(safe_norms),
        "mean_quality": mean(qualities),
        "mean_preprocess_ms": mean(preprocess_values),
        "mean_infer_ms": mean(infer_values),
        "mean_total_ms": mean(total_values),
        "latest": rows[-1],
    }


def print_section(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{title}: 0 trials")
        return

    summary = summarize_rows(rows)
    latest = summary["latest"]

    print(title)
    print(f"  Trials: {summary['total']}")
    print(f"  Success rate: {summary['successes']}/{summary['total']} = {summary['successes'] / summary['total']:.3f}")
    print(f"  Real camera trials: {summary['real_capture']}/{summary['total']} = {summary['real_capture'] / summary['total']:.3f}")
    print(f"  Safety clipped trials: {summary['clipped']}/{summary['total']} = {summary['clipped'] / summary['total']:.3f}")
    print(f"  Mean policy confidence: {format_float(summary['mean_confidence'])}")
    print(f"  Mean raw delta xyz norm: {format_float(summary['mean_raw_norm'])}")
    print(f"  Mean safe delta xyz norm: {format_float(summary['mean_safe_norm'])}")
    print(f"  Mean refined grasp quality: {format_float(summary['mean_quality'])}")
    print(f"  Mean preprocess latency ms: {format_float(summary['mean_preprocess_ms'])}")
    print(f"  Mean inference latency ms: {format_float(summary['mean_infer_ms'])}")
    print(f"  Mean total latency ms: {format_float(summary['mean_total_ms'])}")
    print("  Latest trial:")
    print(f"    timestamp_utc: {latest['timestamp_utc']}")
    print(f"    instruction: {latest['instruction']}")
    print(f"    rgb_path_hint: {latest['rgb_path_hint']}")
    print(f"    success: {latest['success']}")
    print(f"    safe_action_clipped: {latest['safe_action_clipped']}")
    print(f"    raw_delta_xyz_norm: {format_float(latest['raw_delta_xyz_norm'])}")
    print(f"    total_ms: {latest['total_ms'] if latest['total_ms'] is not None else 'n/a'}")


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print(f"No trial log found at {LOG_PATH}")
        return

    print("Trial Summary")
    print(f"Log path: {LOG_PATH}")
    print(f"CSV path: {CSV_PATH}")
    print()
    print_section("Overall", rows)
    print()
    print_section("Mock Camera", [row for row in rows if row["camera_mode_label"] == "mock_camera"])
    print()
    print_section("Real Camera", [row for row in rows if row["camera_mode_label"] == "real_camera"])
    print()
    print_section("Run Demo", [row for row in rows if row["test_type_label"] == "run_demo"])
    print()
    print_section("Robot Step Test", [row for row in rows if row["test_type_label"] == "robot_step_test"])
    print()
    print_section(
        "Robot Multistep Test",
        [row for row in rows if row["test_type_label"] == "robot_multistep_test"],
    )


def main() -> None:
    trials = load_trials()
    rows = build_rows(trials)
    write_csv(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
