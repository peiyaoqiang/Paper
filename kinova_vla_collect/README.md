# kinova_vla_collect

Kinova Gen3 + wrist-mounted Intel RealSense D435i + Xbox controller + Modbus gripper data collection scaffold for VLA training.

The first task is `pick up the red ball`. The raw action is:

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

where `dx/dy/dz` are end-effector delta translations in meters per step,
`droll/dpitch/dyaw` are end-effector delta RPY rotations in radians per step,
and `gripper` is `-1` open, `0` hold, `+1` close.

## Install

```bash
cd kinova_vla_collect
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dry-run collection

The default config uses `hardware.dry_run: true`, so it runs without real hardware.

```bash
PYTHONPATH=src python scripts/collect_pick_red_block.py --config configs/collect_pick_red_block.yaml
```

Xbox controls:

- `Start`: start recording
- `A`: finish current episode as success
- `B`: abort and discard current episode
- `Back`: stop program
- `LT`: open gripper while held
- `RT`: close gripper while held
- left stick: `dx/dy`
- right stick vertical: `dz`
- right stick horizontal: `dyaw`
- `LB` / `RB`: `droll`
- D-pad up/down: `dpitch`

If no Xbox is available in dry-run mode, the collector still runs and emits zero actions.

## Real Robot Collection

The current config is filled for the previous ROS2 Kinova twist stack and CTAG
Modbus RTU gripper. Before real collection, start the Kinova ROS2 driver and
twist controller so these interfaces are alive:

```text
/joint_states
/twist_controller/commands
TF: base_link -> end_effector_link
TF: base_link -> tool_frame
```

Then set:

```yaml
hardware:
  dry_run: false
```

Run collection:

```bash
PYTHONPATH=src python3 scripts/collect_pick_red_block.py \
  --config configs/collect_pick_red_block.yaml
```

Recommended first real run:

- Keep the workspace clear and hand near the physical E-stop.
- Test `Back` stops the program and sends zero twist.
- Start with no object and collect one short failure episode.
- Then collect `pick up the red ball` episodes with `Start` / `A` / `B`.

## Inspect dataset

```bash
PYTHONPATH=src python scripts/inspect_dataset.py --dataset-root data/raw --task pick_up_the_red_ball
```

## Replay metadata

```bash
PYTHONPATH=src python scripts/replay_episode.py data/raw/pick_up_the_red_ball/episode_000000
```

## Run VLA Policy

Deployment uses the same robot execution interface as teleoperation:

```text
robot.step_delta_action(action, dt)
```

During deployment, action comes from the OpenPI/VLA policy server instead of
the Xbox controller. Xbox is not used in `run_policy.py`.

```bash
PYTHONPATH=src python scripts/run_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-url http://127.0.0.1:8000/act \
  --task-prompt "pick up the red ball"
```

Dry-run without hardware or policy server:

```bash
PYTHONPATH=src python scripts/run_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --dry-run \
  --max-steps 20
```

The policy server request contains wrist RGB image, robot state, and task
prompt. The expected policy output is either:

```json
{"action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}
```

or an action chunk:

```json
{"actions": [[dx, dy, dz, droll, dpitch, dyaw, gripper], [dx, dy, dz, droll, dpitch, dyaw, gripper]]}
```

All deployment actions are safety-clipped before execution:

```text
dx/dy/dz: max 0.01 m per step
droll/dpitch/dyaw: max 0.0349 rad per step
gripper: clipped to [-1, 1]
```

The action definition must remain identical to collection and OpenPI
fine-tuning:

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

## Convert To LeRobot-Style Dataset

Convert a successful raw episode or a full task directory into a LeRobot-style
intermediate dataset:

```bash
PYTHONPATH=src python scripts/convert_to_lerobot.py \
  data/raw/pick_up_the_red_ball \
  --output-root data/lerobot
```

You can also convert a single episode:

```bash
PYTHONPATH=src python scripts/convert_to_lerobot.py \
  data/raw/pick_up_the_red_ball/episode_000000 \
  --output-root data/lerobot
```

The converter only includes episodes with `meta.json` field `success: true`.
If the `lerobot` package is not installed, it writes a dependency-light
intermediate format:

```text
data/lerobot/
  pick_up_the_red_ball/
    data/
      episode_000000.npz
    images/
      episode_000000/
        000000.jpg
        000001.jpg
    meta/
      info.json
```

Each shard uses LeRobot-style field names:

```text
observation.images.wrist
observation.state
action
task
timestamp
episode_index
frame_index
```

TODO: when using HuggingFace LeRobot directly, replace this intermediate writer
with `LeRobotDataset.create(...)` / `add_frame(...)` calls using the same field
names. For OpenPI fine-tuning, keep the action definition unchanged:

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

## Raw dataset format

```text
dataset_root/
  task_name/
    episode_000000/
      meta.json
      frames/
        000000.jpg
        000001.jpg
      steps.npz
```

`meta.json` contains:

```text
task, robot, camera, control_hz, action_space, action_dim, success, num_steps, created_at
```

`steps.npz` contains:

```text
image_paths, states, actions, timestamps, frame_indices
```

`image_paths` are stored as paths relative to the episode directory, for example `frames/000000.jpg`.

State vector layout:

```text
[eef_x, eef_y, eef_z,
 eef_roll, eef_pitch, eef_yaw,
 gripper_pos,
 joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
```

## Real hardware TODOs

- `configs/collect_pick_red_block.yaml` now includes the previous project's `ros2_twist` Kinova fields and CTAG Modbus RTU gripper fields.
- `src/kinova_vla_collect/kinova_robot.py`: the current implementation still executes through the Kortex-style `step_delta_action(action, dt)` backend. If you want to reuse the previous ROS2 twist stack directly, add a ROS2 backend using `joint_state_topic`, `twist_command_topic`, `base_frame`, `ee_frame`, and `twist_command_frame` from the config.
- `src/kinova_vla_collect/modbus_gripper.py`: CTAG Modbus RTU registers and motion parameters are filled from the previous code; verify `/dev/ttyUSB0`, baudrate, stroke, torque, and RS485 adapter settings on the actual machine.
- `src/kinova_vla_collect/realsense_camera.py`: verify camera serial, RGB stream format, and intrinsics logging.
