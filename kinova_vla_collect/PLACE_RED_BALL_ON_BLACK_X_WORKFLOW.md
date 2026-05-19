# Place Red Ball On Black X: Collection And Training Notes

This note is for the task:

```text
put the red ball on the black X
```

The local task config is:

```text
configs/collect_place_red_ball_on_black_x.yaml
```

The local dataset task name is:

```text
place_red_ball_on_black_x
```

Keep the action contract unchanged through collection, training, serving, and
deployment:

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
gripper = -1 open target, +1 close target
```

Do not introduce `0` gripper labels in collected training data. The deployment
client can tolerate policy outputs near `0`, but this collector writes target
state labels only.

## 0. Local Robot Setup

On the robot laptop:

```bash
cd /home/pyq/Paper/kinova_vla_collect
source .venv/bin/activate
source /opt/ros/<your_ros_distro>/setup.bash
# source /home/pyq/code/ros2_kortex_ws/install/setup.bash if your Kinova stack needs it
```

Before real collection, make sure these interfaces are alive:

```text
/joint_states
/twist_controller/commands
TF: base_link -> end_effector_link
TF: base_link -> tool_frame
RealSense RGB stream
CTAG gripper serial device, normally /dev/ttyUSB0
Xbox controller
```

Check the config still says:

```yaml
hardware:
  dry_run: false
```

## 1. Scene Setup

Use the same physical setup for collection and evaluation.

- Put the black X flat on the table.
- Put the red ball at varied starting positions around the workspace.
- Keep the X visible in the wrist camera for most of the episode.
- Avoid lighting changes during a single data batch.
- Vary starts deliberately: ball left/right/front/back of X, different robot
  initial poses, and slightly different approach paths.

Good episode structure:

```text
observe ball and X -> approach red ball -> close gripper -> lift -> move above X -> lower -> open gripper -> retreat
```

End the episode only after the final state is visible: the red ball should be
on or very near the black X, with the gripper moved slightly away.

## 2. Start Collection

Run:

```bash
cd /home/pyq/Paper/kinova_vla_collect
source .venv/bin/activate
PYTHONPATH=src python3 scripts/collect_pick_red_block.py \
  --config configs/collect_place_red_ball_on_black_x.yaml
```

The output directory is:

```text
datasets/data/lerobot/place_red_ball_on_black_x/
```

Current saved layout:

```text
datasets/data/lerobot/place_red_ball_on_black_x/
  data/
    episode_000000.npz
  images/
    episode_000000/
      000000.jpg
      000001.jpg
  meta/
    info.json
    summary.json
```

## 3. Xbox Controls

```text
Start: start recording
A: save current episode as success
B: save current episode as failure
Back: stop program
LT: open gripper target (-1)
RT: close gripper target (+1)
left stick: dx / dy
right stick vertical: dz
right stick horizontal: dyaw
LB / RB: droll
D-pad up / down: dpitch
```

Important: in the current collector, `B` saves a failed episode instead of
deleting it. Keep failed episodes out of the training export unless you
explicitly want to train from failures.

## 4. Recommended Collection Plan

Pilot pass:

```text
10 episodes
```

Use it to verify camera color, action direction, gripper labels, and final data
shape.

Main pass:

```text
50-100+ successful episodes
```

Target quality over quantity. For this task, low-quality data is especially
harmful because the policy must learn both object transport and the spatial
goal.

Mark `A` only when:

- the red ball is released on the black X or very close to it,
- the trajectory is smooth enough to imitate,
- the wrist camera saw the relevant objects,
- there was no collision, slip, or emergency correction,
- the gripper open/close timing is correct.

Use `B` when:

- the ball was missed or dropped,
- the X was occluded too much,
- the operator corrected with a strange recovery path,
- the episode ended before the release/retreat phase.

## 5. Quick Dataset Checks

After a collection session, inspect the generated metadata:

```bash
cd /home/pyq/Paper/kinova_vla_collect
python3 -m json.tool \
  datasets/data/lerobot/place_red_ball_on_black_x/meta/summary.json | less
```

Check episode count and frame count:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
import json
root = Path("datasets/data/lerobot/place_red_ball_on_black_x")
info = json.load(open(root / "meta/info.json", encoding="utf-8"))
summary = json.load(open(root / "meta/summary.json", encoding="utf-8"))
print("task:", info["task"])
print("episodes:", info["num_episodes"])
print("frames:", info["num_frames"])
print("summary episodes:", summary["episode_count"])
print("summary frames:", summary["total_frames"])
print("invalid gripper values:", summary["invalid_gripper_value_count"])
print("open frames:", summary["gripper_open_frames"])
print("close frames:", summary["gripper_close_frames"])
print("episode frame counts:", summary["episode_frame_counts"])
PY
```

Check all shards have the expected keys and dimensions:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
import numpy as np
root = Path("datasets/data/lerobot/place_red_ball_on_black_x")
for shard_path in sorted((root / "data").glob("episode_*.npz")):
    with np.load(shard_path, allow_pickle=False) as shard:
        actions = shard["action"]
        states = shard["observation.state"]
        tasks = shard["task"]
        image_paths = shard["observation.images.wrist"]
        assert actions.ndim == 2 and actions.shape[1] == 7, (shard_path, actions.shape)
        assert states.ndim == 2 and states.shape[1] == 14, (shard_path, states.shape)
        assert len(actions) == len(states) == len(tasks) == len(image_paths)
        assert np.all(np.isfinite(actions))
        assert np.all((actions[:, -1] == -1.0) | (actions[:, -1] == 1.0))
        for rel in image_paths[:1]:
            assert (root / str(rel)).exists(), (shard_path, rel)
        print(shard_path.name, "frames", len(actions), "task", tasks[0])
print("OK")
PY
```

For visual spot checks, open a few images:

```bash
xdg-open datasets/data/lerobot/place_red_ball_on_black_x/images/episode_000000/000000.jpg
```

The older `scripts/inspect_dataset.py` expects the legacy raw layout
`episode_xxx/meta.json + steps.npz + frames/`. The current collector writes the
LeRobot-style intermediate layout directly, so prefer the checks above.

## 6. Optional Clean Export

If you collected failed episodes with `B`, make a clean training copy that
contains only good episodes. The current intermediate `info.json` records each
episode with `success`.

Example clean export:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
import json
import shutil

src = Path("datasets/data/lerobot/place_red_ball_on_black_x")
dst = Path("datasets/data/lerobot/place_red_ball_on_black_x_success_only")
if dst.exists():
    raise SystemExit(f"Refusing to overwrite existing directory: {dst}")

info = json.load(open(src / "meta/info.json", encoding="utf-8"))
good = [ep for ep in info["episodes"] if ep.get("success") is True]

(dst / "data").mkdir(parents=True)
(dst / "images").mkdir(parents=True)
(dst / "meta").mkdir(parents=True)

new_episodes = []
total_frames = 0
for new_idx, ep in enumerate(good):
    old_name = ep["episode"]
    new_name = f"episode_{new_idx:06d}"
    shutil.copy2(src / "data" / f"{old_name}.npz", dst / "data" / f"{new_name}.npz")
    shutil.copytree(src / "images" / old_name, dst / "images" / new_name)
    item = dict(ep)
    item["episode"] = new_name
    item["episode_index"] = new_idx
    item["shard"] = f"data/{new_name}.npz"
    new_episodes.append(item)
    total_frames += int(item["num_frames"])

info["task_name"] = dst.name
info["num_episodes"] = len(new_episodes)
info["num_frames"] = total_frames
info["episodes"] = new_episodes
json.dump(info, open(dst / "meta/info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote", dst, "episodes", len(new_episodes), "frames", total_frames)
PY
```

If your training loader assumes image paths contain the same episode names as
the `.npz`, update paths inside the copied shards as well. The simplest safe
route is often to avoid `B` during final data collection and manually remove bad
episodes before training.

## 7. Move Data To The GPU Server

From the robot laptop:

```bash
cd /home/pyq/Paper/kinova_vla_collect
rsync -avh --progress \
  datasets/data/lerobot/place_red_ball_on_black_x \
  USER@GPU_SERVER_IP:/HUBU-AI093/peiyaoqiang_2025/openpi/datasets/lerobot/
```

Expected GPU path:

```text
/HUBU-AI093/peiyaoqiang_2025/openpi/datasets/lerobot/place_red_ball_on_black_x
```

On the GPU server, check:

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openpi
find datasets/lerobot/place_red_ball_on_black_x -maxdepth 3 -type f | head
python -m json.tool datasets/lerobot/place_red_ball_on_black_x/meta/info.json | head -80
```

## 8. OpenPI Training

The local repository does not contain the OpenPI training config. The previous
successful deployment metadata used:

```text
OpenPI project: /HUBU-AI093/peiyaoqiang_2025/openpi
old config name: pi0_kinova_red_ball_lora
old checkpoint pattern:
/HUBU-AI093/peiyaoqiang_2025/openpi/checkpoints/pi0_kinova_red_ball_lora/<run_name>/<step>
old dataset root:
./datasets/lerobot/kinova_red_ball
```

Create a new OpenPI config by copying the old Kinova red-ball LoRA config, then
change at least:

```text
config name: pi0_kinova_place_red_ball_on_black_x_lora
dataset repo/id/name: place_red_ball_on_black_x
dataset root: ./datasets/lerobot/place_red_ball_on_black_x
default prompt: put the red ball on the black X
camera names: wrist
state dim: 14
action dim: 7
action keys: [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Training command template on the GPU server:

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openpi
conda activate openpi

# Replace this with the actual OpenPI training entrypoint used on the server.
python scripts/train.py \
  --config-name pi0_kinova_place_red_ball_on_black_x_lora \
  --run-name place_red_ball_on_black_x_v1 \
  --dataset-root ./datasets/lerobot/place_red_ball_on_black_x
```

If the server uses a different OpenPI command, keep the same three invariants:

```text
dataset root = ./datasets/lerobot/place_red_ball_on_black_x
prompt = put the red ball on the black X
action output dim = 7 with the same action order
```

During training, watch for:

- data loader can resolve `observation.images.wrist`,
- state shape is `[T, 14]`,
- action shape is `[T, 7]`,
- image resize/crop matches deployment, normally 224 square with padding,
- gripper channel is not silently inverted,
- loss decreases without exploding,
- validation samples show reasonable open/close timing.

## 9. Start The Trained Policy Server

After training, serve the new checkpoint from the GPU server. The exact command
depends on the OpenPI server code, but it should expose a WebSocket endpoint
compatible with the local client:

```text
server URI: ws://GPU_SERVER_IP:8000
accepted keys:
  observation.images.wrist: uint8 [224, 224, 3]
  observation.state: float32 [14]
  task: string
  prompt: string
returned keys:
  actions or action, shape [T, 7] or [7]
```

Server command template:

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openpi
conda activate openpi

python scripts/serve_policy.py \
  --config-name pi0_kinova_place_red_ball_on_black_x_lora \
  --checkpoint /HUBU-AI093/peiyaoqiang_2025/openpi/checkpoints/pi0_kinova_place_red_ball_on_black_x_lora/place_red_ball_on_black_x_v1/<STEP> \
  --host 0.0.0.0 \
  --port 8000
```

The server should print received observation keys, image shape, state shape,
and returned action shape during the first smoke test.

## 10. Local Deployment Smoke Test

On the robot laptop:

```bash
cd /home/pyq/Paper/kinova_vla_collect
source .venv/bin/activate
source /opt/ros/<your_ros_distro>/setup.bash

PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_place_red_ball_on_black_x.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "put the red ball on the black X" \
  --real \
  --smoke-test
```

Check:

```text
outputs/deploy_runs/<timestamp>_openpi0_deploy/smoke_wrist.jpg
outputs/deploy_runs/<timestamp>_openpi0_deploy/smoke_result.json
```

Expected:

```text
observation keys include observation.images.wrist, observation.state, task, prompt
image shape is [640, 480, 3] locally and [224, 224, 3] in the policy observation
state shape is (14,)
action chunk shape is (T, 7) or more columns with first 7 used
safe action is finite and clipped
```

## 11. First Real Closed-Loop Test

Start conservatively:

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_place_red_ball_on_black_x.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "put the red ball on the black X" \
  --real \
  --hz 3 \
  --max-steps 10 \
  --chunk-steps 1 \
  --max-delta-m 0.003 \
  --gripper-mode passthrough \
  --preview
```

Stop with `q`; use the physical E-stop for unsafe motion.

If the first short rollout is safe, increase gradually:

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_place_red_ball_on_black_x.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "put the red ball on the black X" \
  --real \
  --hz 3 \
  --max-steps 30 \
  --chunk-steps 1 \
  --max-delta-m 0.004 \
  --gripper-mode passthrough \
  --preview
```

Only after repeated safe tests:

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_place_red_ball_on_black_x.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "put the red ball on the black X" \
  --real \
  --hz 5 \
  --max-steps 100 \
  --chunk-steps 1 \
  --max-delta-m 0.005 \
  --gripper-mode passthrough \
  --preview
```

## 12. Evaluation Log

For each run, record:

```text
checkpoint path:
number of training episodes:
number of frames:
prompt:
scene variation:
success/failure:
failure mode:
deployment log directory:
```

Common failure modes:

- approaches ball but never closes,
- closes too early before reaching the ball,
- lifts without ball,
- reaches X but does not open,
- opens before arriving at X,
- ball placement is offset from X,
- action direction is inverted,
- image color/order mismatch.

Deployment logs are saved under:

```text
outputs/deploy_runs/<timestamp>_openpi0_deploy/
```

Use `run_config.json`, `steps.jsonl`, and saved images to compare policy output
against the training action contract.

