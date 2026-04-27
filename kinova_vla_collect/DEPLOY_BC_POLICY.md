# Kinova BC Baseline Real-Robot Deployment

This is the Ubuntu control-PC side for the GPU-hosted BC baseline.

## 0. Start the GPU policy server

On the GPU server:

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openpi

/root/anaconda3/envs/openpi/bin/python scripts/serve_kinova_bc_policy.py \
  --checkpoint /HUBU-AI093/peiyaoqiang_2025/openpi/outputs/kinova_bc/full_99eps_64/checkpoints/best.pt \
  --host 0.0.0.0 \
  --port 8001 \
  --device cuda \
  --xyz-clip 0.01 \
  --xyz-scale 1.0
```

Wait 1-2 minutes for the model to load.

## 1. Prepare the Ubuntu control PC

Use the same ROS2/Kinova/RealSense environment that worked for data collection.

```bash
cd /home/pyq/Paper/kinova_vla_collect
source .venv/bin/activate  # if you use the project venv
source /opt/ros/<your_ros_distro>/setup.bash
# source your Kinova ROS2 workspace install/setup.bash if needed
```

Replace `GPU_SERVER_IP` below with the actual GPU server IP.

## 2. Check the HTTP service only

This does not touch the robot.

```bash
PYTHONPATH=src python3 scripts/deploy_bc_policy.py \
  --server-url http://GPU_SERVER_IP:8001 \
  --check-server
```

Expected: `/healthz` and `/metadata` JSON print successfully.

## 3. Full software dry-run

This uses fake camera, fake robot, fake gripper, but calls the real GPU policy.

```bash
PYTHONPATH=src python3 scripts/deploy_bc_policy.py \
  --server-url http://GPU_SERVER_IP:8001 \
  --task-prompt "pick up the red ball" \
  --dry-run \
  --max-steps 10 \
  --hz 3 \
  --policy-timeout-s 30
```

## 4. Real sensors + one policy inference, no robot motion

This starts RealSense, ROS2 Kinova state reading, and gripper state reading,
then sends one image/state to the GPU server. It does not execute the action.

```bash
PYTHONPATH=src python3 scripts/deploy_bc_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-url http://GPU_SERVER_IP:8001 \
  --task-prompt "pick up the red ball" \
  --real \
  --smoke-test \
  --policy-timeout-s 30
```

Check the printed `raw action` and `safe action`. Also check the saved wrist
image under `outputs/deploy_runs/.../smoke_wrist.jpg`.

## 5. First real closed-loop run

Keep the robot away from the object at first. Keep one hand near E-stop.
This starts very conservatively: 3 Hz, 3 mm max translation per step, 10 steps.

```bash
PYTHONPATH=src python3 scripts/deploy_bc_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-url http://GPU_SERVER_IP:8001 \
  --task-prompt "pick up the red ball" \
  --real \
  --hz 3 \
  --max-steps 10 \
  --max-delta-m 0.003 \
  --policy-timeout-s 30
```

Type `yes` only after checking the initial pose. Press `q` to stop from the
terminal; use the physical E-stop for any unsafe motion.

## 6. Increase cautiously

If direction, state, camera, and gripper behavior look correct:

```bash
PYTHONPATH=src python3 scripts/deploy_bc_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-url http://GPU_SERVER_IP:8001 \
  --task-prompt "pick up the red ball" \
  --real \
  --hz 5 \
  --max-steps 80 \
  --max-delta-m 0.005 \
  --policy-timeout-s 30
```

Only after repeated safe short tests should you try `--max-delta-m 0.01`.

## Notes

- The client sends RGB wrist images. Do not feed BGR images.
- The client sends the task prompt in every `/act` request. The GPU server does not need a default prompt.
- The client overwrites `state[6]` with the current gripper position, matching collection.
- The client applies a second safety limit on the Ubuntu side.
- Logs are written to `outputs/deploy_runs/<timestamp>_bc_deploy/`.
