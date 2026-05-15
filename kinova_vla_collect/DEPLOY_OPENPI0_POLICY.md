# OpenPI0 / pi0 Real-Robot Deployment Local Client

This is the Ubuntu robot-laptop side only. It reuses the real-robot stack that
worked for the BC imitation-learning deployment:

- RealSense wrist RGB camera
- Kinova Gen3 ROS2 twist control
- 14-D robot state readout
- CTAG/Modbus gripper
- local safety clipping and workspace guard
- closed-loop grasp rollout logs

The GPU server should load the OpenPI0/pi0 checkpoint and expose an OpenPI
WebSocket policy endpoint. Ask Codex on the GPU server to implement that side
with the prompt at the end of this file.

## Local Setup

```bash
cd /home/pyq/Paper/kinova_vla_collect
source .venv/bin/activate
source /opt/ros/<your_ros_distro>/setup.bash
# source /home/pyq/code/ros2_kortex_ws/install/setup.bash if needed

pip install -r requirements.txt
```

Start RealSense and Kinova exactly as in the previous real-robot runbook.

## 1. Check The OpenPI0 Server

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --check-server
```

Expected: the server metadata prints successfully.

## 2. Full Software Dry Run

This uses fake camera, fake robot, fake gripper, and fake policy actions:

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-uri ws://127.0.0.1:8000 \
  --dry-run \
  --policy-dry-run \
  --max-steps 5
```

## 3. Real Observation + One Inference, No Motion

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "pick up the red ball" \
  --real \
  --smoke-test
```

Check:

- `outputs/deploy_runs/<timestamp>_openpi0_deploy/smoke_wrist.jpg`
- printed `raw action` and `safe action`
- action shape should be `(T, >=7)` or `(7,)`

## 4. Conservative Real Closed Loop

```bash
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "pick up the red ball" \
  --real \
  --hz 3 \
  --max-steps 10 \
  --chunk-steps 1 \
  --max-delta-m 0.003 \
  --gripper-mode passthrough \
  --preview
```

Press `q` in the terminal or in the preview window to stop. Use the physical
E-stop for unsafe motion. Do not run `realsense-viewer` or the ROS2 RealSense
node at the same time, because this client opens the RealSense directly.

## Observation Contract

Default `--observation-format kinova_lerobot` sends:

```text
observation.images.wrist: uint8 [image_size, image_size, 3]
observation.state: float32 [14]
task: string
prompt: string
```

The 14-D state is:

```text
[eef_x, eef_y, eef_z,
 eef_roll, eef_pitch, eef_yaw,
 gripper_pos,
 joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
```

The expected action row is:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

where gripper uses the collection convention:

```text
-1 open, 0 hold, +1 close
```

## Prompt For GPU-Server Codex

```text
我已经在本地机器人笔记本端实现了 OpenPI0/pi0 部署客户端：
/home/pyq/Paper/kinova_vla_collect/scripts/deploy_openpi0_policy.py

请你在 GPU 服务器上的 OpenPI 项目中实现/确认远程推理服务端。要求：

1. 加载我训练好的 OpenPI0/pi0 checkpoint。
2. 暴露 OpenPI WebSocket policy server，地址类似 ws://0.0.0.0:8000。
3. 兼容本地客户端发送的 observation dict：
   - observation.images.wrist: uint8 RGB, shape [224, 224, 3]
   - observation.state: float32, shape [14]
   - task 和 prompt: 字符串任务指令
4. 返回 dict，包含 actions 或 action：
   - actions: float32/float list, shape [T, 7] 或 [7]
   - 每行语义固定为 [dx, dy, dz, droll, dpitch, dyaw, gripper]
   - gripper 语义为 -1 open, 0 hold, +1 close
5. 请先给出服务端启动命令和最小可运行代码；如果 OpenPI 官方 serve_policy.py 已满足，请指出需要使用的 config、checkpoint 参数、port，以及 observation/action key 如何对齐。
6. 服务端需要打印收到的 observation keys、image shape、state shape、输出 action shape，方便第一次真机 smoke test 排查。

本地端 smoke test 命令会是：
PYTHONPATH=src python3 scripts/deploy_openpi0_policy.py \
  --config configs/collect_pick_red_block.yaml \
  --server-uri ws://GPU_SERVER_IP:8000 \
  --task-prompt "pick up the red ball" \
  --real \
  --smoke-test
```
