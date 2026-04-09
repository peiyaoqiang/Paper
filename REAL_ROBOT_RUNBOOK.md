# Real Robot Runbook / 真实机器人运行说明

## Purpose / 文档目的

**English**

This runbook records the currently validated startup and testing procedure for:

- `Intel RealSense D435i` via ROS2
- remote `OpenVLA` inference service
- `Kinova Gen3` real-robot Cartesian twist control

It is intended as the operational reference for future sessions.

**中文**

本文档记录当前已经验证通过的真实系统启动与测试流程，包括：

- 通过 ROS2 启动 `Intel RealSense D435i`
- 启动远程 `OpenVLA` 推理服务
- 启动 `Kinova Gen3` 真实机械臂笛卡尔 twist 控制

后续重新启动系统时，优先参考本说明。

## Current Validated Setup / 当前已验证配置

**English**

- Local camera mode: `ros2`
- Local robot mode: `ros2_twist`
- Remote policy mode: `remote_api`
- End-effector TF frame currently used by the local driver: `end_effector_link`
- Gripper note: the real gripper is controlled separately via `Modbus`, not via the Kinova ROS controller
- Hand-eye note: the calibration in the project is the `end_effector_link -> camera_color_optical_frame` transform from `easy_handeye2`

**中文**

- 本地相机模式：`ros2`
- 本地机械臂模式：`ros2_twist`
- 远程策略模式：`remote_api`
- 本地驱动当前使用的末端 TF 帧：`end_effector_link`
- 夹爪说明：真实夹爪通过独立 `Modbus` 控制，不通过当前 Kinova ROS 控制器链路控制
- 手眼说明：项目中当前使用的是 `easy_handeye2` 标定得到的 `end_effector_link -> camera_color_optical_frame` 外参

## Recommended Startup Order / 建议启动顺序

### 1. Start RealSense / 启动 RealSense

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

Expected topics:

```bash
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
```

### 2. Start Remote OpenVLA Service / 启动远程 OpenVLA 服务

On the remote server:

在远程服务器上执行：

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openvla
uvicorn openvla_server:app --host 0.0.0.0 --port 8000
```

If you use VS Code Remote SSH, local port `8000` may be auto-forwarded by VS Code.

如果你使用 VS Code Remote SSH，本地 `8000` 端口可能会被 VS Code 自动转发。

Health check on the local machine:

在本机上健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 3. Start Kinova With Twist Controller Active / 启动 Kinova，并激活 twist 控制器

```bash
source /home/pyq/code/ros2_kortex_ws/install/setup.bash

ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 \
  dof:=7 \
  use_internal_bus_gripper_comm:=false \
  robot_controller:=twist_controller \
  robot_pos_controller:=joint_trajectory_controller \
  launch_rviz:=false
```

Then verify:

然后检查：

```bash
ros2 control list_controllers
```

Expected state:

- `twist_controller`: `active`
- `joint_trajectory_controller`: `inactive`

## Safety Stop Command / 安全停止命令

If the arm does not stop as expected, continuously publish zero twist:

如果机械臂没有按预期停住，请持续发布零速度命令：

```bash
ros2 topic pub -r 20 /twist_controller/commands geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Current Local Config Expectations / 当前本地配置要求

In `configs/default_config.json`, the following fields should remain aligned with the validated setup:

在 `configs/default_config.json` 中，以下字段应与当前已验证配置保持一致：

```json
"camera": {
  "mode": "ros2",
  "color_topic": "/camera/camera/color/image_raw",
  "aligned_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw"
}
```

```json
"robot": {
  "mode": "ros2_twist",
  "joint_state_topic": "/joint_states",
  "twist_command_topic": "/twist_controller/commands",
  "base_frame": "base_link",
  "ee_frame": "end_effector_link"
}
```

```json
"policy": {
  "mode": "remote_api",
  "remote_url": "http://127.0.0.1:8000/predict"
}
```

## Validated Test Commands / 已验证测试命令

### Minimal Real Arm Step Test / 最小实机单步测试

Use this first before running the full demo:

在运行完整 demo 前，优先使用这个脚本：

```bash
cd /home/pyq/Paper
python3 experiments/run_robot_step_test.py
```

This script:

- captures one real wrist RGB-D frame
- queries remote OpenVLA once
- applies the local action adapter
- executes one small safe robot step only
- does not run final approach, gripper close, or lift

该脚本会：

- 采集一帧真实 wrist RGB-D
- 调用一次远程 OpenVLA
- 经过本地 action adapter
- 只执行一个安全小步
- 不执行 final approach、夹爪闭合或抬升

### Manual Twist Test / 手动 twist 测试

```bash
python3 experiments/test_kinova_twist.py --dx 0.005 --duration 0.4 --stop-duration 0.8
```

If needed, increase to:

如有需要，可提高到：

```bash
python3 experiments/test_kinova_twist.py --dx 0.01 --duration 0.4 --stop-duration 0.8
```

### Full Demo / 完整 demo

Only use this after the step test is stable:

只有在单步测试稳定后再使用：

```bash
python3 experiments/run_demo.py
```

## Verified Observations / 当前已验证结论

**English**

- RealSense ROS2 image capture works
- Local wrapper sends real wrist RGB via `rgb_b64`
- Remote OpenVLA service receives and logs the real image
- Kinova real robot can execute a small twist-based step under the current setup
- The full grasp loop should still be treated as experimental and safety-constrained

**中文**

- RealSense ROS2 图像采集已跑通
- 本地 wrapper 已能通过 `rgb_b64` 发送真实 wrist RGB
- 远程 OpenVLA 服务已确认收到并记录真实图像
- 当前配置下，Kinova 真实机械臂已能执行基于 twist 的小步动作
- 完整抓取闭环仍应视为实验性流程，并保持严格安全约束

## Progress Archive / 进展归档

For the full archived summary of what has already been completed, verified, and what is still blocked, see:

关于截至当前的完整进展、实验结论、配置快照和主要瓶颈，请参考：

- `PROGRESS_ARCHIVE_2026-04-06.md`
