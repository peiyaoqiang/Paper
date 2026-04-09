# Progress Archive 2026-04-06 / 进展归档 2026-04-06

## Purpose / 文档目的

**English**

This document archives the real-robot OpenVLA integration progress up to `2026-04-06`.
It is intended to replace long chat-history lookup with one compact project record.

It covers:

- validated startup commands
- completed engineering milestones
- current configuration snapshot
- experiment evidence already obtained
- current bottlenecks
- the most reasonable next step

**中文**

本文档归档截至 `2026-04-06` 的真实机器人 OpenVLA 集成进展。
目的是用一份固定文档替代长聊天记录回溯。

内容包括：

- 已验证的启动命令
- 已完成的工程里程碑
- 当前配置快照
- 已获得的实验证据
- 当前主要瓶颈
- 最合理的下一步

## Current Bottom Line / 当前阶段结论

**English**

The project has already passed the "basic chain works" stage.

What is already true:

- real `RealSense` RGB-D capture works
- local code can call remote `OpenVLA`
- `OpenVLA` outputs change with language and scene
- the real `Kinova` arm can execute safety-limited small steps
- hand-eye calibration has been integrated into the project
- target-centered evaluation scripts now exist

What is not yet true:

- stable 3D approach to the target is not solved yet
- `z` refinement is still unreliable
- twist-based Cartesian control is still too weak for robust visible approach
- full autonomous grasping is not ready yet

**中文**

项目已经通过了“基础链路打通”阶段。

目前已经成立的事情：

- 真实 `RealSense` RGB-D 采集已打通
- 本地代码已能调用远程 `OpenVLA`
- `OpenVLA` 输出会随语言和场景变化
- 真实 `Kinova` 机械臂已能执行安全限制下的小步动作
- 手眼标定已经接入项目
- 已有目标居中/接近评测脚本

目前还没有成立的事情：

- 稳定的 3D 接近还没有解决
- `z` 方向 refine 还不够可靠
- 基于 twist 的笛卡尔控制对实机来说仍偏弱
- 完整自主抓取还未达到可用状态

## Validated Startup Commands / 已验证启动命令

### 1. RealSense / 相机

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

### 2. Remote OpenVLA / 远程 OpenVLA

Remote server:

```bash
cd /HUBU-AI093/peiyaoqiang_2025/openvla
uvicorn openvla_server:app --host 0.0.0.0 --port 8000
```

Local health check:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Kinova With Twist Controller / Kinova 启动为 twist 控制模式

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

Controller check:

```bash
ros2 control list_controllers
```

Expected:

- `twist_controller`: `active`
- `joint_trajectory_controller`: `inactive`

### 4. Emergency Stop / 紧急停车

```bash
ros2 topic pub -r 20 /twist_controller/commands geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Completed Milestones / 已完成里程碑

### A. Remote OpenVLA Path / 远程 OpenVLA 路径

Completed:

- remote `FastAPI` OpenVLA service can be called from local code
- local wrapper sends real wrist RGB via `rgb_b64`
- latency metadata is logged

Relevant files:

- [policy/openvla_wrapper.py](/home/pyq/Paper/policy/openvla_wrapper.py)
- [policy/REMOTE_API_EXAMPLE.md](/home/pyq/Paper/policy/REMOTE_API_EXAMPLE.md)

### B. RealSense ROS2 Integration / RealSense ROS2 接入

Completed:

- RGB topic subscription works
- aligned depth topic subscription works
- images are saved to `analysis/captures/*.png`
- depths are saved to `analysis/captures/*.npy`

Relevant files:

- [drivers/realsense_driver.py](/home/pyq/Paper/drivers/realsense_driver.py)

### C. Real Kinova Twist Driver / 真实 Kinova Twist 驱动

Completed:

- `/joint_states` subscription works
- TF lookup works
- `/twist_controller/commands` publishing works
- active zero-twist braking logic has been added

Important practical finding:

- the controller can move the arm
- but small twist commands are often too weak to produce clearly visible end-effector motion
- latest single-axis tests on `2026-04-06` show axis coupling, sign inconsistency, and much smaller-than-commanded motion
- local inspection of `ros2_kortex` shows the Kinova driver hard-codes twist commands to `CARTESIAN_REFERENCE_FRAME_TOOL`
- this means `/twist_controller/commands` should be interpreted in the tool frame, not directly as base-frame Cartesian deltas

Relevant files:

- [drivers/kinova_driver.py](/home/pyq/Paper/drivers/kinova_driver.py)
- [experiments/test_kinova_twist.py](/home/pyq/Paper/experiments/test_kinova_twist.py)
- `/home/pyq/code/ros2_kortex_ws/src/ros2_kortex/kortex_driver/src/hardware_interface.cpp`
- `/home/pyq/code/ros2_kortex_ws/src/ros2_kortex/kortex_description/arms/gen3/7dof/config/ros2_controllers.yaml`

### D. Logging And Summaries / 日志与统计

Completed:

- trial logger exists
- JSONL logs are being written
- summary script groups multiple experiment types

Relevant files:

- [analysis/trial_logger.py](/home/pyq/Paper/analysis/trial_logger.py)
- [analysis/summarize_trials.py](/home/pyq/Paper/analysis/summarize_trials.py)
- [analysis/logs/trial_log.jsonl](/home/pyq/Paper/analysis/logs/trial_log.jsonl)

### E. Hand-Eye Integration / 手眼标定接入

Completed:

- real calibration file located and verified
- calibration values copied into project config
- transform direction bug was corrected
- end-effector frame mismatch was identified and corrected

Calibration source:

- `/home/pyq/.ros2/easy_handeye2/calibrations/kinova_d435_eye_in_hand.calib`

Calibration values in use:

- translation:
  - `x = 0.02816549492211143`
  - `y = -0.06079202177808648`
  - `z = -0.03017898793502192`
- quaternion:
  - `x = 0.011321934239452403`
  - `y = -0.003620687265627698`
  - `z = 0.34158486052971326`
  - `w = 0.9397757644702796`

Important finding:

- calibration is published relative to `end_effector_link`
- using `bracelet_link` in the local project was wrong for refine
- local config now uses `end_effector_link`

Relevant files:

- [calibration/tf_manager.py](/home/pyq/Paper/calibration/tf_manager.py)
- [configs/default_config.json](/home/pyq/Paper/configs/default_config.json)

## Experiment Scripts Added / 已增加实验脚本

### Stable utility scripts / 已有工具脚本

- [experiments/test_kinova_twist.py](/home/pyq/Paper/experiments/test_kinova_twist.py)
- [experiments/run_robot_step_test.py](/home/pyq/Paper/experiments/run_robot_step_test.py)
- [experiments/run_robot_multistep_test.py](/home/pyq/Paper/experiments/run_robot_multistep_test.py)
- [experiments/run_policy_compare.py](/home/pyq/Paper/experiments/run_policy_compare.py)
- [experiments/run_target_reach_test.py](/home/pyq/Paper/experiments/run_target_reach_test.py)
- [experiments/run_refine_preview.py](/home/pyq/Paper/experiments/run_refine_preview.py)
- [experiments/run_planar_coarse_approach_test.py](/home/pyq/Paper/experiments/run_planar_coarse_approach_test.py)

### Their roles / 作用说明

- `run_robot_step_test.py`
  - one safe OpenVLA step on the real arm
- `run_robot_multistep_test.py`
  - repeated safe OpenVLA steps
- `run_policy_compare.py`
  - frozen-image instruction comparison
- `run_target_reach_test.py`
  - measures whether the ball gets closer to image center
- `run_refine_preview.py`
  - prints the full 3D refine chain without executing final approach
- `run_planar_coarse_approach_test.py`
  - uses refined target in `x/y`, with optional guarded `z`

## Strong Experimental Findings / 已获得的重要实验结论

### 1. OpenVLA is not fixed-output / OpenVLA 不是固定输出

Evidence:

- under the same frozen image, changing instructions such as
  - `pick up the red ball`
  - `pick up the green ball`
  - `pick up the ball`
  changes `Policy delta_xyz_m` significantly

Interpretation:

- OpenVLA is using language and scene jointly
- it is not simply replaying a fixed Cartesian action

### 2. Real robot execution chain is live / 真实机械臂执行链已通

Evidence:

- robot step and multistep tests execute successfully
- controllers, image capture, inference, and logging all work together

Interpretation:

- the project has moved beyond mock-only status

### 3. Heavy clipping is still the dominant behavior / 大量裁剪仍是主现象

Latest summary snapshot:

- `Overall Trials: 90`
- `Overall Success rate: 81/90 = 0.900`
- `Real Camera Safety clipped trials: 86/87 = 0.989`
- `Mean total latency ms: 1321.546`

Interpretation:

- raw OpenVLA outputs are still much larger than the currently safe action space
- safe action adaptation remains a central research point

### 4. Green-ball target reach succeeded once clearly, but is not yet stable / 绿球目标接近出现过明确成功，但尚不稳定

One strong positive case:

- initial center distance reduced from about `150.79 px` to `91.10 px`

Later stricter tests:

- some runs succeeded marginally
- some runs failed and moved the ball farther from center

Interpretation:

- the system can sometimes produce meaningful target approach
- but it is not yet repeatably reliable

### 5. Latest twist semantics check suggests a frame mismatch / 最新 twist 语义排查提示存在坐标系错位

Evidence:

- after fixing a test-script default-value issue, the latest isolated twist tests were still strongly inconsistent with the commanded axes
- commanded `+x 10 mm` produced about `(-0.39, -1.86, -1.12) mm`
- commanded `-y 10 mm` produced about `(+1.97, +2.71, -0.68) mm`
- commanded `-z 5 mm` produced about `(-0.56, +0.31, +1.61) mm`
- `picknik_twist_controller` simply forwards the 6 `Twist` numbers to the hardware interfaces
- the `ros2_kortex` hardware interface sets `TwistCommand.reference_frame = CARTESIAN_REFERENCE_FRAME_TOOL`

Interpretation:

- the current local project logic treats Cartesian deltas as if they were base-frame motions
- the actual Kinova twist driver interprets them in the tool frame
- this frame-semantic mismatch is a strong candidate explanation for the observed axis coupling and wrong-sign motion
- even beyond the frame mismatch, the measured motion magnitude remains too small for reliable closed-loop grasp approach

## Current Configuration Snapshot / 当前配置快照

From [default_config.json](/home/pyq/Paper/configs/default_config.json):

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
  "max_translation_step_m": 0.015,
  "joint_state_topic": "/joint_states",
  "twist_command_topic": "/twist_controller/commands",
  "base_frame": "base_link",
  "ee_frame": "end_effector_link",
  "twist_command_duration_s": 0.8,
  "twist_publish_rate_hz": 20.0,
  "twist_stop_duration_s": 0.6,
  "workspace_xyz_min": [0.2, -0.55, 0.0],
  "workspace_xyz_max": [0.8, 0.4, 0.6]
}
```

```json
"policy": {
  "mode": "remote_api",
  "remote_url": "http://127.0.0.1:8000/predict"
}
```

## Main Current Bottlenecks / 当前主要瓶颈

### 1. Twist control is too weak / Twist 控制实效太弱

Evidence:

- even manually issued small twist tests produce almost zero measured end-effector displacement
- latest isolated axis tests also show sign inconsistency and strong cross-axis motion

Interpretation:

- the current real bottleneck is now low-level execution strength
- not just OpenVLA output quality

### 1.5. Twist frame semantics are likely mismatched / Twist 坐标语义很可能存在错位

Evidence:

- local project code publishes `geometry_msgs/Twist` from desired Cartesian deltas without any explicit base-to-tool conversion
- the Kinova ROS2 driver hard-codes twist commands to the tool reference frame
- local state monitoring currently uses TF from `base_link` to `end_effector_link`, while the hardware twist interface is exported as `tcp`

Interpretation:

- command-space semantics and observation-space semantics are probably not aligned
- fixing only high-level policy or refinement logic will not solve this class of error

### 2. `z` refinement is still unreliable / `z` 方向 refine 仍不可靠

Evidence:

- after correcting frame usage, refine no longer flies to absurdly high targets
- but raw base `z` still becomes unrealistic and needs clamping

Interpretation:

- `x/y` geometry is closer to usable
- `z` should still be treated as guarded or secondary

### 3. Safety limits are still strongly shaping behavior / 安全边界仍强烈塑造行为

Evidence:

- useful negative `y` motion was previously clipped to `0`
- behavior only improved after relaxing workspace `y` lower bound

Interpretation:

- workspace tuning still materially changes whether the robot can move toward the target

## Recommended Next Step / 推荐下一步

**English**

The next highest-value step is not full grasping.
It is:

**resolve the Kinova twist command semantics before continuing higher-level grasp experiments**

Reason:

- the project already has enough evidence that OpenVLA is alive
- the current main blocker is low-level execution strength and likely tool-frame/base-frame semantic mismatch
- without fixing that, more high-level logic will not change the outcome much
- the next concrete check should be whether local deltas are converted into the tool frame before publishing, or whether a more reliable Cartesian control path should replace the current twist path

**中文**

当前最有价值的下一步不是直接上完整抓取，而是：

**先解决 Kinova twist 命令的坐标语义问题，再继续更高层的抓取实验**

原因：

- 目前已经有足够证据说明 OpenVLA 在起作用
- 当前主要瓶颈已经转移到底层执行强度，以及 tool frame / base frame 语义可能不一致
- 如果这一步不解决，再继续堆上层逻辑，收益会很有限
- 下一步更具体的检查应当是：确认本地期望的 base 坐标增量是否先转换成了 tool 坐标速度；如果没有，就应优先修正这层，或直接更换更可靠的笛卡尔控制路径

## Related Documents / 相关文档

- [REAL_ROBOT_RUNBOOK.md](/home/pyq/Paper/REAL_ROBOT_RUNBOOK.md)
- [HANDOFF.md](/home/pyq/Paper/HANDOFF.md)
- [HANDOFF_NEXT_STEPS.md](/home/pyq/Paper/HANDOFF_NEXT_STEPS.md)
- [README_zh.md](/home/pyq/Paper/README_zh.md)
