# SpaceMouse 控制 Kinova 与采集数据记录

这份文档记录本次围绕 3Dconnexion SpaceMouse 控制 Kinova Gen3 机械臂，以及使用 SpaceMouse 采集训练数据所做的工作。

## 目标

使用 SpaceMouse 直接控制 Kinova 机械臂末端完成抓取任务，并复用同一套控制逻辑采集训练数据。

最终控制目标如下：

- SpaceMouse 平移控制机械臂末端 `x/y/z`。
- SpaceMouse 旋转控制机械臂末端 `roll/pitch/yaw`。
- SpaceMouse button0 用于夹爪开/合切换。
- SpaceMouse button1 在遥操作中用于停止退出，在采集中用于 episode 控制。
- 运动时不需要一直按住 button0。

## 主要文件

- `src/kinova_vla_collect/spacemouse_controller.py`
  - 封装 `pyspacemouse`。
  - 负责 SpaceMouse 连接、校准、死区、轴缩放、按钮状态和夹爪目标切换。
  - 新增 `reset_gripper_target()`，用于每个采集 episode 开始时重置夹爪状态。

- `src/kinova_vla_collect/teleop_spacemouse_real.py`
  - 真实 Kinova 机械臂 SpaceMouse 遥操作代码。
  - 高频发布 ROS 2 `geometry_msgs/Twist` 到 `/twist_controller/commands`。
  - 内部复用 Xbox 控制代码的 action 形式：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

  - 其中可复用到采集代码的函数包括：
    - `_spacemouse_to_xbox_action()`
    - `_decouple_spacemouse_groups()`
    - `_action_to_twist()`
    - `DirectTwistPublisher`

- `src/kinova_vla_collect/teleop_spacemouse_collect.py`
  - SpaceMouse 采集训练数据代码。
  - 使用 SpaceMouse 实时控制机械臂，同时记录图像、状态和动作。
  - 使用 `EpisodeRecorder` 保存 intermediate 数据格式。

- `src/kinova_vla_collect/inspect_intermediate_dataset.py`
  - 检查采集出来的 intermediate `.npz` 数据。
  - 检查 key 是否完整、shape 是否正确、时间戳是否递增、图像文件是否存在、是否有 NaN/Inf、夹爪值是否合法、动作是否全 0 等。

- `scripts/teleop_spacemouse_real.py`
  - 真实机械臂 SpaceMouse 遥操作启动脚本。

- `scripts/collect_spacemouse.py`
  - SpaceMouse 采集启动脚本。

- `scripts/inspect_intermediate_dataset.py`
  - 采集数据检查启动脚本。

## SpaceMouse 设备权限

如果 SpaceMouse 能被识别，但是启动时报错：

```text
Failed to open device
```

通常是 Linux HID 设备权限不够。

临时处理命令：

```bash
sudo chmod a+rw /dev/hidraw1 /dev/hidraw2 /dev/hidraw3 /dev/hidraw4 /dev/hidraw5 /dev/hidraw9
```

这个方法只是临时有效，拔插设备或重启后可能需要重新执行。长期使用建议添加 udev 规则。

单独测试 SpaceMouse 是否能打开：

```bash
python3 -c "import pyspacemouse; d=pyspacemouse.open(nonblocking=True); print(d.describe_connection()); print(d.read()); d.close()"
```

## ROS 2 环境

运行真实机械臂遥操作或采集之前，先执行：

```bash
source /opt/ros/humble/setup.bash
source /home/pyq/code/ros2_kortex_ws/install/setup.bash
cd /home/pyq/Paper
```

当前可用的控制方式是向下面这个 topic 发布 `geometry_msgs/Twist`：

```text
/twist_controller/commands
```

之前尝试过通过 `ros2_control` 切换到 twist command interface，但失败了。报错里显示硬件没有暴露这些 command interface：

```text
tcp/twist.linear.x
tcp/twist.linear.y
tcp/twist.linear.z
tcp/twist.angular.x
tcp/twist.angular.y
tcp/twist.angular.z
```

所以最终可用方案是：不切换底层 command interface，而是直接向已有 twist controller 的 topic 发布 `Twist` 消息。

## 真实机械臂遥操作

启动命令：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/teleop_spacemouse_real.py
```

当前已经调好的默认参数：

```text
backend=direct_ros_twist
hz=100
deadzone=0.03
max_linear_speed=0.07 m/s
max_angular_speed=10.0 rad/s
linear_scale=1.2
angular_scale=16.0
control_layout=normal
sign_x=+1
sign_y=+1
sign_z=-1
sign_roll=-1
sign_pitch=+1
sign_yaw=+1
require_enable_button=false
```

操作说明：

- 前后推动 SpaceMouse：控制末端前后运动。
- 左右推动 SpaceMouse：控制末端左右运动。
- 上下提/压 SpaceMouse：控制末端上下运动。
- 倾斜或旋转 SpaceMouse：控制末端 `roll/pitch/yaw`。
- button0：夹爪闭合/开启切换。
- button1：停止并退出。
- Ctrl+C：停止并退出。

单独测试 XYZ：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/teleop_spacemouse_real.py \
  --translation-only
```

单独测试 roll/pitch/yaw：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/teleop_spacemouse_real.py \
  --rotation-only
```

查看更详细的 raw action 和 filtered action：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/teleop_spacemouse_real.py \
  --debug
```

## 之前控制不正常的原因

调试过程中主要发现了这些问题：

- SpaceMouse 原始轴和 Xbox 手柄代码中的动作含义不一致。
  - 因此新增了 `_spacemouse_to_xbox_action()`，把 SpaceMouse 映射到 Xbox 相同的 action 形式。

- SpaceMouse 很容易同时产生平移和旋转输入。
  - 比如你想前后推，但实际 raw 数据里可能也有 pitch/roll。
  - 因此新增了 `_decouple_spacemouse_groups()`，默认保留更强的一组输入，减少平移和旋转串扰。

- 部分轴方向和直觉相反。
  - 后续逐步调过 `sign_x/sign_y/sign_z/sign_roll/sign_pitch/sign_yaw`。
  - 当前最终默认值已经是实机测试后比较顺手的一组。

- 低频发布和频繁读取状态会让机械臂反应慢。
  - 所以真实遥操作改为 100 Hz 发布 twist。
  - 状态读取频率降低，避免影响控制响应。

- 通过 fake controller 或 joint position 做仿真时，RViz 里看起来会动，但不等于真实的末端遥操作手感。
  - 最后实机使用 direct ROS twist topic 方式。

## SpaceMouse 采集训练数据

启动命令：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/collect_spacemouse.py \
  --config /home/pyq/Paper/kinova_vla_collect/configs/collect_pick_red_block.yaml
```

采集时按键逻辑：

- SpaceMouse button0：夹爪开/合切换。
- SpaceMouse button1 短按：
  - 当前没有录制时，开始新的 episode。
  - 当前正在录制时，保存当前 episode 为成功。
- SpaceMouse button1 长按：停止并退出采集程序。
- 键盘备用按键：
  - `n`：保存当前 episode 为失败。
  - `d`：丢弃当前 episode。
  - `q`：退出程序。
  - `h`：显示帮助。

采集代码设计：

- 遥操作控制频率默认是 `--teleop-hz 100`。
- 数据记录频率使用配置文件里的 `config.control.hz`，当前是 5 Hz。
- 记录的 action 形式是：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

- `dx/dy/dz` 根据当前 twist 和记录周期换算得到，并限制在 `config.control.max_delta_m` 范围内。
- `droll/dpitch/dyaw` 根据当前 twist 和记录周期换算得到，并限制在 `config.control.max_delta_rad` 范围内。
- `gripper=-1.0` 表示夹爪打开。
- `gripper=+1.0` 表示夹爪闭合。

当前配置默认输出目录：

```text
/home/pyq/Paper/datasets/data/lerobot/pick_up_the_red_ball
```

每个任务目录下的重要文件：

```text
data/episode_000000.npz
images/episode_000000/*.jpg
meta/info.json
meta/summary.json
```

## 检查采集数据是否合理

检查全部 episode：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/inspect_intermediate_dataset.py \
  --dataset-root /home/pyq/Paper/datasets/data/lerobot \
  --task pick_up_the_red_ball
```

只检查指定 episode：

```bash
PYTHONPATH=/home/pyq/Paper/kinova_vla_collect/src:$PYTHONPATH \
python3 /home/pyq/Paper/kinova_vla_collect/scripts/inspect_intermediate_dataset.py \
  --dataset-root /home/pyq/Paper/datasets/data/lerobot \
  --task pick_up_the_red_ball \
  --episodes 0 1
```

需要重点看这些指标：

- `errors=0 warnings=0`
  - 说明数据格式、shape、图像路径、时间戳基本正常。

- `dt`
  - 应接近 `1 / control.hz`。
  - 当前 5 Hz 采集时，正常值约为 `0.2s`。

- `zero_motion`
  - 表示动作接近 0 的比例。
  - 少量静止是正常的。
  - 如果高于 `0.8`，通常说明采集里太多时间没有动。

- `dx/dy/dz/droll/dpitch/dyaw`
  - 不能全部接近 0。
  - 如果经常达到最大裁剪值，说明采集时速度可能偏快，或者 `max_delta_m/max_delta_rad` 偏小。

- `gripper(open/close)`
  - 应该和任务动作一致。
  - 例如抓取任务里应该能看到夹爪从 open 到 close 的变化。

本次检查前两个 SpaceMouse episode 的结果：

```text
episode_000000: steps=100 zero_motion=0.320 dt=0.2037 gripper(open/close)=0.76/0.24 errors=0 warnings=0
episode_000001: steps=163 zero_motion=0.399 dt=0.2044 gripper(open/close)=0.82/0.18 errors=0 warnings=0
```

检查报告保存位置：

```text
/home/pyq/Paper/datasets/data/lerobot/pick_up_the_red_ball/inspection/intermediate_report.json
```

## 常见问题

### SpaceMouse 能找到但打不开

通常是 HID 权限问题。重新执行临时 `chmod` 命令，或者添加 udev 规则。

### 夹爪能动，但机械臂不动

检查：

- ROS 2 环境是否已经 source。
- `/twist_controller/commands` 是否存在。
- twist controller 是否 active。
- 机械臂是否处于可接收 twist 命令的模式。
- 是否误加了 `--translation-only` 或 `--rotation-only`。

### 机械臂反应慢

检查：

- `--hz` 或 `--teleop-hz` 是否接近 100。
- `--max-linear-speed` 是否限制了 XYZ 最大速度。
- `--max-angular-speed` 是否限制了姿态最大速度。
- `--linear-scale` 和 `--angular-scale` 是否过小。

### 机械臂运动太快

降低 XYZ 速度：

```bash
--max-linear-speed
--linear-scale
```

降低姿态速度：

```bash
--max-angular-speed
--angular-scale
```

### 某个轴方向反了

可以通过 sign 参数调整：

```bash
--sign-x -1
--sign-y -1
--sign-z -1
--sign-roll -1
--sign-pitch -1
--sign-yaw -1
```

当前代码默认值已经是实机测试后确定的一组。

## 已完成验证

新增 Python 文件已做过语法检查：

```bash
python3 -m py_compile \
  kinova_vla_collect/src/kinova_vla_collect/teleop_spacemouse_collect.py \
  kinova_vla_collect/scripts/collect_spacemouse.py \
  kinova_vla_collect/src/kinova_vla_collect/inspect_intermediate_dataset.py \
  kinova_vla_collect/scripts/inspect_intermediate_dataset.py
```

数据检查脚本已在下面目录上运行：

```text
/home/pyq/Paper/datasets/data/lerobot/pick_up_the_red_ball
```

检查结果：

```text
Inspection complete: errors=0 warnings=0
```
