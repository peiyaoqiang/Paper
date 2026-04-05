# HANDOFF / 项目交接文档

## 1. Project Summary / 项目概述

**English**

This project is a paper-oriented prototype for real-world eye-in-hand robotic grasping with:

- `Kinova Gen3` robotic arm
- `Intel RealSense D435i` wrist-mounted camera
- parallel gripper
- `OpenVLA` as the high-level vision-language-action policy

The current research direction is:

**OpenVLA-guided eye-in-hand grasping with RGB-D geometric refinement and closed-loop execution**

The main idea is:

- `OpenVLA` provides high-level language-conditioned action suggestions
- `RGB-D geometry` refines the grasp pose locally
- `closed-loop execution` improves real-robot robustness

**中文**

本项目是一个面向论文原型验证的真实机器人眼在手上抓取系统，硬件包括：

- `Kinova Gen3` 机械臂
- `Intel RealSense D435i` 腕部相机
- 并联夹爪
- `OpenVLA` 作为高层视觉-语言-动作策略

当前研究方向为：

**基于 OpenVLA 引导、RGB-D 几何修正与闭环执行的眼在手上抓取**

核心思想是：

- `OpenVLA` 负责语言条件下的高层动作建议
- `RGB-D` 几何模块负责局部抓取位姿修正
- `闭环执行` 负责提升真实抓取稳定性

## 2. Current Status / 当前进度

**English**

The repository is currently a minimal runnable scaffold.

Completed:

- project folder structure created
- minimal Python modules created
- mock drivers for camera, robot, and gripper
- mock `OpenVLA` wrapper implemented
- action adapter and safety clipping scaffold implemented
- depth-based grasp refinement scaffold implemented
- closed-loop grasp state machine implemented
- English and Chinese README files created
- initial bilingual paper abstract draft created

Not completed yet:

- real `D435i` driver integration
- real `Kinova Gen3` control integration
- real gripper integration
- actual `OpenVLA` inference integration
- real experiment logging
- quantitative evaluation scripts

**中文**

当前仓库是一个最小可运行工程骨架。

已完成：

- 已创建项目目录结构
- 已创建最小 Python 模块
- 已实现 mock 相机、机械臂、夹爪驱动
- 已实现 mock 版 `OpenVLA` 封装
- 已实现动作适配与安全裁剪骨架
- 已实现基于深度的抓取修正骨架
- 已实现闭环抓取状态机
- 已创建中英文 README
- 已创建中英文论文摘要初稿

尚未完成：

- 真实 `D435i` 驱动接入
- 真实 `Kinova Gen3` 控制接入
- 真实夹爪控制接入
- 真实 `OpenVLA` 推理接入
- 真实实验日志系统
- 定量评测与分析脚本

## 3. Important Files / 重要文件

**English**

- English README: `README.md`
- Chinese README: `README_zh.md`
- Project handoff: `HANDOFF.md`
- Main demo entry: `experiments/run_demo.py`
- Default config: `configs/default_config.json`
- OpenVLA wrapper: `policy/openvla_wrapper.py`
- Action adapter: `adapters/action_adapter.py`
- Transform manager: `calibration/tf_manager.py`
- Grasp refinement: `geometry/grasp_refiner.py`
- State machine: `executor/task_state_machine.py`
- Paper abstract draft: `paper/abstract_draft.md`

**中文**

- 英文说明文档：`README.md`
- 中文说明文档：`README_zh.md`
- 项目交接文档：`HANDOFF.md`
- 主 demo 入口：`experiments/run_demo.py`
- 默认配置：`configs/default_config.json`
- OpenVLA 封装：`policy/openvla_wrapper.py`
- 动作适配器：`adapters/action_adapter.py`
- 坐标变换管理：`calibration/tf_manager.py`
- 抓取修正模块：`geometry/grasp_refiner.py`
- 状态机：`executor/task_state_machine.py`
- 论文摘要草稿：`paper/abstract_draft.md`

## 4. Current System Flow / 当前系统流程

**English**

The current software flow is:

1. capture wrist-view RGB-D and robot state
2. send wrist RGB and language instruction to `OpenVLA`
3. adapt policy output into safe Cartesian robot action
4. execute a coarse approach
5. use depth and hand-eye calibration to refine the grasp
6. execute final approach, close gripper, and lift

**中文**

当前软件流程为：

1. 采集腕部视角 RGB-D 与机器人状态
2. 将腕部 RGB 和语言指令输入 `OpenVLA`
3. 将策略输出转换为安全的笛卡尔动作
4. 执行粗接近
5. 利用深度和手眼标定进行抓取位姿修正
6. 执行末端接近、闭合夹爪并抬升

## 5. How To Run / 运行方式

**English**

Run the current demo from the project root:

```bash
python experiments/run_demo.py
```

Expected behavior:

- the script runs with mock hardware
- the state machine finishes one grasp attempt
- terminal output prints success flag, trace, and refined grasp pose

**中文**

在项目根目录下运行当前 demo：

```bash
python experiments/run_demo.py
```

预期行为：

- 脚本使用 mock 硬件运行
- 状态机完成一次抓取尝试
- 终端输出成功标志、状态轨迹和修正后的抓取位姿

## 6. Recommended Next Step / 推荐的下一步

**English**

The highest-priority next step is:

**integrate the real `Intel RealSense D435i` into `drivers/realsense_driver.py`**

Recommended order:

1. integrate the real camera
2. integrate real Kinova Cartesian motion
3. validate hand-eye projection accuracy
4. add real experiment logging
5. replace mock `OpenVLA` with actual inference

**中文**

当前优先级最高的下一步是：

**先把真实 `Intel RealSense D435i` 接入 `drivers/realsense_driver.py`**

建议顺序：

1. 先接真实相机
2. 再接真实 Kinova 笛卡尔运动控制
3. 验证手眼坐标投影精度
4. 增加真实实验日志系统
5. 把 mock `OpenVLA` 替换为真实推理

## 7. Research Goal / 研究目标

**English**

The short-term paper goal is not to reproduce the full original `OpenVLA` training pipeline.

Instead, the intended contribution is:

- adapting `OpenVLA` to an eye-in-hand setup
- adding RGB-D geometric refinement
- improving real-world grasp reliability with closed-loop execution

Suggested paper title:

**OpenVLA-Guided Eye-in-Hand Grasping with RGB-D Geometric Refinement**

**中文**

短期论文目标不是复现完整原版 `OpenVLA` 训练流程。

更适合当前项目的贡献点是：

- 将 `OpenVLA` 适配到眼在手上场景
- 引入 RGB-D 几何修正
- 通过闭环执行提升真实抓取稳定性

建议论文题目：

**基于 OpenVLA 引导与 RGB-D 几何修正的眼在手上抓取**

## 8. Documentation Rule / 文档约定

**English**

From this point on, new project documentation should be written in both English and Chinese by default.

**中文**

从现在开始，项目中的新增说明文档默认使用中英文双语编写。
