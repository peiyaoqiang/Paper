# HANDOFF NEXT STEPS / 后续协作与推进文档

## 1. Purpose / 文档目的

**English**

This document is a collaboration-oriented extension of `HANDOFF.md`.
It is intended to help continue the project smoothly across:

- another computer
- another chat thread
- another collaborator
- a later stage of the same research cycle

It focuses on:

- the next two weeks of work
- technical priorities
- experiment priorities
- risks and mitigation
- expected deliverables

**中文**

本文档是 `HANDOFF.md` 的协作推进版，目的是帮助项目在以下场景中顺利延续：

- 切换到另一台电脑
- 切换到新的对话线程
- 与其他合作者协作
- 在后续研究阶段继续推进

重点内容包括：

- 未来两周的任务安排
- 技术优先级
- 实验优先级
- 风险与应对
- 预期交付物

## 2. Current Research Direction / 当前研究方向

**English**

Current project direction:

**OpenVLA-guided eye-in-hand grasping with RGB-D geometric refinement and closed-loop execution**

The intended contribution is not full-scale OpenVLA reproduction.
Instead, the paper should focus on:

- adapting OpenVLA to eye-in-hand observations
- mapping VLA outputs into a safe Kinova action space
- using RGB-D geometry for local correction
- improving grasp robustness with closed-loop execution

**中文**

当前项目方向为：

**基于 OpenVLA 引导、RGB-D 几何修正与闭环执行的眼在手上抓取**

本项目的目标不是完整复现大规模 OpenVLA 训练流程。
更适合论文表达的贡献点是：

- 将 OpenVLA 适配到眼在手上观测场景
- 将 VLA 输出映射到安全的 Kinova 动作空间
- 利用 RGB-D 几何做局部修正
- 通过闭环执行提升抓取鲁棒性

## 3. Collaboration Baseline / 协作基线

**English**

Anyone continuing this project should first read:

1. `README.md`
2. `README_zh.md`
3. `HANDOFF.md`
4. `HANDOFF_NEXT_STEPS.md`

Then run:

```bash
python experiments/run_demo.py
```

If the demo runs successfully, the collaborator can move on to real hardware integration.

**中文**

任何接手该项目的人，建议首先阅读：

1. `README.md`
2. `README_zh.md`
3. `HANDOFF.md`
4. `HANDOFF_NEXT_STEPS.md`

然后运行：

```bash
python experiments/run_demo.py
```

如果 demo 能正常跑通，就可以继续推进真实硬件接入工作。

## 4. Next Two Weeks Plan / 未来两周计划

### Week 1 / 第 1 周

**English**

Primary goal:

**replace the mock hardware path with a real sensing and motion path**

Tasks:

1. integrate `Intel RealSense D435i`
2. integrate `Kinova Gen3` Cartesian control
3. integrate real gripper open/close control
4. validate hand-eye transform consistency
5. save raw observation and action logs

Expected outcome:

- the robot can observe, move, and close the gripper under software control
- image/depth frames and robot states can be recorded
- image-to-robot projection works approximately correctly

**中文**

核心目标：

**把 mock 硬件链路替换成真实感知与运动链路**

任务：

1. 接入 `Intel RealSense D435i`
2. 接入 `Kinova Gen3` 笛卡尔控制
3. 接入真实夹爪开合控制
4. 验证手眼坐标变换一致性
5. 保存原始观测和动作日志

预期结果：

- 机器人可以在软件控制下完成观察、移动和夹爪开合
- 可以记录图像、深度和机器人状态
- 图像到机器人坐标的投影基本正确

### Week 2 / 第 2 周

**English**

Primary goal:

**run the first real grasping loop and collect data for a pilot experiment**

Tasks:

1. connect real OpenVLA inference or keep a placeholder high-level policy if blocked
2. implement real local RGB-D grasp refinement
3. add a trial logger
4. run at least `30-50` pilot trials
5. label failures and summarize the dominant failure modes

Expected outcome:

- one complete real-robot grasping loop works
- a pilot dataset is collected
- major system failure points become visible

**中文**

核心目标：

**跑通第一版真实抓取闭环，并采集试验数据**

任务：

1. 接入真实 OpenVLA 推理，如果受阻可先保留高层策略占位版本
2. 实现真实的局部 RGB-D 抓取修正
3. 增加实验日志记录器
4. 至少完成 `30-50` 次试验
5. 标注失败类型并总结主要失败模式

预期结果：

- 第一版真实机器人抓取闭环可运行
- 形成一份初步试验数据集
- 系统的主要失败点开始变得清晰

## 5. Technical Priorities / 技术优先级

**English**

Priority order:

1. real sensing
2. real safe motion
3. hand-eye consistency
4. logging
5. depth-based refinement
6. actual OpenVLA inference

Important note:

Do not make OpenVLA integration the first blocker.
The system should first be able to:

- capture data
- move safely
- project correctly
- execute a closed-loop grasp skeleton

**中文**

技术优先级顺序：

1. 真实感知
2. 真实安全运动
3. 手眼一致性验证
4. 日志记录
5. 基于深度的抓取修正
6. 真实 OpenVLA 推理

重要说明：

不要把 OpenVLA 接入当成第一阻塞点。
系统应先具备以下能力：

- 能采集数据
- 能安全运动
- 能正确投影
- 能运行抓取闭环骨架

## 6. Experiment Priorities / 实验优先级

**English**

Recommended experiment order:

1. static target, no clutter
2. single target, light clutter
3. language-specified target among distractors
4. moderate clutter with partial occlusion

Do not begin with:

- transparent objects
- reflective objects
- highly deformable objects
- severe clutter

**中文**

建议实验顺序：

1. 静态单目标无遮挡
2. 单目标轻度杂乱
3. 语言指定目标、带干扰物
4. 中度杂乱与部分遮挡

不要一开始就做：

- 透明物体
- 强反光物体
- 高度可变形物体
- 重度杂乱场景

## 7. First Deliverables / 第一阶段交付物

**English**

The first stage should produce:

- real camera driver
- real robot driver
- hand-eye validation script or test routine
- pilot experiment logger
- `30-50` trial records
- a short failure analysis note

**中文**

第一阶段应形成以下交付物：

- 真实相机驱动
- 真实机器人驱动
- 手眼验证脚本或测试流程
- 试验日志记录器
- `30-50` 条试验记录
- 一份简短失败分析说明

## 8. Suggested Task Ownership / 建议任务分工

**English**

If multiple collaborators work in parallel, use the following split:

- Person A: camera and logging
- Person B: Kinova and gripper control
- Person C: OpenVLA inference wrapper and action adapter
- Person D: depth refinement and analysis scripts

If only one person is working, follow this order:

1. camera
2. robot
3. calibration check
4. logger
5. refinement
6. OpenVLA integration

**中文**

如果多人并行协作，建议按以下方式分工：

- A：相机与日志系统
- B：Kinova 与夹爪控制
- C：OpenVLA 推理封装与动作适配
- D：深度修正与分析脚本

如果只有一个人推进，建议顺序为：

1. 相机
2. 机器人
3. 标定验证
4. 日志系统
5. 修正模块
6. OpenVLA 接入

## 9. Major Risks / 主要风险

**English**

Risk 1:
The hand-eye transform may be numerically valid but practically inaccurate for grasping.

Mitigation:
Use simple projection tests before any real grasp experiments.

Risk 2:
Depth quality from `D435i` may be unstable on dark, reflective, or thin objects.

Mitigation:
Start with matte, rigid, medium-sized objects.

Risk 3:
Direct OpenVLA outputs may not match Kinova control semantics.

Mitigation:
Keep the action adapter and safety clipping layer mandatory.

Risk 4:
The project may over-focus on model integration and under-invest in data/logging.

Mitigation:
Record all trials from the first day.

**中文**

风险 1：
手眼标定在数值上看似正确，但对真实抓取仍可能不够准。

应对：
在正式抓取前先做简单投影验证。

风险 2：
`D435i` 在深色、反光或细长物体上的深度质量可能不稳定。

应对：
先从亚光、刚性、中等尺寸物体开始。

风险 3：
OpenVLA 直接输出的动作语义可能与 Kinova 控制接口不匹配。

应对：
保留动作适配器与安全裁剪层，不要直接下发原始动作。

风险 4：
项目可能过度关注模型接入，而忽略数据和日志记录。

应对：
从第一天起记录所有试验。

## 10. Data Logging Recommendation / 数据记录建议

**English**

For every trial, record:

- trial id
- timestamp
- language instruction
- RGB frame path
- depth frame path
- robot end-effector pose
- raw policy action
- adapted safe action
- refined grasp pose
- success or failure
- failure type

Suggested failure types:

- target missed
- grasped wrong object
- collision
- slip after lift
- depth failure
- localization error

**中文**

每次试验建议记录：

- trial id
- 时间戳
- 语言指令
- RGB 图像路径
- depth 图像路径
- 末端位姿
- 原始策略动作
- 安全适配后的动作
- 修正后的抓取位姿
- 成功或失败
- 失败类型

建议的失败标签包括：

- 抓空
- 抓错物体
- 碰撞
- 抬升后滑落
- 深度失败
- 定位误差

## 11. Paper-Oriented Milestones / 面向论文的阶段里程碑

**English**

Milestone 1:
The real system can complete at least one successful grasp in a simple scene.

Milestone 2:
The system can run repeated trials with logging.

Milestone 3:
Three comparison settings are available:

- direct policy execution
- policy plus refinement
- policy plus refinement plus closed-loop execution

Milestone 4:
A pilot table and one success-rate figure are generated.

**中文**

里程碑 1：
真实系统能在简单场景中至少成功完成一次抓取。

里程碑 2：
系统能进行重复试验并保存日志。

里程碑 3：
能完成三组对比：

- 直接策略执行
- 策略加几何修正
- 策略加几何修正再加闭环执行

里程碑 4：
能产出一张初步结果表和一张成功率图。

## 12. Immediate Action / 立刻要做的事

**English**

If resuming the project from another machine or another session, start here:

1. open the project root
2. read `HANDOFF.md`
3. read `HANDOFF_NEXT_STEPS.md`
4. run `python experiments/run_demo.py`
5. begin integrating `drivers/realsense_driver.py`

**中文**

如果是在另一台电脑或新的对话线程中继续推进，请从这里开始：

1. 打开项目根目录
2. 阅读 `HANDOFF.md`
3. 阅读 `HANDOFF_NEXT_STEPS.md`
4. 运行 `python experiments/run_demo.py`
5. 开始接入 `drivers/realsense_driver.py`
