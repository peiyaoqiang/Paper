# Kinova OpenPI0 科研项目诊断

## 结论

当前项目最适合表述为：**面向 Kinova 眼在手真实机器人操作平台的 VLA 策略适配与安全部署研究**。它不应被表述为“提出新的 VLA 模型”，也不应被表述为“已经完整复现 pi0/OpenPI0 并证明性能提升”。

最稳妥的研究主题是：

> 在 Kinova Gen3 眼在手操作平台上，对 pi0/OpenPI0 风格的 VLA 策略进行安全约束部署，重点研究 observation/action contract 对齐、动作空间适配、夹爪语义、远程推理和真实闭环 rollout 诊断。

当前证据足够支撑一个系统诊断型、demo 型或 workshop 型方向。它还**不足以支撑** ICRA/IROS/CoRL/RA-L 正文中常见的强结果结论，例如“我们的方法提升了抓取成功率”，因为仓库中有 step 级部署日志，但没有 trial 级 success/failure 结果表。

## 已阅读证据

本地证据：

- 根目录文档：`README.md`、`HANDOFF.md`、`HANDOFF_NEXT_STEPS.md`、`PROGRESS_ARCHIVE_2026-04-06.md`、`REAL_ROBOT_RUNBOOK.md`。
- 论文草稿：`paper/abstract_draft.md`、`paper/outline.md`。
- OpenPI0/pi0 部署：`kinova_vla_collect/DEPLOY_OPENPI0_POLICY.md`、`kinova_vla_collect/src/kinova_vla_collect/deploy_openpi0_policy.py`、`kinova_vla_collect/outputs/deploy_runs` 下的部署日志。
- BC baseline 部署：`kinova_vla_collect/DEPLOY_BC_POLICY.md`、`kinova_vla_collect/src/kinova_vla_collect/deploy_bc_policy.py`。
- 数据采集与转换：`recorder.py`、`convert_to_lerobot.py`、任务配置文件、`PLACE_RED_BALL_ON_BLACK_X_WORKFLOW.md`。
- 图片/视频：仓库中有部署图像和 smoke-test 图像；没有找到本地视频文件。

外部背景来源：

- OpenPI 官方仓库说明 pi0 是 flow-based VLA，并提供 checkpoint 与 fine-tuning 示例：https://github.com/Physical-Intelligence/openpi
- OpenPI README 明确提醒，将 pi0 适配到不同机器人平台不一定直接可行。因此，把 Kinova 平台适配作为研究问题是合理的。
- pi0 论文提供模型家族背景：https://arxiv.org/abs/2410.24164
- OpenVLA 论文提供早期 OpenVLA 阶段的模型背景：https://arxiv.org/abs/2406.09246

这些外部来源只用于背景定位，不用于替代本项目自己的实验结果。

## 实验产物审计

`kinova_vla_collect/outputs/deploy_runs` 中的部署产物：

- 部署目录总数：163。
- step 级日志总数：13,739。
- 保存的部署图像总数：1,418。
- BC 部署：62 个 run 目录；55 个含 `steps.jsonl`；1,146 条 step 记录。
- OpenPI0 部署：101 个 run 目录；100 个含 `steps.jsonl`；12,593 条 step 记录。
- step 日志中显式 success/failure/result 标签数量：0。

OpenPI0 run metadata：

- `pi0_kinova_red_ball_lora`：72 个 run。
- `pi0_kinova_red_ball_lora_v1_to_v2`：29 个 run。
- dataset id 包括 `kinova_red_ball` 和 `kinova_red_ball_v2`。
- prompt 包括红球抓取、绿球抓取、英文变体、中文变体和“红球放到黑色 X”变体。

安全适配信号：

- OpenPI0 raw action 与 safe action 在 12,593/12,593 条 step 中不同。
- BC raw action 与 safe action 在 1,067/1,146 条 step 中不同。
- OpenPI0 gripper command：11,391 open，1,202 close。
- BC gripper command：791 open，199 close，156 hold。
- OpenPI0 每 step elapsed latency：均值约 402 ms，中位数约 335 ms。
- BC 每 step elapsed latency：均值约 4,666 ms，中位数约 4,480 ms。

关键限制：这些是 step 级部署统计，不是任务成功率。

## 当前研究主题如何表述

推荐标题方向：

1. **面向 Kinova 眼在手操作的 pi0/OpenPI0 安全约束部署研究**
2. **将 VLA 策略适配到 Kinova 真实机器人：动作语义、安全裁剪与 rollout 诊断**
3. **从 OpenPI0 checkpoint 到 Kinova rollout：VLA 动作空间对齐的真实机器人研究**

最诚实的 abstract-level 表述：

> 本项目研究如何将 pi0/OpenPI0 风格的 VLA 策略适配到 Kinova Gen3 眼在手真实机器人平台。系统连接 wrist RGB observation、14-D robot state、远程 WebSocket policy inference、本地 7-D 安全约束动作执行和 step-level rollout logging。当前证据表明真实机器人部署链路已经打通，并且 action-space mismatch 与 safety clipping 是主导部署现象；任务级成功率提升仍需要通过带标签的重复评测来证明。

## 相比单纯复现 pi0/OpenPI0 的潜在贡献

### C1. 真实机器人平台适配层

已有代码和日志支持这一点。

项目已经把 OpenPI/OpenPI0 策略输出映射到：

- wrist RGB observation，
- 14-D robot state，
- 7-D delta end-effector action，
- CTAG/Modbus gripper command，
- ROS2 twist execution，
- workspace safety guard，
- JSONL rollout logging。

这已经超过“调用一次 pi0 inference”。它是一个 Kinova-specific deployment layer。

目前不能声称：

- 提升了成功率；
- 实现了鲁棒操作；
- 具备跨任务泛化能力。

### C2. action-space alignment 与 safety clipping 的部署诊断

已有日志支持这一点。

当前最强的经验观察是：raw policy action 在进入真实机器人前被大量转换和裁剪。OpenPI0 日志中，每一条 step 的 raw action 与 safe action 都不同。

可发展成论文贡献的方向：

> 系统分析 VLA raw action chunk 如何与 Kinova 小步长、安全约束控制空间相互作用。

要成为论文级贡献，还需要：

- 比较不同 clipping threshold；
- 比较不同 action scaling；
- 比较不同 gripper mode；
- 报告 success/failure 和 safety incident。

### C3. BC vs OpenPI0 的对比基础设施

部分支持。

仓库同时包含 BC deployment client 和 OpenPI0 deployment client。两者共享真实机器人栈和 step-level 日志。

缺失的是：

- matched evaluation task；
- 相同 initial-state distribution；
- trial-level outcome labels；
- confidence interval。

### C4. 真实机器人上的 prompt/language variation

部分支持。

OpenPI0 runs 中包含多种英文和中文 prompt。这个方向可以发展成 language-conditioned deployment sensitivity。

当前限制：

- prompt 变化只体现在 run config 中；
- 还没有 outcome metric 证明不同 prompt 对任务成功率或失败模式的影响。

### C5. Kinova 的 LeRobot/OpenPI 数据 contract

作为软件基础设施已有支持，但还不是结果贡献。

`recorder.py` 和 `convert_to_lerobot.py` 约束了：

- `observation.images.wrist`；
- 14-D `observation.state`；
- 7-D `action`；
- gripper target label 为 `-1` 或 `+1`；
- raw episode 转换时只保留 `success: true`。

缺失的是：

- 本地仓库中没有数据集 summary 文件；
- 没有 train/validation split 报告；
- 没有 training loss 或 validation rollout 报告。

## 当前结果是否足以支撑 AI/robotics 论文

结论：**还不足以支撑一篇 claims-driven 的 ICRA/IROS/CoRL/RA-L 正文论文**。

足够支撑：

- 项目报告；
- 系统 demo note；
- workshop-style submission，如果主题定位为 “lessons from adapting pi0/OpenPI0 to Kinova”；
- 内部 milestone report。

目前不够支撑：

- 声称抓取成功率提升；
- 声称系统鲁棒；
- 声称 OpenPI0 优于 BC；
- 声称语言泛化；
- 声称 efficient inference 是方法贡献。

主要瓶颈不是代码少，而是**缺少带标签的 trial-level evaluation evidence**。

## 缺少哪些关键实验

P0：主会论文前必须补的实验：

- Trial-level evaluation table：
  - method；
  - task；
  - prompt；
  - scene variation；
  - checkpoint；
  - initial condition；
  - success/failure；
  - failure mode；
  - deployment log directory；
  - video/image evidence。
- Matched baselines：
  - teleoperation/reference trajectory，如适用；
  - BC baseline；
  - OpenPI0 direct execution；
  - OpenPI0 + safety clipping；
  - OpenPI0 + proposed refinement/execution strategy。
- 重复试验：
  - pilot claim 至少每个主条件 20-30 次；
  - 如果比较差异较小，需要更多重复。
- 失败类型：
  - miss object；
  - wrong object；
  - no close；
  - premature close；
  - slip after lift；
  - collision/unsafe motion；
  - localization 或 action-frame error；
  - black-X 任务中的 place offset failure。

P1：建议消融：

- `max_delta_m`：例如 0.003、0.005、0.008。
- gripper mode：passthrough、close-only、open/close thresholded。
- observation state：real state vs zero state。
- prompt language：相同初始场景下比较英文和中文。
- checkpoint：`pi0_kinova_red_ball_lora` vs `pi0_kinova_red_ball_lora_v1_to_v2`。
- dataset version：v1 vs v2 visual distractor dataset。
- camera preprocessing：RGB/BGR、resize/crop/pad、wrist-only vs 多相机。

P2：如果冲更强 venue：

- distractor 和 unseen object position 泛化；
- 红球放到黑色 X 的 object transport 与 release；
- 光照、遮挡、distractor 数量、初始位姿扰动；
- latency budget 与控制频率影响；
- learned/calibrated action adapter vs hard clipping。

## 适合投稿的方向

最适合：

- **robot manipulation / real-robot systems**：本地证据最强。
- **VLA adaptation**：适合，但重点应放在平台适配和 action contract，而不是新模型。
- **embodied AI systems**：可行，但需要任务级结果。
- **robustness**：只有补充扰动实验后才适合。
- **efficient inference**：目前较弱；有 latency log，但没有 isolated inference study。

当前不适合或证据不足：

- **sim-to-real**：没有仿真训练或 sim-to-real 证据。
- **new foundation model**：没有新模型证据。
- **state-of-the-art benchmark**：没有 benchmark protocol 和 success table。

## ICRA / IROS / CoRL / RA-L 风格应突出什么

### ICRA / IROS

最佳 framing：

> 一个真实机器人系统论文：如何把 VLA policy output 适配到安全的 Kinova eye-in-hand manipulation。

应突出：

- hardware stack；
- action semantics；
- safety constraints；
- real rollout logging；
- failure modes；
- VLA policy 部署到真实机器人时的实践经验。

必须补：

- 清楚的 success-rate table；
- videos；
- ablations；
- safety discussion。

### CoRL

最佳 framing：

> 针对非原生机器人平台，研究 pi0/OpenPI0 的 action-space alignment 和 deployment distribution mismatch。

应突出：

- VLA policy adaptation；
- dataset/action contract；
- checkpoint/dataset variants；
- 能解释失败的诊断证据。

必须补：

- 更强 baseline；
- repeated trials；
- quantitative learning/adaptation insight；
- 减少纯工程叙事。

### RA-L

最佳 framing：

> 一篇短而集中的真实机器人 letter：提出一个具体部署干预，并用 before/after 结果证明它有效。

必须补：

- 一个尖锐的 technical intervention；
- 强 before/after 证据；
- 紧凑但严格的评测。

当前状态还不适合 RA-L，因为主要正结果还没有 trial label 支撑。

## 已有证据支持的结论

- 项目包含真实 Kinova/OpenPI0 部署客户端，能够把 wrist RGB、14-D robot state 和 prompt 发送给 WebSocket policy server。
- 本地客户端能够解析 OpenPI action chunk，并把前 7 维作为 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` 执行。
- 部署栈会记录 raw action、safe action、robot state、gripper command、elapsed time 和保存的 wrist image。
- 仓库中存在 BC 和 OpenPI0 两类真实机器人部署日志。
- safety clipping 是 BC 和 OpenPI0 日志中都很明显的主导现象。
- 早期 OpenVLA 阶段已经记录了真实机器人执行链打通，以及 Kinova twist frame/control 的关键问题。
- 当前论文草稿还没有完成实验结果；摘要中使用的是 “expected to show”。

## 需要补充证据的结论

- OpenPI0 的抓取成功率优于 BC。
- 任一方法优于 direct pi0/OpenPI0 execution。
- 系统在 cluttered tabletop environment 中鲁棒。
- 中文 prompt 和英文 prompt 表现相当。
- v2 visual-distractor fine-tuning 提升鲁棒性。
- place-red-ball-on-black-X 任务已经解决。
- 远程 A100 inference 已经足够高效到能支持最终控制 claim。
- gripper policy 最终可靠。
- safety adapter 不只是防止危险动作，而且提升任务成功率。

## 推荐论文问题

最干净的问题表述是：

> 如何把 pi0/OpenPI0 VLA policy 从开源 checkpoint 和 Kinova-specific demonstration 适配成一个安全、可观察、可诊断的 Kinova 眼在手真实机器人操作系统？

这个表述避免过度声称。它让论文聚焦于仓库中真实可见的难点：

- observation key mismatch；
- action dimensionality and semantics；
- gripper label semantics；
- safety clipping；
- ROS2 twist frame semantics；
- remote inference latency；
- dataset/action contract consistency；
- real rollout failure diagnosis。

## 最小下一步

先创建 trial-level evaluation sheet，并给已有 OpenPI0 和 BC run directory 做人工标注。

最小字段：

```text
run_dir, method, checkpoint, dataset_version, prompt, task, scene_id,
max_delta_m, gripper_mode, steps, success, failure_mode, notes
```

然后做一个小规模 matched evaluation：

```text
Task: pick up the red ball
Methods: BC vs OpenPI0 v1 vs OpenPI0 v2
Trials: 20 per method
Metrics: success rate, close timing failure, slip/lift failure, mean rollout time
Evidence: run_dir + final image/video + manual label
```

只有完成这一步后，摘要里才适合把 “expected to show” 改成 “results show”。
