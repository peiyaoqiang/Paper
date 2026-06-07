# Kinova + pi0/OpenPI0 科研方向筛选与论文方案

## 结论

最推荐的论文方向是：

> **非原生机器人平台上的 VLA action-space alignment 与 safety-constrained deployment：以 Kinova + pi0/OpenPI0 为例。**

理由很直接：当前仓库已经有 Kinova 真实机器人部署链路、OpenPI0 和 BC 两套客户端、A100 server inference 记录、13,739 条 step-level 日志，以及最清楚的现象：OpenPI0 的 12,593/12,593 条 step 中 raw action 与 safe action 不同。这个方向不需要先证明“大泛化”，而是把真实部署中已经暴露出来的 action semantics、safety clipping、gripper behavior、latency 和 failure mode 做成严谨机器人论文。

当前不能写成“我们提升了抓取成功率”。正确写法是：**当前证据支持系统可运行和 action-space mismatch 存在；任务成功率、鲁棒性和方法优越性需要新的 trial-level 实验验证。**

## 方向 1：VLA Action-Space Alignment 与安全约束部署

### 研究问题
pi0/OpenPI0 风格的 VLA policy 在迁移到 Kinova Gen3 眼在手平台时，raw action 与真实机器人安全控制空间之间的不匹配如何影响执行行为？能否通过明确的 action adapter 和 safety-constrained deployment protocol 提升真实 rollout 的可控性和任务成功率？

### 核心假设
- H1：未适配或弱适配的 raw VLA action 会频繁触发 safety clipping，并导致动作幅度、方向或 gripper 时序不稳定。
- H2：显式 action scaling、workspace guard、gripper semantic handling 和小步长闭环执行能降低 unsafe/invalid action，并可能提升任务成功率。
- H3：action-space alignment 的效果可以通过 raw-vs-safe action divergence、成功率、失败模式和安全事件共同评估。

### 方法思路
- 保持 pi0/OpenPI0 model 不变，把贡献放在 deployment adapter。
- 将 OpenPI0 action chunk 的前 7 维固定解释为 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`。
- 系统比较不同 adapter：
  - raw/direct preview 或极保守 direct；
  - hard clipping；
  - clipping + action scaling；
  - clipping + gripper threshold/mode；
  - clipping + state-aware workspace guard。

### 所需实验
- Matched real-robot trials：相同任务、相同物体分布、相同初始位姿范围。
- Baselines：BC、OpenPI0 v1、OpenPI0 v2。
- Ablations：`max_delta_m`、gripper mode、checkpoint、prompt、state mode。
- Metrics：success rate、failure mode、raw/safe divergence、safety clip ratio、close timing、rollout time、latency。

### 风险
- 如果所有方法成功率都低，论文要转向“deployment failure diagnosis”而不是“性能提升”。
- 如果 direct raw action 不安全，不能真实执行，只能做 no-motion preview 或严格小步限幅。
- 如果 gripper policy 一直 open，必须把失败归因写清楚，不能硬说方法有效。

### 投稿适配度
- ICRA/IROS：高。真实机器人系统与部署诊断很适配。
- CoRL：中高。需要强调 action-space mismatch 和 policy deployment distribution shift。
- RA-L：中。需要一个清晰 before/after 干预和强表格。
- CVPR workshop / embodied AI workshop：高。适合讲 VLA deployment lessons。

## 方向 2：远程 A100 推理延迟与闭环控制频率的权衡

### 研究问题
A100 server inference 通过 WebSocket/HTTP 服务接入 Kinova 闭环控制时，推理延迟、控制频率和 action chunk 使用方式如何影响真实机器人执行稳定性？

### 核心假设
- H1：OpenPI0 远程推理延迟低于 BC 远程 HTTP baseline 时，闭环控制更容易保持稳定。
- H2：较高控制频率不一定提升成功率；如果 action semantics 不稳定，频率升高可能放大错误。
- H3：action chunk reuse 与 `chunk_steps` 会影响 latency/behavior tradeoff。

### 方法思路
- 固定 policy 与任务，只改变 deployment timing。
- 比较 `hz = 2/3/5`、`chunk_steps = 1/2/5`、保存图像频率、网络条件。
- 将 latency 分为 observation capture、policy inference、action execution、logging。

### 所需实验
- OpenPI0 v1/v2 在同一任务上运行不同控制频率。
- 对照 BC baseline。
- 记录 loop time、policy timing、成功率、early stop、unsafe stop、轨迹稳定性。

### 风险
- 当前日志中 OpenPI0 mean elapsed 约 402 ms，BC 约 4,666 ms，但这不是完整控制性能结论。
- 如果 latency 不是主要瓶颈，方向会变成工程 profiling，论文价值下降。

### 投稿适配度
- ICRA/IROS：中。
- CoRL：中低，除非能连接到 policy chunking 和 closed-loop learning。
- RA-L：低到中，需要明确干预。
- CVPR workshop：中，适合 embodied AI system note。

## 方向 3：Kinova Demonstration 数据集与 OpenPI0 LoRA 适配

### 研究问题
对于 Kinova 眼在手操作，多少真实 demonstration、怎样的 gripper label 和 visual distractor 数据能让 OpenPI0 更稳定地执行目标抓取或放置任务？

### 核心假设
- H1：Kinova-specific demonstrations 比直接使用通用 checkpoint 更适合当前 action/state contract。
- H2：v2 visual distractor 数据集能改善干扰物场景表现。
- H3：gripper label 从 hold/ambiguous 改成 target-state `-1/+1` 会改善 close/open 时序。

### 方法思路
- 使用现有 LeRobot-style recorder 和 converter。
- 比较 dataset version：v1 red-ball、v2 visual distractor、place-on-black-X。
- 比较 checkpoint：`pi0_kinova_red_ball_lora` vs `pi0_kinova_red_ball_lora_v1_to_v2`。

### 所需实验
- 需要补齐本地 dataset summary：episode 数、frame 数、success-only split、gripper distribution。
- 需要 server 侧训练配置、loss curve、checkpoint step。
- 真实机器人评测：每个 checkpoint 20-30 trials。

### 风险
- 当前本地仓库没有完整训练日志和数据集 summary，论文证据链不完整。
- 如果只有少量 demonstrations，结果可能不稳定。

### 投稿适配度
- CoRL：中高，前提是训练与数据证据齐全。
- ICRA/IROS：中。
- RA-L：中，需要简洁数据 scaling 结论。
- CVPR workshop：高，适合 embodied dataset adaptation。

## 方向 4：语言 Prompt 与目标条件泛化

### 研究问题
OpenPI0 在 Kinova 真实机器人上是否对不同语言、不同 prompt phrasing、不同目标颜色或目标位置表现出一致的操作行为？

### 核心假设
- H1：同义英文 prompt 对行为影响较小。
- H2：中文 prompt 可能因训练分布不同导致行为变化。
- H3：目标颜色或目标位置变化会暴露 visual grounding 与 action policy 的耦合问题。

### 方法思路
- 固定场景与 checkpoint，比较 prompt。
- prompt 组：
  - `pick up the red ball`
  - `grab the red ball`
  - `pick the red ball`
  - `抓取红球`
  - `pick up the green ball`
  - `move to the black X`
- 评估行为是否朝向正确目标、是否 close、是否完成 task。

### 所需实验
- 每个 prompt 至少 10-20 trials。
- 需要同一物理场景的 video/image evidence。
- 需要人工标注目标选择是否正确。

### 风险
- 当前 run config 中有 prompt variation，但没有 outcome label。
- 如果 policy 主要由 demonstration/action prior 驱动，prompt 敏感性可能弱。

### 投稿适配度
- CVPR workshop / embodied AI workshop：高。
- CoRL：中，需要更强实验。
- ICRA/IROS：中低，除非结合真实任务成功率。
- RA-L：低到中。

## 方向 5：真实机器人 Failure Mode Taxonomy 与 VLA Deployment Benchmark

### 研究问题
VLA policy 部署到 Kinova 真实平台时，主要失败模式是什么？是否可以建立一个轻量 benchmark protocol 来诊断 action、gripper、latency、视觉和安全约束问题？

### 核心假设
- H1：失败主要来自 action-space mismatch、gripper timing、workspace clipping 和视觉/状态分布偏移，而不只是模型“不聪明”。
- H2：step-level logs + final images/videos + manual labels 可以形成可复用的 deployment diagnosis protocol。

### 方法思路
- 不把目标设为最高成功率，而是建立诊断基准。
- 每个 run 标注 failure mode。
- 输出 failure taxonomy 和示例图。

### 所需实验
- 给已有 BC/OpenPI0 run directory 补人工标签。
- 新跑 matched trials，覆盖简单抓取、干扰物、black-X 放置。
- 统计每种 failure mode 的比例和代表性日志。

### 风险
- 如果缺少视频，部分 failure mode 难以准确标注。
- 作为主会论文可能显得“方法贡献不足”，更适合 workshop 或系统论文的一部分。

### 投稿适配度
- CVPR workshop / embodied AI workshop：高。
- ICRA/IROS：中。
- CoRL：中低，除非和具体 adaptation 方法绑定。
- RA-L：低到中。

## 方向筛选

| 方向 | 可落地性 | 论文价值 | 当前证据起点 | 主要缺口 | 推荐级别 |
|---|---:|---:|---:|---|---|
| Action-space alignment 与安全约束部署 | 高 | 高 | 强 | trial success labels | 第一推荐 |
| A100 远程推理延迟与控制频率 | 高 | 中 | 中 | latency 分解与任务结果 | 可作为主线副实验 |
| Demonstration 数据集与 LoRA 适配 | 中 | 高 | 中 | server 侧训练证据 | 第二阶段方向 |
| Prompt/语言条件泛化 | 中 | 中 | 中 | outcome label | workshop 友好 |
| Failure taxonomy / benchmark | 高 | 中 | 中 | 视频和人工标注 | 可作为论文分析章节 |

最容易落地且最有论文价值的是方向 1。方向 2 和方向 5 应作为方向 1 的 supporting experiments。方向 3 可作为后续扩展；方向 4 适合做 workshop 或附加实验。

## 最推荐方向的完整实验方案

### 推荐题目
中文题目：

> 面向 Kinova 真实机器人的 pi0/OpenPI0 动作空间对齐与安全约束部署研究

英文题目候选：

1. **Action-Space Alignment for Safety-Constrained pi0/OpenPI0 Deployment on a Kinova Manipulator**
2. **From OpenPI0 Actions to Kinova Rollouts: Diagnosing and Adapting VLA Policies on a Real Eye-in-Hand Robot**
3. **Safety-Constrained VLA Policy Deployment on Non-Native Robot Platforms**

### 核心研究问题
在 Kinova Gen3 眼在手平台上，OpenPI0 raw action 与真实机器人控制空间之间的不匹配是否会系统性影响 rollout？通过显式 action adapter 和 safety-constrained execution，能否改善真实任务执行和失败模式？

### 实验对象
- Robot：Kinova Gen3。
- Camera：wrist-mounted RealSense D435i。
- Policy server：A100 server running OpenPI/OpenPI0 policy server。
- Local client：`deploy_openpi0_policy.py`。
- Baseline client：`deploy_bc_policy.py`。
- Task 1：pick up the red ball。
- Task 2：pick up the red ball with distractors。
- Task 3：pick up/place red ball on black X，只作为扩展任务，除非已经能稳定运行。

### 实验变量

主变量：

- Method：
  - BC baseline。
  - OpenPI0 v1：`pi0_kinova_red_ball_lora`。
  - OpenPI0 v2：`pi0_kinova_red_ball_lora_v1_to_v2`。
- Adapter：
  - Conservative clipping：`max_delta_m = 0.003`。
  - Medium clipping：`max_delta_m = 0.005`。
  - Larger clipping：`max_delta_m = 0.008`，只在安全验证后使用。
- Gripper mode：
  - passthrough。
  - close_only。
  - open_close thresholded。
- Prompt：
  - English canonical：`pick up the red ball`。
  - English paraphrase：`grab the red ball`。
  - Chinese：`抓取红球`。
- Scene：
  - single red ball。
  - red ball + distractor。
  - red ball + black X，扩展。

控制变量：

- 初始相机视角范围。
- 物体位置采样区域。
- 光照条件。
- 控制频率，例如固定 `hz = 3` 作为主实验。
- 每次 rollout 最大步数。
- 相同 gripper 初始状态。

### 对照组

最低可行对照：

| 组别 | 说明 | 目的 |
|---|---|---|
| BC | 已有 BC deployment client | 行为克隆 baseline |
| OpenPI0 v1 + conservative clipping | 当前主要可部署设置 | OpenPI0 baseline |
| OpenPI0 v2 + conservative clipping | v2 visual distractor checkpoint | checkpoint/data variant |
| OpenPI0 v2 + medium clipping | 更大 action budget | 测试 action limit 对任务的影响 |
| OpenPI0 v2 + close_only | 改变 gripper 执行逻辑 | 测试 gripper failure 是否是主瓶颈 |

不建议直接真实执行完全 unclipped raw action。可以用 no-motion preview 分析 raw action 分布，但不要作为真实机器人执行组。

### 评价指标

任务指标：

- Success rate：成功抓起目标并 lift 的比例。
- Target selection accuracy：是否朝正确目标执行。
- Grasp close timing：close 是否发生在接近目标之后。
- Lift success：close 后是否带起目标。
- Completion time / steps：完成任务步数或时间。

安全与部署指标：

- Clip ratio：raw action 与 safe action 不同的 step 比例。
- Clip magnitude：`||raw_xyz - safe_xyz||`。
- Workspace block rate：因 workspace guard 被置零的比例。
- Gripper command distribution：open/close/hold 比例。
- Emergency/abort count。
- Mean/median loop latency。

失败分析指标：

- miss object。
- wrong object。
- no close。
- premature close。
- close but no grasp。
- slip after lift。
- collision/unsafe approach。
- moves away from target。
- stuck due to clipping。

### 消融实验

Ablation 1：`max_delta_m`

- 固定 OpenPI0 v2、prompt、scene。
- 比较 0.003、0.005、0.008。
- 目标：回答 action budget 过小是否导致无法接近，过大是否导致不安全或 overshoot。

Ablation 2：gripper mode

- passthrough vs close_only vs open_close thresholded。
- 目标：回答失败是否主要来自 gripper policy。

Ablation 3：checkpoint/data version

- OpenPI0 v1 vs OpenPI0 v2。
- 目标：回答 visual distractor / v2 data 是否改善真实场景行为。

Ablation 4：prompt language

- English canonical vs English paraphrase vs Chinese。
- 目标：回答 prompt variation 是否影响目标选择和行为稳定性。

Ablation 5：control timing

- `hz = 2/3/5` 或 `chunk_steps = 1/2/5`。
- 目标：回答 A100 server inference latency 与控制频率是否影响成功率和稳定性。

### 最小实验量

建议先做 pilot：

- 3 个方法组：BC、OpenPI0 v1、OpenPI0 v2。
- 每组 10 trials。
- 只做 single red ball。
- 目标：验证 success label、failure taxonomy 和安全流程。

主实验：

- 5 个方法/消融组。
- 每组 20 trials。
- 共 100 trials。
- 每个 trial 保存 run_dir、最终图像或视频、人工 success/failure、failure mode。

如果时间有限，最低可投稿 workshop 版本：

- 3 个组，每组 20 trials，共 60 trials。
- 附加 step-level raw/safe action 统计。

### 失败分析流程

每个 trial 标注：

```text
run_dir
method
checkpoint
prompt
scene_id
max_delta_m
gripper_mode
success
failure_mode
first_failure_step
notes
final_image_or_video
```

每个 failure mode 至少保留 2-3 个可视化案例。论文中不要只报成功率，要报失败分布，因为当前项目真正有价值的是部署诊断。

## 论文结构设计

### 摘要框架

第 1 句：问题

> VLA policies such as pi0/OpenPI0 are increasingly accessible, but deploying them on non-native real robot platforms remains difficult because their action outputs, gripper semantics, and control assumptions may not match the target robot.

第 2 句：本文做什么

> We study this deployment gap on a Kinova Gen3 eye-in-hand manipulation platform with a wrist RealSense camera and an A100-hosted OpenPI0 policy server.

第 3 句：方法

> The system aligns wrist RGB observations, 14-D robot state, 7-D delta end-effector actions, safety clipping, workspace guards, and gripper modes into a logged closed-loop deployment stack.

第 4 句：证据类型

> We evaluate BC and OpenPI0 checkpoints under matched real-robot trials, reporting task success, action clipping, latency, gripper behavior, and failure modes.

第 5 句：结果占位

> 当前不能写具体数字；完成实验后再填入 “results show ...”。现在应写成计划或 `[RESULTS NEED EVIDENCE]`。

### Introduction 逻辑

1. VLA model 让机器人策略部署门槛降低，但真实机器人部署不是“加载 checkpoint”。
2. 非原生平台会出现 observation key、state dimension、action unit、gripper semantics、control frame 和 safety constraints 的错配。
3. Kinova eye-in-hand 是一个典型测试场景：wrist RGB、14-D state、7-D action、ROS2 twist、A100 remote inference。
4. 当前项目观察到 raw action 与 safe action 大量不同，说明 action-space mismatch 是实际瓶颈。
5. 本文研究问题：如何诊断并适配 VLA action，使其可安全部署到 Kinova 真实平台。
6. 贡献：
   - 一个 Kinova + OpenPI0 safety-constrained deployment stack。
   - 一个 action-space alignment 诊断协议。
   - 一个 matched real-robot evaluation，比较 BC、OpenPI0 v1/v2 和 adapter ablations。
   - 一个 failure mode taxonomy。

注意：第 6 点中的后两项需要实验完成后才能写成完成式。

### Method 结构

1. System Overview
   - Kinova Gen3、wrist RealSense、gripper、A100 policy server。
   - 数据流：image/state/prompt -> policy -> raw action -> adapter -> robot。
2. Observation and Action Contract
   - `observation.images.wrist`。
   - `observation.state` 14-D。
   - `[dx, dy, dz, droll, dpitch, dyaw, gripper]`。
3. Safety-Constrained Action Adapter
   - xyz clipping。
   - rpy clipping。
   - workspace guard。
   - gripper threshold/mode。
4. Closed-Loop Deployment
   - frequency、max steps、chunk steps、logging。
5. Diagnostics
   - raw/safe divergence。
   - gripper distribution。
   - latency。
   - failure taxonomy。

### Experiments 表格设计

表 1：实验条件表

| Condition | Policy | Checkpoint | Adapter | Gripper mode | Prompt | Scene | Trials |
|---|---|---|---|---|---|---|---:|
| BC | BC | best.pt | clipping | passthrough | red ball | single | 20 |
| OPI-v1 | OpenPI0 | v1 | clipping 0.003 | passthrough | red ball | single | 20 |
| OPI-v2 | OpenPI0 | v2 | clipping 0.003 | passthrough | red ball | single | 20 |
| OPI-v2-md | OpenPI0 | v2 | clipping 0.005 | passthrough | red ball | single | 20 |
| OPI-v2-close | OpenPI0 | v2 | clipping 0.003 | close_only | red ball | single | 20 |

表 2：主结果表

| Method | Success ↑ | Target correct ↑ | Close timing fail ↓ | Slip/lift fail ↓ | Abort/unsafe ↓ | Steps ↓ |
|---|---:|---:|---:|---:|---:|---:|
| BC | TBD | TBD | TBD | TBD | TBD | TBD |
| OpenPI0 v1 | TBD | TBD | TBD | TBD | TBD | TBD |
| OpenPI0 v2 | TBD | TBD | TBD | TBD | TBD | TBD |

表 3：deployment diagnostics

| Method | Clip ratio ↓ | Mean clip magnitude ↓ | Close command ratio | Mean latency ms ↓ | Median latency ms ↓ |
|---|---:|---:|---:|---:|---:|
| BC | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 |
| OpenPI0 v1/v2 | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 | 已有 step log 可算 |

表 4：failure taxonomy

| Failure mode | BC | OpenPI0 v1 | OpenPI0 v2 | Representative run_dir |
|---|---:|---:|---:|---|
| no close | TBD | TBD | TBD | TBD |
| premature close | TBD | TBD | TBD | TBD |
| miss object | TBD | TBD | TBD | TBD |
| slip after lift | TBD | TBD | TBD | TBD |
| clipped/stuck | TBD | TBD | TBD | TBD |

图 1：系统总览图

- 左侧：Kinova + wrist camera。
- 中间：A100 OpenPI0 server。
- 右侧：local safety adapter + rollout logger。
- 下方：raw action vs safe action diagnostic。

图 2：raw vs safe action 分布

- xyz raw/safe histogram。
- clip magnitude over rollout。

图 3：失败案例图

- 每个 failure mode 一行。
- 包含 final wrist image、trajectory/step timeline、gripper command。

## 已有证据支持 vs 需要继续实验

### 已有证据支持

- Kinova + OpenPI0 本地部署客户端存在。
- A100 server inference 通过 WebSocket policy endpoint 的 local client contract 已实现。
- 真实部署日志存在，包含 13,739 条 step-level records。
- OpenPI0 和 BC 两类 deployment logs 都存在。
- raw action 与 safe action 的差异非常明显，尤其 OpenPI0 是 12,593/12,593。
- 当前 step logs 可用于分析 clipping、latency、gripper command、raw/safe action divergence。

### 需要继续实验

- 任务成功率。
- BC vs OpenPI0 的性能对比。
- OpenPI0 v1 vs v2 是否更好。
- clipping threshold 是否提升成功率。
- gripper mode 是否减少 no-close 或 premature-close。
- 中文 prompt 是否可靠。
- distractor/black-X task 是否能完成。
- 方法是否适合 ICRA/IROS/CoRL/RA-L 主会。

## 最小下一步执行清单

1. 新建 `trial_eval.csv`，字段如下：

```text
run_dir, method, checkpoint, dataset_version, prompt, task, scene_id,
max_delta_m, gripper_mode, steps, success, failure_mode, notes
```

2. 先人工标注已有 20 个 OpenPI0 run 和 20 个 BC run，测试 failure taxonomy 是否够用。
3. 做 30 次 pilot：BC、OpenPI0 v1、OpenPI0 v2 各 10 次。
4. 如果 pilot 安全且标签清楚，再扩展到每组 20 次。
5. 实验完成前，论文所有结果句统一写为 `[RESULTS NEED EVIDENCE]` 或 “we will evaluate”。
