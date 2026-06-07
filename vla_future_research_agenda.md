# VLA 未来研究议程：基于 Kinova + pi0/OpenPI0 + A100 推理条件

## 结论

不要把当前项目只看成“复现 pi0/OpenPI0”。更有价值的研究问题是：**VLA 模型到了真实机器人上，为什么不能直接稳定工作，以及怎样用最少的真实机器人实验把它变得可控、可评估、可改进。**

结合 Kinova + wrist camera + gripper + A100 server inference，最值得做的方向可以分成三层：

- 近期最容易发论文：action-space alignment、safety-constrained deployment、failure taxonomy、BC/OpenPI0 对比。
- 中期更有研究价值：数据效率、prompt/目标泛化、checkpoint/dataset scaling、closed-loop recovery。
- 长期更有影响力：3D/force/memory/world-model enhanced VLA、长程任务规划、跨 embodiment transfer。

## 背景依据

相关主线文献：

- RT-2 提出将 web-scale vision-language knowledge 转移到机器人控制，说明 VLA 的核心愿景是视觉、语言和动作统一建模：https://arxiv.org/abs/2307.15818
- OpenVLA 提供开源 7B VLA，并强调新任务 fine-tuning 与多物体、多任务泛化：https://arxiv.org/abs/2406.09246
- pi0 使用 flow matching action head，目标是 general robot control：https://arxiv.org/abs/2410.24164
- OpenPI 官方仓库提供 pi0/openpi 的工程入口、checkpoint 和 fine-tuning 路线：https://github.com/Physical-Intelligence/openpi
- 近期 VLA survey 总结了 manipulation VLA 的架构、数据、评测和未来方向：https://arxiv.org/abs/2507.10672

这些文献只能作为研究背景。你自己的论文结果必须来自 Kinova 实验。

## 方向 1：Action-Space Alignment / Embodiment Adapter

### 可以研究什么
VLA 模型输出的 action 往往来自原始训练 embodiment 或数据集定义。换到 Kinova 后，action 的单位、坐标系、gripper 语义、速度限制、workspace 都可能不匹配。

### 研究问题
不同 action adapter 如何影响 VLA 在 Kinova 上的成功率、安全性和失败模式？

### 应做实验
- 比较 adapter：
  - direct small-step execution；
  - hard clipping；
  - scale + clipping；
  - workspace-aware clipping；
  - gripper-threshold adapter；
  - learned residual adapter，如果后续有数据。
- 任务：
  - pick up red ball；
  - pick up red ball with distractors；
  - place red ball on black X。
- 指标：
  - task success；
  - raw/safe action divergence；
  - clip ratio；
  - clip magnitude；
  - no-close / premature-close；
  - abort/unsafe count。

### 为什么适合你
你现在已经有 raw_action、safe_action、gripper_command、latency 和 run_config。这个方向最接近现有代码和日志。

## 方向 2：Data-Efficient Fine-Tuning / Demonstration Scaling

### 可以研究什么
VLA 模型能否用少量 Kinova demonstration 快速适配？数据数量、数据质量、gripper label、distractor coverage 哪个最关键？

### 研究问题
在 Kinova red-ball task 上，OpenPI0 的真实成功率如何随 demonstration 数量和数据版本变化？

### 应做实验
- 数据规模：
  - 10 episodes；
  - 25 episodes；
  - 50 episodes；
  - 100 episodes。
- 数据质量：
  - success-only；
  - success + failure；
  - clean gripper target label；
  - noisy gripper label。
- 数据场景：
  - single object；
  - distractor；
  - varied camera pose；
  - varied object start。
- 指标：
  - success rate；
  - data collection time；
  - training loss；
  - validation rollout success；
  - failure mode distribution。

### 为什么适合你
你已有 recorder、LeRobot-style conversion、OpenPI0 checkpoint metadata。但要做这个方向，必须把 server 侧训练日志、dataset summary 和 checkpoint config 补齐。

## 方向 3：Robustness to Scene, Prompt, and Object Variation

### 可以研究什么
VLA 的卖点是语言和视觉泛化，但真实机器人上常见失败来自光照、遮挡、distractor、prompt wording、目标颜色和物体位置变化。

### 研究问题
OpenPI0 在 Kinova 上的行为对 prompt、目标颜色、distractor 和遮挡有多敏感？

### 应做实验
- Prompt：
  - `pick up the red ball`；
  - `grab the red ball`；
  - `pick the red ball`；
  - `抓取红球`；
  - `pick up the green ball`。
- 场景：
  - 单红球；
  - 红球 + 绿球；
  - 红球 + distractor blocks；
  - 部分遮挡；
  - 黑色 X 放置目标。
- 指标：
  - target selection accuracy；
  - success rate；
  - wrong-object rate；
  - target-center motion trend；
  - language inconsistency cases。

### 为什么适合你
你现有 run_config 里已经出现中英文 prompt 和 green ball/red ball prompt。下一步只需要做 matched trial labels。

## 方向 4：Closed-Loop Recovery / Failure-Aware VLA Execution

### 可以研究什么
多数 VLA rollout 失败不是一步失败，而是逐渐偏离。可以研究机器人如何检测“快失败了”，并触发 retry、re-localize、re-grasp 或 human intervention。

### 研究问题
step-level logs 和 wrist images 能否预测 rollout failure，并通过简单 recovery policy 提高任务完成率？

### 应做实验
- Failure predictor：
  - 基于 action clipping 过大；
  - gripper 长时间 open；
  - end-effector 没有接近目标；
  - object 不在视野中心；
  - repeated low-motion/stuck。
- Recovery strategies：
  - reset to pre-grasp pose；
  - re-query policy with updated prompt；
  - reduce max_delta_m；
  - switch gripper mode；
  - ask human for one correction step。
- 指标：
  - recovery success；
  - extra steps；
  - reduced abort；
  - failure mode shift。

### 为什么适合你
你已经有 step-level logging，非常适合做 failure-aware execution。但需要目标检测或人工标注来判断是否接近目标。

## 方向 5：3D / Depth / Affordance-Guided VLA

### 可以研究什么
纯 RGB VLA 可能能理解目标，但最终抓取需要几何精度。可以研究 depth、目标点、grasp affordance 或 RGB-D refinement 是否能弥补 VLA 的低层控制误差。

### 研究问题
VLA 提供语义目标，RGB-D/3D 模块提供几何修正，是否比纯 VLA action 更可靠？

### 应做实验
- 对照：
  - VLA-only；
  - VLA + depth target localization；
  - VLA + RGB-D grasp refinement；
  - geometry-only baseline；
  - VLA selects target + geometry grasps。
- 指标：
  - target localization error；
  - grasp pose error；
  - success rate；
  - depth failure cases；
  - occlusion sensitivity。

### 为什么适合你
你早期项目有 RGB-D refinement 和 hand-eye calibration 思路。这个方向更像 ICRA/IROS 传统机器人论文，但需要把 depth pipeline 做稳。

## 方向 6：Cloud/Server Inference and Real-Time Control

### 可以研究什么
VLA 大模型常需要 GPU server。真实机器人闭环控制要求稳定频率和低延迟。A100 server inference 是你的实际条件，不是短板，也可以变成研究对象。

### 研究问题
远程 A100 VLA inference 的延迟、jitter、action chunking 如何影响真实机器人闭环控制？

### 应做实验
- 变量：
  - `hz = 2/3/5`；
  - `chunk_steps = 1/2/5`；
  - image size；
  - save_image_every；
  - WebSocket vs HTTP，如可行。
- 指标：
  - mean/median/p95 latency；
  - jitter；
  - success rate；
  - control smoothness；
  - stale action rate。

### 为什么适合你
你已有 A100 server 和 local client。但这个方向单独做可能偏系统 profiling，最好作为 action alignment 论文的实验之一。

## 方向 7：VLA Evaluation Benchmark for Small Labs

### 可以研究什么
很多实验室没有大规模机器人集群，但有单臂机器人和 GPU server。你可以设计一个小实验室可复现的 VLA evaluation protocol。

### 研究问题
能否用一个 Kinova 单臂、一个 wrist camera、一个 A100 server，建立低成本但信息密度高的 VLA 真实机器人评测协议？

### 应做实验
- 任务集合：
  - reach target；
  - pick red ball；
  - pick specified object among distractors；
  - place on black X；
  - recovery after failed grasp。
- 指标集合：
  - success；
  - safety；
  - latency；
  - action clipping；
  - gripper timing；
  - failure taxonomy。
- 输出：
  - protocol；
  - logging schema；
  - trial sheet；
  - visualization templates。

### 为什么适合你
这个方向适合 CVPR/ICRA workshop，也适合做你项目的长期基础设施。但主会论文可能需要更明确的新方法。

## 方向 8：Human-in-the-Loop VLA Correction

### 可以研究什么
真实部署中，让人类在关键失败点提供少量 correction，可能比重新训练大模型更实用。

### 研究问题
在 VLA rollout 即将失败时，少量 human correction 是否能显著提升成功率，并产生可用于后续 fine-tuning 的高价值数据？

### 应做实验
- 策略：
  - no intervention；
  - one-step teleop correction；
  - correction + resume VLA；
  - correction data added to fine-tuning。
- 指标：
  - success rate；
  - number of interventions；
  - correction time；
  - post-correction success；
  - data efficiency。

### 为什么适合你
你已经有 Xbox teleop collection scaffold。这个方向实用、容易展示，但需要设计清楚 intervention trigger。

## 推荐优先级

### 近期 1-2 个月：最应该做
1. **Action-Space Alignment 与 Safety-Constrained Deployment**
2. **Failure Taxonomy / Evaluation Protocol**
3. **A100 Inference Latency + Control Frequency**

这三者可以合成一篇最现实的论文：不是宣称“VLA 很强”，而是回答“VLA 到 Kinova 上怎样才可安全、可诊断、可评估”。

### 中期 2-4 个月：论文价值更高
4. **Data-Efficient Fine-Tuning**
5. **Robustness to Prompt/Object/Scene Variation**
6. **Closed-Loop Recovery**

这些方向需要更多真实试验，但更接近 CoRL/RA-L 的研究味道。

### 长期方向
7. **RGB-D / 3D / Affordance-guided VLA**
8. **Human-in-the-loop correction and continual adaptation**
9. **Small-lab VLA benchmark**

这些可以成为博士/硕士课题主线，而不只是一次复现实验。

## 你现在应该先做哪些实验

### 实验 1：建立 trial-level evaluation
先不要急着训练新模型。先把现有 rollout 变成可评价数据。

最小字段：

```text
run_dir, method, checkpoint, prompt, scene, max_delta_m, gripper_mode,
steps, success, failure_mode, final_image, notes
```

目标：

- 给 20 个 OpenPI0 run 和 20 个 BC run 做人工标签。
- 看 failure taxonomy 是否足够。
- 形成第一版 success/failure 表。

### 实验 2：BC vs OpenPI0 v1 vs OpenPI0 v2 matched trials
任务：pick up red ball。

设计：

- BC：20 trials。
- OpenPI0 v1：20 trials。
- OpenPI0 v2：20 trials。

指标：

- success rate；
- no-close rate；
- premature-close rate；
- slip/lift failure；
- clip ratio；
- mean latency。

目标：

- 判断 OpenPI0 是否真的比 BC 更适合当前 Kinova task。
- 如果不是，也能得到重要负结果和失败分析。

### 实验 3：Action adapter 消融
固定 OpenPI0 v2。

变量：

- `max_delta_m = 0.003 / 0.005 / 0.008`。
- gripper mode = passthrough / close_only / open_close。

指标：

- success；
- unsafe/abort；
- stuck due to clipping；
- close timing。

目标：

- 回答“失败是不是因为动作被裁得太小或 gripper 语义不对”。

### 实验 4：Prompt 和目标变化
固定 OpenPI0 v2 + 最安全 adapter。

变量：

- English canonical；
- English paraphrase；
- Chinese prompt；
- red ball vs green ball；
- with/without distractor。

指标：

- target correct；
- wrong object；
- success；
- behavior consistency。

目标：

- 判断 VLA 的语言/目标 grounding 在你的真实机器人上是否成立。

### 实验 5：Latency / Control Frequency
固定方法和任务。

变量：

- `hz = 2 / 3 / 5`；
- `chunk_steps = 1 / 2 / 5`。

指标：

- loop latency；
- jitter；
- success；
- motion smoothness；
- stale action failure。

目标：

- 判断 A100 server inference 是否是瓶颈。

## 最适合写成论文的组合

### 论文 A：最现实
题目方向：

> Safety-Constrained Deployment of VLA Policies on a Kinova Manipulator

实验组合：

- trial-level evaluation；
- BC vs OpenPI0；
- adapter ablation；
- failure taxonomy；
- latency diagnostic。

适合：

- ICRA/IROS workshop；
- CVPR embodied AI workshop；
- IROS/ICRA system paper，如果实验足够扎实。

### 论文 B：更偏 CoRL
题目方向：

> Data-Efficient OpenPI0 Adaptation for Eye-in-Hand Manipulation

实验组合：

- data scaling；
- checkpoint comparison；
- prompt/object variation；
- real robot success；
- failure mode。

适合：

- CoRL workshop；
- CoRL short/regular，取决于结果强度。

### 论文 C：更偏 RA-L/ICRA
题目方向：

> Vision-Language-Guided RGB-D Grasp Refinement for Real-Robot Manipulation

实验组合：

- VLA-only；
- geometry-only；
- VLA + RGB-D refinement；
- clutter/occlusion；
- failure analysis。

适合：

- RA-L；
- ICRA/IROS。

## 不建议现在优先做的事

- 不建议一开始追“大规模多任务泛化”。你的硬件和数据规模还不支撑。
- 不建议把重点放在新 VLA 模型结构。A100 和单臂 Kinova 更适合做 adaptation/evaluation。
- 不建议只报 demo 视频，不做 trial table。机器人论文会很难过审。
- 不建议只优化 prompt，不标注 failure mode。prompt 结果很容易被认为偶然。
- 不建议把已有 13,739 step 当成 13,739 次成功实验。它们是 step-level logs，不是 trial outcomes。

## 最小可执行路线

第一周：

- 建 `trial_eval.csv`。
- 人工标注 40 个已有 run。
- 整理 failure taxonomy。

第二周：

- 跑 60 次 matched trials：BC、OpenPI0 v1、OpenPI0 v2 各 20 次。
- 记录 final image/video。

第三周：

- 做 adapter ablation：`max_delta_m` 和 gripper mode。
- 生成 raw/safe action 分布图、failure mode 表。

第四周：

- 写 workshop/ICRA-style paper draft。
- 摘要中只写已有结果；所有未完成项标 `[NEEDS EVIDENCE]`。
