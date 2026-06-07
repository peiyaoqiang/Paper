# 笔记：Kinova OpenPI0 科研项目诊断

## 范围
本次审计只使用 `E:\code\Paper` 当前仓库中可读到的本地证据，目标是判断 Kinova/OpenPI0 项目的真实论文潜力，并避免编造不存在的实验结果。

## 证据清单
- 根目录文档：
  - `README.md`：早期研究定位是 OpenVLA-guided eye-in-hand grasping with RGB-D refinement and closed-loop execution。
  - `HANDOFF.md`：记录初始工程骨架和预期贡献。
  - `PROGRESS_ARCHIVE_2026-04-06.md`：记录截至 2026-04-06 的真实机器人 OpenVLA 集成状态。
  - `REAL_ROBOT_RUNBOOK.md`：记录 RealSense、远程 OpenVLA、Kinova twist 控制的已验证启动流程。
- 论文草稿：
  - `paper/abstract_draft.md`：英文摘要仍写着 “expected to show”，说明结果结论还没有闭合。
  - `paper/outline.md`：只有论文大纲，没有结果细节。
- OpenPI0/pi0 部署：
  - `kinova_vla_collect/DEPLOY_OPENPI0_POLICY.md`：本地机器人端 OpenPI0 部署流程和 observation/action contract。
  - `kinova_vla_collect/src/kinova_vla_collect/deploy_openpi0_policy.py`：WebSocket 客户端、observation 组装、action chunk 解析、安全限制和 step 日志。
  - `kinova_vla_collect/outputs/deploy_runs/*_openpi0_deploy`：OpenPI0 部署日志、配置和 wrist RGB 图像。
- BC baseline 部署：
  - `kinova_vla_collect/DEPLOY_BC_POLICY.md`：GPU-hosted BC baseline 部署流程。
  - `kinova_vla_collect/src/kinova_vla_collect/deploy_bc_policy.py`：BC 部署客户端和日志逻辑。
  - `kinova_vla_collect/outputs/deploy_runs/*_bc_deploy`：BC 部署日志、配置和 wrist RGB 图像。
- 数据采集与转换：
  - `kinova_vla_collect/src/kinova_vla_collect/recorder.py`：记录 LeRobot-style intermediate episode，包含 14-D state 和 7-D action。
  - `kinova_vla_collect/src/kinova_vla_collect/convert_to_lerobot.py`：只转换 `success: true` 的 raw episode，并验证 shape 和 gripper label。
  - 当前仓库快照中没有找到本地 `datasets/data/lerobot/.../meta/info.json` 或 `summary.json`。
- 图片和视频：
  - 没有找到 `.mp4`、`.avi`、`.mov`、`.webm` 文件。
  - 部署图像存在于 `kinova_vla_collect/outputs/deploy_runs/*/images/*.jpg`。
  - OpenPI RGB 捕获图存在于 `openpi_kinova_client/analysis/captures/openpi_rgb_*.png`。

## 定量产物审计
- `kinova_vla_collect/outputs/deploy_runs` 下共有 163 个部署目录。
- BC deploy：62 个目录；55 个目录含 `steps.jsonl`；共有 1,146 条 step 记录；141 张保存图像；2 个 smoke artifact。
- OpenPI0 deploy：101 个目录；100 个目录含 `steps.jsonl`；共有 12,593 条 step 记录；1,277 张保存图像。
- 总 step 记录数：13,739。
- 总部署图像数：1,418。
- step 记录中显式含 `success`、`failure` 或 `result` 字段的数量：0。
- OpenPI0 run metadata 中的配置分布：
  - `pi0_kinova_red_ball_lora`：72 个 run。
  - `pi0_kinova_red_ball_lora_v1_to_v2`：29 个 run。
- OpenPI0 run metadata 中的数据集分布：
  - `kinova_red_ball`：72 个 run。
  - `kinova_red_ball_v2`：29 个 run。
- OpenPI0 prompts 包含 red-ball pickup、green-ball pickup、英文 prompt 变体、中文 prompt 变体，以及 red-ball-to-black-X 变体。
- OpenPI0 action safety clipping：
  - 12,593/12,593 条 step 的 raw action 与 safe action 不同。
  - raw gripper close-threshold `> 0.5`：1,202/12,593。
  - 记录到的 gripper command：11,391 open，1,202 close。
  - 每 step elapsed latency：均值约 402 ms，中位数约 335 ms。
- BC action safety clipping：
  - 1,067/1,146 条 step 的 raw action 与 safe action 不同。
  - 记录到的 gripper command：791 open，199 close，156 hold。
  - 每 step elapsed latency：均值约 4,666 ms，中位数约 4,480 ms。

## Claim Ledger
- Claim：项目已经从纯软件骨架推进到真实机器人感知、远程策略推理和闭环执行。
  - 来源证据：`PROGRESS_ARCHIVE_2026-04-06.md` 记录 RealSense RGB-D 采集、远程 OpenVLA 调用、Kinova 安全小步执行和日志系统；`kinova_vla_collect/outputs/deploy_runs` 中有 13,739 条部署 step 记录。
  - 允许表述：系统已经具备真实机器人部署链路，包括 wrist RGB observation、14-D robot state、远程策略推理、本地 action adaptation 和 step-level rollout logging。
  - 禁止过强表述：系统已经实现鲁棒自主抓取；系统已有可靠抓取成功率。
  - 不确定性：部署日志缺少 success/failure 标签。
  - 下一步证据：trial 级人工成功标签和失败模式表。
- Claim：真实 OpenPI0/pi0 Kinova 部署客户端已经实现。
  - 来源证据：`DEPLOY_OPENPI0_POLICY.md`、`deploy_openpi0_policy.py`、run config 中的 OpenPI metadata、checkpoint path、observation keys 和 action semantics。
  - 允许表述：本地机器人客户端兼容 OpenPI WebSocket policy endpoint，并记录 raw/safe 7-D action。
  - 禁止过强表述：项目复现了 pi0 的全部训练和评测结果。
  - 不确定性：GPU server 侧训练代码、完整 OpenPI config 和训练日志不在本地仓库中。
  - 下一步证据：server 侧训练配置、训练曲线、checkpoint validation 日志。
- Claim：action-space mismatch 和 safety clipping 是当前最强、最清楚的部署现象。
  - 来源证据：`SafetyLimiter` 对 xyz/rpy/gripper 做裁剪；OpenPI0 有 12,593/12,593 条 step 被 raw-vs-safe 比较识别为裁剪，BC 有 1,067/1,146 条。
  - 允许表述：动作空间不匹配和安全适配是当前部署行为中的主导现象。
  - 禁止过强表述：当前 safety adapter 已经提升任务成功率。
  - 不确定性：没有无 adapter、不同 adapter 或不同 clipping 阈值的消融成功率。
  - 下一步证据：direct policy、hard clipping、learned/rescaled adapter 的 matched trial 对比。
- Claim：仓库具备 BC vs OpenPI0 的 baseline 基础设施。
  - 来源证据：BC deployment client、OpenPI0 deployment client 和两类真实部署日志。
  - 允许表述：仓库包含 BC 和 OpenPI0 两套可部署客户端，且 observation/action interface 可对齐。
  - 禁止过强表述：OpenPI0 已经优于 BC。
  - 不确定性：缺少同任务、同初始分布、同评价协议下的 outcome label。
  - 下一步证据：matched trials 和 success/failure/failure mode 表。
- Claim：当前论文摘要如果写成已完成结果，会过强。
  - 来源证据：`paper/abstract_draft.md` 使用 “expected to show”；仓库中没有找到 success-rate table。
  - 允许表述：当前可写成系统与部署诊断研究，但结果 claim 必须在评测完成前保持条件式或保守表述。
  - 禁止过强表述：实验已经证明方法提升了成功率。
  - 不确定性：可能存在仓库外的人工记录或视频，但当前仓库不可见。
  - 下一步证据：结果表、图、带标签 rollout 视频或 trial sheet。

## 缺失证据
- OpenPI0 和 BC 部署的 trial-level success labels。
- 相同场景/任务分布下的 matched baseline comparison。
- safety clipping、action scale、gripper mode、observation/state mode、checkpoint version、prompt language、camera/color preprocessing 的消融实验。
- 失败类型标注：miss、wrong object、no close、premature close、slip、collision、depth/pose/localization failure、place offset。
- 训练数据集摘要：episode 数、frame 数、success-only export、gripper-label distribution、train/validation split。
- OpenPI0/pi0 训练细节：config、checkpoint step、dataset root、LoRA/full fine-tuning、compute、loss curve、validation rollout。
- 重复试验和置信区间；在 trial label 和重复单位未定义前，不应做显著性结论。
