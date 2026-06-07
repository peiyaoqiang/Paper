# 笔记：Kinova + pi0/OpenPI0 论文方向筛选

## 已有证据底座
- 真实硬件条件：Kinova Gen3、wrist-mounted Intel RealSense D435i、CTAG/Modbus gripper、A100 server inference。
- 已有部署链路：OpenPI0/pi0 WebSocket policy client、BC baseline client、ROS2 twist execution、SafetyLimiter、JSONL step logging。
- 已有日志规模：
  - `kinova_vla_collect/outputs/deploy_runs` 下有 163 个部署目录。
  - step-level 日志共 13,739 条。
  - OpenPI0 step 记录 12,593 条。
  - BC step 记录 1,146 条。
  - 保存部署图像 1,418 张。
- 最强现象：
  - OpenPI0 raw action 与 safe action 在 12,593/12,593 条 step 中不同。
  - BC raw action 与 safe action 在 1,067/1,146 条 step 中不同。
  - 说明 action-space mismatch 与 safety clipping 是当前最清楚、最可量化的部署现象。
- 关键限制：
  - step 日志没有 success/failure/result 字段。
  - 当前不能声称抓取成功率、鲁棒性或方法优于 baseline。
  - 数据集 summary、训练曲线、server 侧完整训练配置不在当前本地仓库中。

## 方向筛选标准
- 可落地性：是否能在现有 Kinova + A100 + OpenPI0 客户端条件下完成。
- 论文价值：是否不仅是工程复现，而能回答机器人/VLA 社区关心的问题。
- 证据起点：当前是否已经有代码或日志支持。
- 评测成本：是否能用 20-30 trials/condition 的小规模真实机器人实验形成可信证据。
- venue 适配：是否适合 ICRA/IROS/CoRL/RA-L/CVPR workshop。

## 初步推荐
最推荐方向应围绕“非原生机器人平台上的 VLA action-space alignment 与 safety-constrained deployment”，因为它同时满足：
- 当前日志已强烈显示该问题存在；
- 可用现有 BC/OpenPI0 客户端做对比；
- 不需要先证明大规模泛化；
- 可以通过小规模 matched trials 得到论文级证据。
