# 基于 OpenVLA 的眼在手上抓取系统

这是一个面向论文原型验证的最小工程骨架，目标是把以下三部分串成一条可落地的真实机器人抓取链路：

- 使用 `OpenVLA` 生成语言条件下的高层动作建议
- 使用 `RGB-D` 几何信息完成局部抓取位姿修正
- 使用闭环执行提升眼在手上抓取的真实成功率

当前版本已经提供：

- 工程目录结构
- 可运行的最小 Python 骨架
- 模拟版相机、机械臂、夹爪和 OpenVLA 接口
- 抓取状态机主流程
- 论文摘要初稿

当前版本还没有接入：

- 真实 `Intel RealSense D435i`
- 真实 `Kinova Gen3`
- 真实夹爪控制
- 真实 `OpenVLA` 推理

## 目录说明

- `configs/`：配置文件
- `common/`：共享数据结构与类型定义
- `drivers/`：相机、机械臂、夹爪驱动接口
- `calibration/`：手眼标定与坐标变换
- `policy/`：OpenVLA 推理封装
- `adapters/`：动作适配与安全裁剪
- `geometry/`：深度处理与抓取几何修正
- `executor/`：闭环抓取状态机
- `experiments/`：实验入口脚本
- `analysis/`：结果统计与绘图占位目录
- `paper/`：论文摘要和提纲草稿

## 系统流程

当前工程按以下流程设计：

1. 采集腕部相机 `RGB-D` 图像和机器人状态
2. 将 `RGB + 语言指令` 输入 OpenVLA
3. 把 OpenVLA 输出转换为可执行的安全动作
4. 机器人完成粗接近
5. 利用深度和手眼标定做局部抓取修正
6. 末端闭环接近后闭合夹爪
7. 抬升验证抓取是否成功

## 快速开始

建议使用 `Python 3.10+`。

运行最小 demo：

```bash
python experiments/run_demo.py
```

当前 demo 使用的是 mock 版本硬件和 mock 版本 OpenVLA，因此可以在还没有接入真实设备之前，先验证整条软件流程是否打通。

## 当前入口

主入口脚本：

- `experiments/run_demo.py`

默认配置文件：

- `configs/default_config.json`

论文摘要草稿：

- `paper/abstract_draft.md`

## 你接下来最先要替换的模块

### 1. 相机驱动

文件：

- `drivers/realsense_driver.py`

需要替换为：

- `pyrealsense2` 或 ROS 下的 RealSense 采集逻辑

### 2. 机械臂驱动

文件：

- `drivers/kinova_driver.py`

需要替换为：

- Kinova API 或 ROS 控制接口

### 3. OpenVLA 推理

文件：

- `policy/openvla_wrapper.py`

需要替换为：

- 实际模型加载
- 输入预处理
- 推理调用
- 动作结果解析

### 4. 几何抓取修正

文件：

- `geometry/grasp_refiner.py`

需要补充：

- 深度图真实采样
- 目标区域筛选
- 抓取点与法向估计
- 桌面碰撞约束

## 建议的开发顺序

1. 先打通 `D435i` 图像采集和时间同步
2. 再打通 `Kinova Gen3` 的笛卡尔增量控制
3. 再验证手眼标定坐标变换是否正确
4. 再接入真实 `OpenVLA`
5. 最后补上深度修正和实验日志记录

## 论文定位

这套工程更适合支撑下面这个方向的论文：

**基于 OpenVLA 引导与 RGB-D 几何修正的眼在手上杂乱场景抓取**

对应的核心思想是：

- `OpenVLA` 负责高层语义理解和粗动作建议
- `RGB-D` 几何模块负责真实抓取位姿修正
- 闭环执行模块负责提升最终抓取稳定性

## 文档约定

从现在开始，本项目中的新增说明文档默认采用 **中英文双语** 形式编写。  
如果某份旧文档暂时只有单语版本，可以后续逐步补齐。
