# 基于视觉-语言-动作模型的真实机器人抓取系统研究报告

## 摘要

随着深度学习、多模态学习和机器人控制技术的发展，视觉-语言-动作模型（Vision-Language-Action Model, VLA）逐渐成为具身智能研究中的重要方向。传统机器人抓取方法通常依赖人工设计的视觉检测、几何规划和控制规则，泛化能力有限；而 VLA 模型能够同时理解图像、语言指令和动作空间，为机器人完成开放词汇目标抓取任务提供了新的思路。

本文基于当前代码工程，总结一个面向 Kinova Gen3 机械臂的真实机器人抓取系统。系统使用腕部 RealSense 相机采集 RGB-D 图像，读取机器人末端位姿和关节状态，将视觉观测、机器人状态和语言指令输入 OpenVLA/OpenPI0 策略模型。模型输出 7 维动作后，系统通过动作安全裁剪、工作空间约束、RGB-D 几何修正和闭环执行，使机械臂完成目标接近、夹爪闭合与抬升验证。该项目体现了深度学习模型在真实机器人操作任务中的部署流程，也展示了多模态策略模型与传统几何控制方法结合的可行性。

## 1. 研究方向与任务背景

本项目选择的深度学习研究方向是：

**基于视觉-语言-动作模型的真实机器人抓取与操作。**

该方向属于具身智能和机器人学习交叉领域。它的核心目标是让机器人根据自然语言指令和视觉输入，直接生成可执行动作。例如，用户输入“pick up the red ball”，机器人需要从腕部相机图像中理解目标物体，再控制机械臂和夹爪完成抓取。

与传统深度学习任务相比，本项目的特点在于：

- 输入是多模态数据，包括图像、语言和机器人状态；
- 输出不是分类标签，而是连续机器人动作；
- 模型需要部署到真实硬件环境中；
- 系统必须考虑安全限制、坐标变换和闭环控制。

## 2. 系统总体结构

当前代码工程主要包含以下模块：

- `drivers/`：RealSense 相机、Kinova 机械臂、夹爪驱动；
- `policy/`：OpenVLA 策略模型接口；
- `adapters/`：模型动作到安全机器人动作的转换；
- `geometry/`：基于深度图的抓取位姿修正；
- `calibration/`：手眼标定和坐标变换；
- `executor/`：完整抓取状态机；
- `kinova_vla_collect/`：数据采集、策略部署和实验日志；
- `analysis/`：实验记录和结果分析。

系统流程如下：

1. 腕部 RealSense 相机采集 RGB-D 图像；
2. Kinova 机械臂读取当前末端位姿、关节状态和夹爪状态；
3. 将图像、状态和语言指令发送给 OpenVLA/OpenPI0 策略模型；
4. 模型输出 7 维动作；
5. 动作经过安全裁剪和工作空间约束；
6. 使用深度图和手眼标定结果修正抓取位置；
7. 机械臂执行接近、夹爪闭合和抬升动作；
8. 保存实验日志和图像记录。

## 3. 多模态输入与动作定义

项目中将一次机器人观测定义为语言指令、相机图像和机器人状态的组合。核心数据结构位于 `common/types.py`。

```python
@dataclass
class Observation:
    instruction: str
    frame: CameraFrame
    robot_state: RobotState


@dataclass
class PolicyAction:
    delta_xyz_m: Vector3
    delta_yaw_deg: float
    gripper_command: str
    confidence: float
    target_pixel: Optional[Tuple[int, int]] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

其中 `Observation` 表示输入给策略模型的观测信息，`PolicyAction` 表示模型输出的动作建议。对于 OpenPI0 部署分支，动作统一定义为：

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

对应代码位于 `kinova_vla_collect/src/kinova_vla_collect/recorder.py`：

```python
ACTION_DEFINITION = "[dx, dy, dz, droll, dpitch, dyaw, gripper]"
ACTION_SEMANTICS = {
    "dx": "end-effector x delta in meters per control step",
    "dy": "end-effector y delta in meters per control step",
    "dz": "end-effector z delta in meters per control step",
    "droll": "end-effector roll delta in radians per control step",
    "dpitch": "end-effector pitch delta in radians per control step",
    "dyaw": "end-effector yaw delta in radians per control step",
    "gripper": "-1 desired open state, +1 desired close state",
}
```

这种设计将深度学习模型输出直接表示为末端执行器的增量控制量，便于真实机器人闭环执行。

## 4. 策略模型接口设计

本项目通过远程 API 或 WebSocket 调用深度学习策略模型。`policy/openvla_wrapper.py` 中的 `OpenVLAWrapper` 封装了 OpenVLA 推理接口。

```python
class OpenVLAWrapper:
    def __init__(self, config: OpenVLAConfig) -> None:
        self.config = config

    def predict_action(self, observation: Observation) -> PolicyAction:
        if self.config.mode == "remote_api":
            return self._predict_action_remote(observation)
        return self._predict_action_mock(observation)
```

在远程推理模式下，系统会构造 JSON 请求，把语言指令、图像路径、RGB 图像 base64 编码和机器人状态发送给模型服务器：

```python
def _build_remote_payload(self, observation: Observation) -> dict[str, Any]:
    return {
        "instruction": observation.instruction,
        "unnorm_key": self.config.unnorm_key,
        "image_input_key": self.config.image_input_key,
        "frame": {
            "rgb_path_hint": observation.frame.rgb_path_hint,
            "depth_path_hint": observation.frame.depth_path_hint,
            "width": observation.frame.width,
            "height": observation.frame.height,
            "rgb_b64": self._maybe_base64_file(observation.frame.rgb_path_hint),
        },
        "robot_state": {
            "joint_positions": observation.robot_state.joint_positions,
            "ee_position_m": observation.robot_state.ee_position_m,
            "ee_yaw_deg": observation.robot_state.ee_yaw_deg,
            "gripper_opening_m": observation.robot_state.gripper_opening_m,
        },
    }
```

这部分代码体现了深度学习系统部署时常见的“本地机器人端 + 远程 GPU 推理端”结构。本地端负责采集真实传感器数据，远程端负责加载大模型并返回动作。

## 5. 动作安全裁剪与真实机器人约束

深度学习模型直接输出的动作可能存在尺度过大、超出工作空间或不适合真实机器人执行的问题。因此代码中引入了安全动作适配模块 `adapters/action_adapter.py`。

```python
@dataclass
class ActionAdapterConfig:
    max_translation_step_m: float
    max_rotation_step_deg: float
    workspace_xyz_min: Vector3
    workspace_xyz_max: Vector3
    safety_clipping_enabled: bool = True
    workspace_enforced: bool = True
```

核心逻辑是限制每一步最大平移和最大旋转，并根据当前末端位置检查下一步是否越界：

```python
def adapt(self, action: PolicyAction, robot_state: RobotState | None = None) -> SafeAction:
    if self.config.safety_clipping_enabled:
        delta_xyz_m, clipped_xyz_step = self._clip_delta(action.delta_xyz_m)
    else:
        delta_xyz_m = action.delta_xyz_m
        clipped_xyz_step = False

    delta_xyz_m, clipped_xyz_workspace = self._clip_to_workspace(delta_xyz_m, robot_state)
    delta_yaw_deg = action.delta_yaw_deg
    clipped_yaw = False

    if self.config.safety_clipping_enabled:
        if delta_yaw_deg > self.config.max_rotation_step_deg:
            delta_yaw_deg = self.config.max_rotation_step_deg
            clipped_yaw = True
        elif delta_yaw_deg < -self.config.max_rotation_step_deg:
            delta_yaw_deg = -self.config.max_rotation_step_deg
            clipped_yaw = True
```

该模块的意义在于：即使深度学习模型输出不稳定，机器人端仍然可以用规则约束保障执行安全。

## 6. RGB-D 几何修正方法

仅依赖 VLA 模型输出动作，容易受到视角偏差、深度估计不准和动作空间不匹配的影响。因此项目加入了 RGB-D 几何修正模块。相关代码位于 `geometry/grasp_refiner.py`。

```python
class GraspRefiner:
    def refine(self, policy_action: PolicyAction, observation: Observation) -> RefinedGrasp:
        robot_state = observation.robot_state
        if policy_action.target_pixel is None:
            return RefinedGrasp(
                target_xyz_m=robot_state.ee_position_m,
                target_yaw_deg=robot_state.ee_yaw_deg,
                grasp_width_m=self.config.default_grasp_width_m,
                quality=0.30,
                source="fallback",
                contact_xyz_m=None,
            )
```

当策略模型给出目标像素点后，系统会在深度图中采样该点深度，并将像素坐标反投影到相机三维坐标：

```python
depth = self.depth_filter.sample_target_depth(
    policy_action.target_pixel,
    observation.frame.depth_path_hint,
)
camera_xyz = self.tf_manager.project_pixel_to_camera_xyz(
    depth.pixel_xy,
    depth.depth_m,
    fx=observation.frame.fx,
    fy=observation.frame.fy,
    cx=observation.frame.cx,
    cy=observation.frame.cy,
)
```

随后结合手眼标定，把相机坐标转换到机器人基坐标，并计算最终腕部目标位置：

```python
base_xyz = self.tf_manager.camera_xyz_to_base_xyz(
    camera_xyz,
    robot_state.ee_position_m,
    robot_state.ee_yaw_deg,
    robot_state.ee_quaternion_xyzw,
)
tip_offset_in_base = self.tf_manager.ee_relative_xyz_to_base_offset(
    self.config.gripper_tip_offset_ee_m,
    robot_state.ee_yaw_deg,
    robot_state.ee_quaternion_xyzw,
)
wrist_target_xyz = tuple(
    target_axis - offset_axis
    for target_axis, offset_axis in zip(base_xyz, tip_offset_in_base)
)
```

这种做法将深度学习模型的语义理解能力与传统几何方法的精确定位能力结合起来，是本项目的重要设计思想。

## 7. 闭环抓取执行状态机

完整抓取流程由 `executor/task_state_machine.py` 实现。它把感知、策略推理、动作适配、几何修正和机器人执行串联成一个闭环流程。

```python
def run_once(self, instruction: str) -> ExecutionResult:
    trace: List[str] = []

    observation = self._build_observation(instruction)
    trace.append("observe")

    policy_action = self.policy.predict_action(observation)
    trace.append("policy_predict")

    safe_action = self.action_adapter.adapt(policy_action, observation.robot_state)
    trace.append("action_adapt")

    if safe_action.gripper_command == "open":
        self.gripper.open()
        trace.append("gripper_open")

    self.robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)
    trace.append("coarse_approach")
```

在粗接近之后，系统调用 RGB-D 修正模块得到更准确的抓取点，然后执行最终接近、夹爪闭合和抬升：

```python
refined_grasp = self.grasp_refiner.refine(policy_action, observation)
trace.append("rgbd_refine")

current_xyz = self.robot.get_state().ee_position_m
delta_to_refined = tuple(target - current for target, current in zip(refined_grasp.target_xyz_m, current_xyz))
self.robot.move_cartesian_delta(delta_to_refined, 0.0)
trace.append("final_approach")

self.gripper.close()
trace.append("grasp_close")

self.robot.move_cartesian_delta((0.0, 0.0, self.config.lift_height_m), 0.0)
trace.append("lift")
```

这说明系统并不是一次性执行模型输出，而是采用“模型粗引导 + 几何精修 + 闭环执行”的策略。

## 8. 数据采集与模型训练数据格式

`kinova_vla_collect/` 目录实现了数据采集流程，用于为 VLA 或行为克隆模型提供训练数据。系统支持 Xbox 遥操作和 Kinova teach-mode 示教。采集时会保存 wrist RGB 图像、机器人状态、动作、时间戳和任务文本。

核心保存逻辑如下：

```python
np.savez_compressed(
    shard_path,
    **{
        IMAGE_KEY: np.array(self._image_paths, dtype=str),
        STATE_KEY: states,
        ACTION_KEY: actions,
        TIMESTAMP_KEY: np.array(self._timestamps, dtype=np.float64),
        FRAME_INDEX_KEY: np.array(self._frame_indices, dtype=np.int32),
        EPISODE_INDEX_KEY: np.full((len(self._frame_indices),), episode_index, dtype=np.int32),
        TASK_KEY: task,
    },
)
```

其中图像字段、状态字段和动作字段采用 LeRobot 风格命名：

```text
observation.images.wrist
observation.state
action
task
timestamp
episode_index
frame_index
```

这种数据格式便于后续对 OpenPI/OpenVLA 模型进行微调，也可以用于训练行为克隆基线模型。

## 9. OpenPI0 真实机器人部署

项目中的 `kinova_vla_collect/src/kinova_vla_collect/deploy_openpi0_policy.py` 支持将 OpenPI0 策略部署到真实机器人。它会将图像缩放到模型输入尺寸，并组装 observation。

```python
def _make_observation(self, image: ImageArray, state: FloatArray) -> dict[str, Any]:
    image_for_policy = resize_with_pad(image, self.image_size, self.image_size)
    policy_state = state if self.policy_state_mode == "real" else np.zeros_like(state)
    if self.observation_format == "kinova_lerobot":
        return {
            "observation.images.wrist": image_for_policy,
            "observation.state": np.asarray(policy_state, dtype=np.float32),
            "task": self.task_prompt,
            "prompt": self.task_prompt,
        }
```

模型返回动作后，代码会进行尺度变换、夹爪语义处理和安全限制：

```python
def _make_safe_action(self, raw_action: FloatArray, state: FloatArray) -> FloatArray:
    action = np.asarray(raw_action[:7], dtype=np.float32).copy()
    action[:3] *= self.xyz_scale
    action[3:6] *= self.rotation_scale
    if self.invert_gripper:
        action[6] *= -1.0
    if self.gripper_mode == "ignore":
        action[6] = 0.0
    return self.safety.limit_action(action, current_position=state[:3])
```

这部分代码体现了真实部署时的关键问题：深度学习模型输出通常不能直接发送给机器人，必须先经过动作尺度对齐和安全过滤。

## 10. 实验设计

根据当前代码，可以设计以下实验：

1. **软件 dry-run 实验**  
   使用 mock 相机、mock 机器人和 mock 策略，验证完整流程是否能运行。

2. **真实感知 smoke test**  
   启动 RealSense 和 Kinova 状态读取，只进行一次模型推理，不执行动作，检查图像、状态和动作维度是否正确。

3. **保守真实闭环抓取实验**  
   在低频率、小步长条件下执行抓取，例如 3 Hz、每步最大 3 mm，观察机械臂是否能稳定接近目标。

4. **策略对比实验**  
   比较 OpenPI0/OpenVLA 策略、BC 行为克隆基线和几何修正模块对抓取成功率的影响。

5. **消融实验**  
   去掉 RGB-D 几何修正，仅使用 VLA 动作直接执行；再与加入几何修正后的结果对比。

可记录的指标包括：

- 抓取成功率；
- 平均执行步数；
- 模型推理延迟；
- 单次任务总耗时；
- 动作被安全裁剪的比例；
- 不同遮挡程度下的成功率。

## 11. 项目特点与不足

本项目的特点是将深度学习模型真正接入机器人硬件环境，而不是只停留在离线推理。代码中既包含多模态策略模型接口，也包含 RealSense、Kinova、夹爪、手眼标定、安全控制和实验日志等工程模块，系统完整性较强。

不过当前系统仍存在一些不足：

- OpenVLA/OpenPI0 的推理依赖远程 GPU 服务，本地实时性受网络影响；
- RGB-D 几何修正目前主要基于目标像素点和深度采样，复杂遮挡场景下仍可能失败；
- 成功判定在部分流程中较简单，需要结合真实夹爪力反馈或物体检测进一步完善；
- 不同模型 checkpoint 的动作语义可能不同，需要仔细做动作尺度和夹爪语义对齐。

## 12. 结论

本文总结了一个基于视觉-语言-动作模型的真实机器人抓取系统。该系统以 OpenVLA/OpenPI0 为深度学习策略核心，结合腕部 RGB-D 感知、Kinova Gen3 机械臂控制、动作安全裁剪、手眼标定和几何抓取修正，实现了从自然语言指令到真实机器人动作执行的完整链路。

从深度学习角度看，本项目体现了多模态模型从“理解图像和语言”到“生成真实动作”的发展趋势；从机器人角度看，项目说明大模型策略仍需要与几何约束、安全控制和闭环执行结合，才能在真实环境中稳定工作。因此，该工程适合作为“视觉-语言-动作模型在真实机器人抓取中的应用研究”的课程报告案例。

