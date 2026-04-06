# OpenVLA Remote API Example / OpenVLA 远程推理接口示例

## Purpose / 文档目的

**English**

This document shows a minimal request and response format for using `policy/openvla_wrapper.py`
in `remote_api` mode.

It is intended for the setup where:

- the local machine runs camera, robot, and grasp execution
- a remote server runs `OpenVLA` inference
- the local policy wrapper sends an HTTP `POST` request to the remote server

**中文**

本文档给出 `policy/openvla_wrapper.py` 在 `remote_api` 模式下可使用的一组最小请求与响应格式示例。

适用场景为：

- 本地机器运行相机、机械臂和抓取执行逻辑
- 远程服务器运行 `OpenVLA` 推理
- 本地策略封装通过 HTTP `POST` 请求调用远程推理服务

## Config Example / 配置示例

Set the `policy` section in `configs/default_config.json` like this:

在 `configs/default_config.json` 中，可将 `policy` 配置设为：

```json
"policy": {
  "model_name": "openvla-remote",
  "mode": "remote_api",
  "remote_url": "http://YOUR_SERVER_IP:8000/predict",
  "remote_timeout_s": 10.0,
  "unnorm_key": "libero_spatial",
  "image_input_key": "wrist_image"
}
```

## Request Format / 请求格式

The local wrapper sends a JSON body like this:

本地 wrapper 会发送如下 JSON：

```json
{
  "instruction": "pick up the red mug",
  "unnorm_key": "libero_spatial",
  "image_input_key": "wrist_image",
  "frame": {
    "rgb_path_hint": "mock_rgb_frame_0001.png",
    "depth_path_hint": "mock_depth_frame_0001.npy",
    "width": 640,
    "height": 480,
    "rgb_b64": null
  },
  "robot_state": {
    "joint_positions": [
      0.0,
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6
    ],
    "ee_position_m": [
      0.45,
      0.00,
      0.30
    ],
    "ee_yaw_deg": 0.0,
    "gripper_opening_m": 0.08
  }
}
```

Notes:

- `rgb_b64` is included only when `rgb_path_hint` points to a real local image file.
- `image_input_key` is intended for the remote server to choose which image source to use, such as `wrist_image`.
- `unnorm_key` should match the checkpoint statistics used by the remote model.

说明：

- 当 `rgb_path_hint` 指向真实本地图像文件时，wrapper 会附带 `rgb_b64`。
- `image_input_key` 供远程服务决定使用哪一路图像输入，例如 `wrist_image`。
- `unnorm_key` 需要与远程模型的动作反归一化统计项一致。

## Response Format / 响应格式

The remote server can return either of the following two styles.

远程服务可以返回以下两种风格中的任意一种。

### Style A / 风格 A

Return the raw OpenVLA action vector:

直接返回 OpenVLA 原始动作向量：

```json
{
  "action": [
    0.01,
    -0.02,
    -0.03,
    0.00,
    0.00,
    -0.05,
    1.00
  ],
  "confidence": 0.93,
  "target_pixel": [
    320,
    240
  ],
  "notes": "remote OpenVLA action"
}
```

Interpretation in the current local wrapper:

- `action[0:3]` -> `delta_xyz_m`
- `action[5]` -> `delta_yaw_deg`
- `action[6] > 0.5` -> `gripper_command = "close"`
- otherwise -> `gripper_command = "open"`

当前本地 wrapper 的解析方式：

- `action[0:3]` 对应 `delta_xyz_m`
- `action[5]` 对应 `delta_yaw_deg`
- `action[6] > 0.5` 时视为 `gripper_command = "close"`
- 否则视为 `gripper_command = "open"`

### Style B / 风格 B

Return an already-adapted action:

直接返回已解析后的动作：

```json
{
  "delta_xyz_m": [
    0.01,
    -0.02,
    -0.03
  ],
  "delta_yaw_deg": -3.0,
  "gripper_command": "open",
  "confidence": 0.88,
  "target_pixel": [
    320,
    240
  ],
  "notes": "remote adapted action"
}
```

## Minimal Server Logic / 最小服务端逻辑

The remote server only needs to:

1. receive the JSON request
2. decode `rgb_b64` if present, or load an image using a server-side rule
3. run `OpenVLA` inference
4. return one of the response formats above

远程服务最少只需要完成：

1. 接收 JSON 请求
2. 若存在 `rgb_b64` 则解码图像，否则按服务端规则读取图像
3. 运行 `OpenVLA` 推理
4. 按上面的任一格式返回结果

## Testing Suggestion / 测试建议

Recommended order:

建议顺序：

1. keep local `mode = "mock"` and verify the main loop still runs
2. start a minimal remote server that returns a fixed action
3. switch local `mode` to `remote_api`
4. verify that the local demo can receive and execute the returned action
5. replace the fixed response with actual `OpenVLA` inference

1. 先保持本地 `mode = "mock"`，确认主流程仍然可运行
2. 在远程启动一个返回固定动作的最小服务
3. 把本地 `mode` 切换为 `remote_api`
4. 验证本地 demo 能接收并执行远程返回动作
5. 最后再把固定返回替换成真实 `OpenVLA` 推理
