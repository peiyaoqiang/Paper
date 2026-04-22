# openpi Kinova Local Client

最小链路：

1. 远程 A100 服务器运行 openpi policy server。
2. 本地电脑采集相机和 Kinova 状态，组装 DROID observation。
3. 本地通过 WebSocket 发送 observation，接收 `action_chunk = response["actions"]`。
4. 先用 `run_mock.py` 只打印动作；确认动作尺度后，再用 `run_kinova.py` 接 Kortex。

## 远程服务器

在远程 openpi 环境里运行其自带 server，例如：

```bash
cd $OPENPI_ROOT
conda activate openpi
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_droid \
  --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
  --port=8000
```

如果服务器有防火墙，先用 SSH 隧道最省心：

```bash
ssh -L 8000:127.0.0.1:8000 user@REMOTE_SERVER
```

然后本地 client 用 `--host 127.0.0.1 --port 8000`。

## 本地安装

```bash
cd /home/pyq/Paper/openpi_kinova_client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果用真机 Kortex，把 Kinova 提供的 `kortex_api` Python 包也装进同一个环境。

## 先只打印 openpi action chunk

```bash
python -m openpi_kinova_client.run_mock \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "pick up the red mug" \
  --steps 5 \
  --camera-index 0
```

没有相机时：

```bash
python -m openpi_kinova_client.run_mock --host 127.0.0.1 --port 8000 --dummy-images --steps 5
```

## 不接 openpi，测试本地 client 流程

终端 A：

```bash
python -m openpi_kinova_client.mock_policy_server --port 8000
```

终端 B：

```bash
python -m openpi_kinova_client.run_mock --host 127.0.0.1 --port 8000 --dummy-images --steps 3
```

## 接 Kinova 真机

先把速度、步长、workspace 都保守一点，确认急停文件可用：

```bash
python -m openpi_kinova_client.run_kinova \
  --host 127.0.0.1 \
  --port 8000 \
  --robot-ip 192.168.1.10 \
  --prompt "pick up the red mug" \
  --steps 20 \
  --dummy-images
```

急停：

```bash
touch /tmp/openpi_kinova_estop
```

恢复前需要删除急停文件，并重新启动程序：

```bash
rm /tmp/openpi_kinova_estop
```

## 复用当前 Paper/OpenVLA 的 RealSense + ROS2 Kinova 栈

如果相机和 Kinova 已经按本仓库原来的 OpenVLA 流程跑通，优先用这个入口。它读取
`/home/pyq/Paper/configs/default_config.json`，复用：

- `drivers/realsense_driver.py`
- `drivers/kinova_driver.py`
- `drivers/gripper_driver.py`

先用真实相机、远程 openpi，只打印动作，不动机械臂：

```bash
cd /home/pyq/Paper/openpi_kinova_client

python3 -m openpi_kinova_client.run_paper_stack \
  --host 127.0.0.1 \
  --port 8000 \
  --instruction "pick up the red mug" \
  --steps 3 \
  --position-scale 0.02 \
  --rotation-scale 0.05
```

确认相机图像和 action 都正常后，再真正发 ROS2 twist：

```bash
python3 -m openpi_kinova_client.run_paper_stack \
  --host 127.0.0.1 \
  --port 8000 \
  --instruction "pick up the red mug" \
  --steps 10 \
  --position-scale 0.02 \
  --rotation-scale 0.05 \
  --max-translation 0.005 \
  --max-rotation 0.03 \
  --chunk-steps 1 \
  --gripper-mode ignore \
  --execute
```

默认 `--gripper-mode ignore`，因为 `pi0_droid` 返回 8 维 action，夹爪/终止位语义需要先确认。

## 重要约定

`action_adapter.py` 默认把模型输出每一行的前 7 维解释成：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

默认单位假设是：

- `dx/dy/dz`: 米级增量，会乘 `position_scale`
- `droll/dpitch/dyaw`: 弧度增量，会乘 `rotation_scale`
- `gripper`: 大于阈值表示 close，小于等于阈值表示 open

不同 openpi checkpoint / 数据集 action 语义可能不同。第一次真机前，请先跑 `run_mock.py` 观察 action chunk 数值范围，再调整 `--position-scale`、`--rotation-scale`、`--invert-gripper`。

参考：

- openpi remote inference: https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md
- openpi websocket client/server: `packages/openpi-client` and `src/openpi/serving/websocket_policy_server.py`
- Kinova Kortex API docs: https://docs.kinovarobotics.com/
