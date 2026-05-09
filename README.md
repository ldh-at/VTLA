## VTLA LeRobot Tactile Fork

이 저장소는 Hugging Face의 원래 `LeRobot`을 기반으로, VTLA/VLA 실험에서 촉각 센서를 같이 쓰기 위해 수정한 fork입니다.
원래 LeRobot의 로봇 제어, 카메라 기록, 데이터셋 저장, 학습 스크립트 구조는 최대한 유지했고, 여기에 Paxini tactile sensor를 SO-101 follower와 함께 읽을 수 있는 경로를 추가했습니다.

쉽게 말하면 원래 LeRobot이 아래 데이터를 주로 다뤘다면,

- 로봇 관절 상태: `observation.state`
- 카메라 이미지: `observation.images.*`
- 행동 라벨: `action`
- 언어/task 입력: policy에 따라 language token

이 fork는 거기에 아래 촉각 데이터를 하나 더 붙입니다.

- 촉각 맵: `observation.tactile.primary`

촉각 값을 RGB 이미지처럼 억지로 바꾸지 않고, `float32` 2D tactile map으로 저장합니다. 그래서 나중에 Pi0/Pi0.5 같은 VLA policy가 이미지, 로봇 상태, 언어 입력과 함께 tactile token을 따로 사용할 수 있습니다.

### 전체 데이터 흐름

현재 구현된 흐름은 다음 순서입니다.

1. LeRobot의 기존 `SOFollower`가 모터 상태와 카메라 이미지를 읽습니다.
2. 새로 추가한 `PaxiniSO101Follower`가 `SOFollower`를 상속해서 기존 observation을 그대로 받습니다.
3. 그 다음 Paxini tactile 센서를 `SerialPaxiniReader`로 읽습니다.
4. UART response에서 taxel raw 값을 꺼냅니다.
5. `TactileHeatmapRenderer`가 taxel 위치 정보를 이용해서 2D tactile heatmap으로 바꿉니다.
6. 최종 observation에 `observation.tactile.primary`가 추가됩니다.
7. LeRobotDataset 저장 시 이 값은 `FeatureType.TACTILE`로 분류됩니다.
8. Pi0/Pi0.5 학습 시 tactile feature를 prefix token으로 encode해서 vision/language token과 함께 넣습니다.

정리하면, 이 fork의 목표는 `카메라 + 로봇 상태 + action`만 쓰던 LeRobot 데이터 구조에 `촉각 맵`을 정식 feature로 추가하는 것입니다.

### 원래 LeRobot과 가장 큰 차이

| 구분 | 원래 LeRobot | 이 VTLA fork |
| --- | --- | --- |
| 로봇 타입 | `so100_follower`, `so101_follower` 등 기본 follower | `paxini_so101_follower` 추가 |
| tactile 센서 | 기본 지원 없음 | Paxini UART 센서 reader 추가 |
| tactile 데이터 저장 | 없음 | `observation.tactile.primary` |
| tactile feature 타입 | 없음 | `FeatureType.TACTILE` |
| tactile 값 형태 | 없음 | normalized `float32` 2D map |
| VLA 학습 | image/state/language/action 중심 | Pi0/Pi0.5에서 tactile prefix token 추가 |
| 다른 센서 적용 | 직접 robot/policy 수정 필요 | sensor reader와 renderer만 바꾸면 재사용 가능하게 구성 |

### 실제로 추가/수정된 코드

#### 1. Paxini 로봇 통합 코드

`src/lerobot_robot_paxini/` 아래에 Paxini tactile sensor를 SO-101 follower에 붙이는 코드가 들어 있습니다.

- `config_paxini_so101.py`
  - `paxini_so101_follower` robot type을 등록합니다.
  - tactile serial port, baudrate, device id, read address, taxel dtype, tactile map size 같은 설정을 둡니다.
  - 기본값은 테스트가 쉽도록 `tactile_mock=True`입니다. 실제 센서를 읽으려면 `--robot.tactile_mock=false`로 바꿔야 합니다.

- `paxini_so101.py`
  - 기존 `SOFollower`를 상속합니다.
  - 기존 motor/camera observation에 `observation.tactile.primary`를 추가합니다.
  - force/torque 값도 `paxini.fx`, `paxini.fy`, `paxini.fz`, `paxini.tx`, `paxini.ty`, `paxini.tz`로 같이 넣습니다.

- `paxini_reader.py`
  - Paxini UART protocol을 Python에서 읽을 수 있게 구현한 파일입니다.
  - vendor 예제 코드의 구조를 반영해서 request header는 `55 AA`, response header는 `AA 55`로 처리합니다.
  - little-endian length/address field, device id, function code, status byte, LRC checksum을 검증합니다.
  - `MockPaxiniReader`도 있어서 실제 센서 없이 recording/training pipeline 테스트가 가능합니다.

- `tactile_render.py`
  - taxel raw vector를 2D tactile map으로 변환합니다.
  - `tactile_map_path`가 있으면 xlsx에서 taxel 좌표를 읽습니다.
  - 없으면 mock taxel map을 만들어서 테스트합니다.
  - 기본 출력 크기는 `64 x 64`입니다.

- `types.py`
  - taxel 배열과 force/torque 값을 담는 `PaxiniSample` dataclass가 있습니다.

#### 2. LeRobot feature/type 수정

LeRobot이 tactile 값을 그냥 state vector로 착각하지 않도록 feature type을 추가했습니다.

- `src/lerobot/configs/types.py`
  - `FeatureType.TACTILE` 추가

- `src/lerobot/utils/constants.py`
  - `OBS_TACTILE = "observation.tactile"` 추가

- `src/lerobot/utils/feature_utils.py`
  - hardware feature에서 tactile feature를 dataset feature로 변환합니다.
  - `observation.tactile.*`를 `FeatureType.STATE`가 아니라 `FeatureType.TACTILE`로 분류합니다.

- `src/lerobot/datasets/feature_utils.py`
  - dataset metadata를 policy input feature로 바꿀 때 tactile feature를 유지하도록 연결됩니다.

#### 3. Policy 쪽 tactile 입력 추가

- `src/lerobot/configs/policies.py`
  - policy config에서 tactile input feature만 쉽게 가져올 수 있는 helper를 추가했습니다.

- `src/lerobot/policies/tactile/`
  - tactile map을 token embedding으로 바꾸는 공통 encoder가 있습니다.
  - `cnn`과 `attention` encoder type을 지원합니다.

- `src/lerobot/processor/tactile_processor.py`
  - tactile shape 검증, temporal filtering 같은 processor step을 넣을 수 있게 했습니다.

- `src/lerobot/policies/pi0/`
  - Pi0에서 tactile map을 prefix token으로 넣을 수 있게 수정했습니다.
  - dataset에 `FeatureType.TACTILE` feature가 있으면 자동으로 tactile encoder를 구성합니다.

- `src/lerobot/policies/pi05/`
  - Pi0.5도 Pi0와 같은 방식으로 tactile prefix token을 사용할 수 있게 수정했습니다.
  - upstream pretrained Pi0/Pi0.5 checkpoint를 불러올 때 tactile encoder/head는 새로 초기화될 수 있도록 strict load를 조정했습니다.

### 실제 센서로 record 하는 예시

기본 config는 `tactile_mock=True`라서 tactile 값이 가짜로 생성됩니다. 실제 Paxini 센서를 연결할 때는 아래처럼 mock을 꺼야 합니다.

```bash
lerobot-record \
  --robot.type=paxini_so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.tactile_mock=false \
  --robot.tactile_port=/dev/ttyUSB0 \
  --robot.tactile_baudrate=115200 \
  --robot.tactile_device_id=1 \
  --robot.tactile_read_func_code=0x7B \
  --robot.tactile_read_addr=0x040E \
  --robot.tactile_taxel_dtype=uint16 \
  --robot.tactile_image_size=64 \
  --dataset.repo_id=<your_hf_id>/<your_dataset_name> \
  --dataset.single_task="your task" \
  --dataset.num_episodes=10
```

센서가 `/dev/ttyUSB0`에 보이지 않으면 WSL에서 USB pass-through가 제대로 잡혔는지 먼저 확인해야 합니다. Windows/WSL 환경에서는 포트 이름이 바뀔 수 있으므로 `ls /dev/ttyUSB* /dev/ttyACM*`로 확인하는 것이 좋습니다.

### tactile map xlsx를 쓰는 경우

Paxini 센서의 taxel 실제 좌표가 있으면 xlsx 파일을 넘길 수 있습니다.

```bash
--robot.tactile_map_path=/path/to/taxel_map.xlsx
```

xlsx에는 다음 중 하나의 column 조합이 있어야 합니다.

- `x_mm`, `y_mm`
- `x`, `y`
- `x(mm)`, `y(mm)`

좌표 파일을 넣으면 taxel raw vector가 실제 센서 배열 위치에 맞춰 `64 x 64` tactile map으로 render됩니다. 좌표 파일이 없으면 mock grid를 사용하므로, 실제 실험 결과 분석에는 좌표 파일을 넣는 쪽이 좋습니다.

### Pi0 / Pi0.5 tactile training

dataset에 `observation.tactile.primary`가 들어 있고, 해당 feature가 `FeatureType.TACTILE`로 잡히면 Pi0/Pi0.5는 tactile feature를 자동으로 감지합니다.

Pi0 예시:

```bash
lerobot-train \
  --policy.type=pi0 \
  --dataset.repo_id=<your_hf_id>/<your_tactile_dataset> \
  --policy.path=<pi0_pretrained_checkpoint_or_hub_id> \
  --policy.tactile_encoder_type=cnn \
  --policy.tactile_n_tokens=1 \
  --batch_size=4 \
  --steps=10000
```

Pi0.5 예시:

```bash
lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=<your_hf_id>/<your_tactile_dataset> \
  --policy.path=<pi05_pretrained_checkpoint_or_hub_id> \
  --policy.tactile_encoder_type=cnn \
  --policy.tactile_n_tokens=1 \
  --batch_size=4 \
  --steps=10000
```

권장 사용 방식:

- tactile map이 단순하고 데이터가 많지 않으면 `--policy.tactile_encoder_type=cnn`부터 시작합니다.
- taxel 수가 많고 공간 패턴이 중요하면 `--policy.tactile_encoder_type=attention`도 비교합니다.
- `--policy.tactile_n_tokens=1`로 먼저 smoke test를 돌리고, 성능 비교용으로 `2` 또는 `4`를 실험합니다.
- Pi0/Pi0.5는 큰 모델이므로 24GB GPU에서는 `batch_size=4` 또는 `8`부터 시작하는 것이 안전합니다.
- pretrained checkpoint 없이 처음부터 학습하면 비용이 커지고 결과가 불안정할 수 있습니다.

### 학습 전에 꼭 확인할 것

짧은 smoke test를 먼저 돌리는 것을 권장합니다.

1. recording 1 episode만 실행합니다.
2. 저장된 dataset metadata에 `observation.tactile.primary`가 있는지 확인합니다.
3. shape가 `(64, 64)`인지 확인합니다.
4. tactile 값이 전부 0인지 확인합니다. 전부 0이면 serial read 또는 taxel decode가 실패한 것입니다.
5. `lerobot-train --steps=10 --batch_size=2`로 import/model forward만 먼저 확인합니다.
6. 그 다음 긴 training을 돌립니다.

### 다른 tactile 센서를 쓰려면

다른 촉각 센서를 쓰더라도 LeRobot 전체를 다시 고칠 필요는 없습니다. 아래 계약만 지키면 policy 쪽은 대부분 재사용할 수 있습니다.

- 최종 observation key는 `observation.tactile.<name>` 형식으로 둡니다.
- 값은 `float32` tactile map으로 둡니다.
- dataset feature type은 `FeatureType.TACTILE`이어야 합니다.
- policy config의 tactile input shape와 dataset shape가 같아야 합니다.

즉, 다른 센서를 붙일 때 주로 바꿀 곳은 reader/renderer입니다.

- serial protocol이 다르면 `src/lerobot_robot_paxini/paxini_reader.py`를 새 센서용 reader로 교체합니다.
- taxel 배치가 다르면 `src/lerobot_robot_paxini/tactile_render.py`의 mapping logic을 바꿉니다.
- observation key와 `FeatureType.TACTILE` 구조는 그대로 유지하는 것이 좋습니다.

### 이 repo를 만든 이유

VTLA 실험에서는 시각 정보만으로는 접촉 여부, 미끄러짐, 압력 변화, 삽입/정렬 같은 정보를 놓치기 쉽습니다.
그래서 이 fork는 LeRobot의 기존 vision-language-action pipeline에 tactile signal을 정식 observation으로 넣는 것을 목표로 합니다.

현재 상태는 Paxini/SO-101 tactile 실험을 시작할 수 있는 baseline입니다. 실험을 안정화하려면 실제 센서 frame format, taxel 좌표 파일, normalization scale, dataset 품질 확인을 프로젝트 환경에 맞게 조정해야 합니다.

<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, and Vision-Language-Action (VLA) models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md) |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                             |
| **VLAs Models**            | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx)            |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
