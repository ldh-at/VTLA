## VTLA LeRobot Tactile Fork

This repository is a VTLA-focused fork of Hugging Face LeRobot. It keeps the original LeRobot
data collection, training, policy, and robot abstractions, while adding tactile-sensor support
for SO-101/Paxini-style experiments.

### What Changed From Original LeRobot

Original LeRobot sensor flow:

1. A robot class implements `get_observation()`.
2. Standard SO-101 followers return motor state as `observation.state`.
3. Cameras return image/video features as `observation.images.*`.
4. Training policies consume `observation.state`, `observation.images.*`, language tokens, and `action`.

VTLA tactile flow added in this fork:

1. `PaxiniSO101Follower` extends the original SO follower, so the normal motor/camera path still runs.
2. After `super().get_observation()`, it reads the Paxini tactile sensor through `SerialPaxiniReader`.
3. The serial reader follows the vendor UART frame example:
   request header `55 AA`, response header `AA 55`, little-endian length/address fields,
   status byte at offset 13, and two's-complement LRC validation.
4. Raw taxel values are converted by `TactileHeatmapRenderer` into a normalized float32 2D tactile map.
5. The final observation includes `observation.tactile.primary`, not a fake RGB tactile image.
6. Dataset metadata marks this feature as `FeatureType.TACTILE`, so it is not merged into robot state.

### Exact Source Changes

- `src/lerobot_robot_paxini/`
  - `config_paxini_so101.py`: Paxini/SO-101 robot config, UART options, taxel map options.
  - `paxini_so101.py`: extends SO follower and appends `observation.tactile.primary`.
  - `paxini_reader.py`: mock reader plus real Paxini UART read/write protocol.
  - `tactile_render.py`: taxel coordinate map to normalized 2D tactile grid.
  - `types.py`: `PaxiniSample` container for taxels and force/torque values.
- `src/lerobot/configs/types.py`
  - Adds `FeatureType.TACTILE`.
- `src/lerobot/utils/constants.py`
  - Adds `OBS_TACTILE = "observation.tactile"`.
- `src/lerobot/utils/feature_utils.py`
  - Converts tactile hardware features into float32 dataset features.
  - Classifies `observation.tactile.*` as `FeatureType.TACTILE` instead of `FeatureType.STATE`.
- `src/lerobot/configs/policies.py`
  - Adds `tactile_features` helper for policy configs.
- `src/lerobot/policies/tactile/`
  - Adds reusable CNN/attention tactile token encoders.
- `src/lerobot/processor/tactile_processor.py`
  - Adds tactile validation and temporal filtering processor steps.
- `src/lerobot/policies/pi0/` and `src/lerobot/policies/pi05/`
  - Adds tactile encoder config fields.
  - Adds tactile prefix token support in training and inference.
  - Allows loading upstream Pi0/Pi0.5 checkpoints while initializing tactile heads from scratch.

### Training With Pi0 / Pi0.5 And Tactile

If the dataset contains `observation.tactile.primary`, Pi0 and Pi0.5 now detect it from
`config.input_features` and encode it as prefix tokens next to image/language tokens.
No dataset key rename is needed.

Example:

```bash
lerobot-train \
  --policy.type=pi0 \
  --dataset.repo_id=<your_tactile_dataset_repo_id> \
  --policy.tactile_encoder_type=cnn \
  --policy.tactile_n_tokens=1
```

For Pi0.5:

```bash
lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=<your_tactile_dataset_repo_id> \
  --policy.tactile_encoder_type=cnn \
  --policy.tactile_n_tokens=1
```

Use `--policy.tactile_encoder_type=attention` if the tactile map is dense enough to benefit from
spatial attention. The current Paxini renderer outputs a 64x64 normalized map by default.

### Main Differences From Upstream

- Adds a Paxini tactile SO-101 follower integration under `src/lerobot_robot_paxini`.
- Adds Paxini UART frame parsing based on the vendor example protocol:
  request header `55 AA`, response header `AA 55`, little-endian length/address fields,
  status byte at offset 13, and two's-complement LRC validation.
- Stores tactile observations as `observation.tactile.*` float32 2D maps instead of treating
  tactile data as RGB images or generic robot state.
- Adds `FeatureType.TACTILE` and dataset feature conversion support for tactile arrays.
- Adds reusable tactile encoder and processor modules for later ACT/Diffusion/VLA policy wiring.

This fork is intended for VTLA/VLA experiments that combine vision, robot state/action data,
language/task inputs, and tactile observations. Most upstream LeRobot workflows still apply,
but tactile-aware policies must explicitly consume the new `observation.tactile.*` features.

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
