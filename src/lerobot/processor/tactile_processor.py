#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tactile observation processor steps."""

from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_TACTILE


def _is_tactile_key(key: str) -> bool:
    return key == OBS_TACTILE or key.startswith(OBS_TACTILE + ".")


@ProcessorStepRegistry.register(name="tactile_validation")
class TactileValidationProcessorStep(ObservationProcessorStep):
    """Validate tactile observation shape and convert arrays to float tensors."""

    def __init__(self, expected_shape: tuple[int, int] = (64, 64)):
        self.expected_shape = tuple(expected_shape)

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for key in list(obs.keys()):
            if not _is_tactile_key(key):
                continue

            tactile_data = obs[key]
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()
            elif isinstance(tactile_data, torch.Tensor):
                tactile_data = tactile_data.float()
            else:
                raise TypeError(f"Tactile observation '{key}' must be a numpy array or torch tensor.")

            if tactile_data.dim() == 4 and tactile_data.shape[1] == 1:
                tactile_data = tactile_data.squeeze(1)
            elif tactile_data.dim() not in (2, 3):
                raise ValueError(
                    f"Tactile observation '{key}' must have shape (H, W), (B, H, W), or (C, H, W); "
                    f"got {tuple(tactile_data.shape)}."
                )

            actual_shape = tuple(tactile_data.shape[-2:])
            if actual_shape != self.expected_shape:
                raise ValueError(
                    f"Tactile observation '{key}' shape mismatch. "
                    f"Expected {self.expected_shape}, got {actual_shape}."
                )

            obs[key] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="tactile_temporal_filter")
class TactileTemporalFilterProcessorStep(ObservationProcessorStep):
    """Apply an exponential moving average filter to tactile observations."""

    def __init__(self, alpha: float = 0.2):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}.")
        self.alpha = alpha
        self._prev_tactile: dict[str, torch.Tensor] = {}

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for key in list(obs.keys()):
            if not _is_tactile_key(key):
                continue

            tactile_data = obs[key]
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()
            elif isinstance(tactile_data, torch.Tensor):
                tactile_data = tactile_data.float()
            else:
                raise TypeError(f"Tactile observation '{key}' must be a numpy array or torch tensor.")

            if key in self._prev_tactile:
                tactile_data = self.alpha * tactile_data + (1 - self.alpha) * self._prev_tactile[key]

            self._prev_tactile[key] = tactile_data.clone().detach()
            obs[key] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._prev_tactile.clear()
