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
"""Reusable tactile encoders.

These modules consume a normalized tactile map and produce token embeddings that
can be attached to policies such as ACT, Diffusion, or VLA-style token models.
Expected input shapes are (H, W), (B, H, W), (C, H, W), or (B, C, H, W).
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _parse_input_shape(input_shape: tuple[int, ...]) -> tuple[int, tuple[int, int]]:
    if len(input_shape) == 2:
        return 1, (input_shape[0], input_shape[1])
    if len(input_shape) == 3:
        return input_shape[0], (input_shape[1], input_shape[2])
    raise ValueError(f"Expected tactile input_shape (H, W) or (C, H, W), got {input_shape}.")


def _as_bchw(x: Tensor, input_channels: int) -> Tensor:
    if x.dim() == 2 and input_channels == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3 and input_channels == 1:
        return x.unsqueeze(1)
    if x.dim() == 3 and x.shape[0] == input_channels:
        return x.unsqueeze(0)
    if x.dim() == 4 and x.shape[1] == input_channels:
        return x
    raise ValueError(
        "Expected tactile tensor shape (H, W), (B, H, W), (C, H, W), or (B, C, H, W), "
        f"got {tuple(x.shape)}."
    )


class TactileCNN(nn.Module):
    """CNN tactile backbone that outputs one feature vector per sample."""

    def __init__(self, input_shape: tuple[int, ...] = (64, 64), feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.input_channels, spatial_shape = _parse_input_shape(input_shape)

        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        feature_h = spatial_shape[0] // 8
        feature_w = spatial_shape[1] // 8
        if feature_h <= 0 or feature_w <= 0:
            raise ValueError(f"Tactile input_shape must be at least 8x8, got {input_shape}.")
        conv_output_dim = 128 * feature_h * feature_w

        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = _as_bchw(x, self.input_channels).float()
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TactileAttentionCNN(nn.Module):
    """CNN tactile backbone with spatial attention."""

    def __init__(self, input_shape: tuple[int, ...] = (64, 64), feature_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.input_channels, _ = _parse_input_shape(input_shape)

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = _as_bchw(x, self.input_channels).float()
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x * self.attention(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(x.flatten(1))


class TactileTokenEncoder(nn.Module):
    """Wrap a tactile CNN backbone and output policy token embeddings.

    Forward input:
        (H, W), (B, H, W), (C, H, W), or (B, C, H, W)

    Forward output:
        (B, n_tokens, feature_dim)
    """

    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, ...],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "cnn":
            self.backbone = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}. Choose 'cnn' or 'attention'.")

        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        if self.n_tokens == 1:
            return feat.unsqueeze(1)

        feat = self.token_proj(feat)
        return feat.view(feat.size(0), self.n_tokens, self.feature_dim)
