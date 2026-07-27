"""1D ResNet baseline classifier for PTB-XL superdiagnostic classes.

A compact 1D residual network (adapted from the standard ECG ResNet1d used
in the PTB-XL benchmark, Wagner et al. 2020). CPU-friendly: ~0.5M params,
trains in minutes on 4 cores for the 5-superclass task.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, ch, k=7, drop=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv1 = nn.Conv1d(ch, ch, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=k // 2, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.conv1(torch.relu(self.bn1(x)))
        h = self.drop(h)
        h = self.conv2(torch.relu(self.bn2(h)))
        return x + h


class ECGResNet1d(nn.Module):
    """1D ResNet for variable lead count.

    n_leads: input channels. Set to 12 for full clinical, 1 for single-lead.
    Pooling to a fixed feature length handles variable input durations.
    """

    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 base_ch: int = 32, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.blocks = nn.Sequential(*[ResBlock1d(base_ch) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_ch, n_classes),
        )

    def forward(self, x):
        # x: (B, n_leads, T)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
