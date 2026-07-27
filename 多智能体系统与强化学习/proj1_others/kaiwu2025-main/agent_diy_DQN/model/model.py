#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from kaiwu_agent.utils.common_func import attached


class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(DuelingDQN, self).__init__()
        # 假设state_shape为一维特征长度
        self.feature = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(),
        )
        # Value分支
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling DQN输出Q值
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals
