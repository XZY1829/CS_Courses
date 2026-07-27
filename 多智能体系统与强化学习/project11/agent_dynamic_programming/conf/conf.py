#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    STATE_SIZE = 64 * 64
    ACTION_SIZE = 4
    GAMMA = 0.9
    THETA = 1e-3
    EPISODES = 300
    ALGO = "value_iteration"
    STEP_PENALTY = -1.0
    TREASURE_BONUS = 1.2
    PROGRESS_BONUS = 0.25
    RETREAT_PENALTY = 0.8
    TREASURE_LEAVE_PENALTY = 1.2

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 214

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 250
