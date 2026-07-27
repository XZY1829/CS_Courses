#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    SAMPLE_DIM = 21624

    VIEW_SIZE = 25

    # CNN output (64*8*8=4096) + vector features (404) = 4500
    DIM_OF_OBSERVATION = 4096 + 404

    DIM_OF_ACTION_DIRECTION = 8
    DIM_OF_TALENT = 8

    # 2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]

    TARGET_UPDATE_FREQ = 500
    EPSILON_GREEDY_PROBABILITY = 300000
    GAMMA = 0.95
    EPSILON = 0.1
    START_LR = 1e-4

    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
