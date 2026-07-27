#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None, done=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def _parse_distances(obs):
    """
    Parse raw observation distances robustly.
    Expected online env format is usually:
      [end_dist, treasure_dist_0, ..., treasure_dist_9]
    """
    if obs is None:
        return None, np.array([])

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None, np.array([])

    end_dist = float(arr[0])
    treasure_dists = arr[1:] if arr.size > 1 else np.array([], dtype=np.float32)
    return end_dist, treasure_dists


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    obs,
    _obs,
    revisit_cnt=0,
    oscillation=False,
    no_progress_steps=0,
    late_stage=False,
    treasure_cleared=False,
):
    # Planner-first mode keeps reward shaping as a compatibility utility.
    # It is no longer the primary behavior driver.
    reward = -0.02

    end_dist, _ = _parse_distances(obs)
    next_end_dist, _ = _parse_distances(_obs)
    if end_dist is not None and next_end_dist is not None:
        if next_end_dist < end_dist:
            reward += 0.6
        elif next_end_dist > end_dist:
            reward -= 0.6

    if score > 0 and not terminated:
        reward += min(float(score) / 100.0, 0.1)

    if oscillation:
        reward -= 0.2
    if revisit_cnt > 0:
        reward -= min(0.02 * revisit_cnt, 0.3)
    if no_progress_steps > 0:
        reward -= min(0.005 * no_progress_steps, 0.25)
    if truncated:
        reward -= 4.0
    if terminated:
        reward += 10.0

    return float(reward)
