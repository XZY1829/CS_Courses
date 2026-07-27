#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None, done=False)


@attached
def sample_process(list_game_data, done=False, **kwargs):
    processed = []
    for i in list_game_data:
        payload = dict(i.__dict__)
        payload["done"] = done
        processed.append(SampleData(**payload))
    return processed


def reward_shaping(frame_no, delta_score, terminated, truncated, obs, _obs):
    # obs/_obs are raw distance vectors:
    # [end_dist, treasure_dist_0, ..., treasure_dist_9]
    if len(obs) < 1 or len(_obs) < 1:
        return -1.0

    reward = -0.02
    end_dist, next_end_dist = float(obs[0]), float(_obs[0])
    if next_end_dist < end_dist:
        reward += 0.08
    elif next_end_dist > end_dist:
        reward -= 0.05

    reward += max(-1.0, min(1.0, float(delta_score) / 150.0))

    if terminated:
        reward += 8.0
    if truncated and not terminated:
        reward -= 8.0

    return float(max(-10.0, min(10.0, reward)))
