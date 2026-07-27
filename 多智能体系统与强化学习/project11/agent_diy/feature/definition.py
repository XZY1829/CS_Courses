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
    # Priority: finish without timeout first, then collect treasure.
    # 优先级：先稳定通关，再尽量拿宝箱
    reward = -0.02

    # In train workflow, obs/_obs are raw env distances:
    # [end_dist, treasure_dist_0, ..., treasure_dist_9]
    # 在训练流程中，obs/_obs是环境原始距离向量，不是250维特征
    if len(obs) < 1 or len(_obs) < 1:
        return -1.0

    # Reward for moving closer to the end.
    # 靠近终点奖励
    end_dist, next_end_dist = float(obs[0]), float(_obs[0])
    if next_end_dist < end_dist:
        reward += 0.08
    elif next_end_dist > end_dist:
        reward -= 0.05

    # Mild shaping towards nearest available treasure.
    # 对最近可用宝箱给轻量引导
    treasure_dists, next_treasure_dists = obs[1:11], _obs[1:11]
    valid_idx = [i for i, d in enumerate(treasure_dists) if d < 999]
    if valid_idx:
        nearest_idx = min(valid_idx, key=lambda i: treasure_dists[i])
        if next_treasure_dists[nearest_idx] < treasure_dists[nearest_idx]:
            reward += 0.02

    # Keep score delta, but constrain its scale.
    # 保留积分增量信号并限制幅度
    reward += max(-1.0, min(1.0, delta_score / 150.0))

    # Terminal rewards are dominant to prevent timeout.
    # 终止奖励权重更高，避免超时
    if terminated:
        reward += 8.0
    if truncated and not terminated:
        reward -= 8.0

    return float(max(-10.0, min(10.0, reward)))
