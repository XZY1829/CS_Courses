#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
DIY reward shaping: based on target_dqn baseline.
Improvements: timeout penalty, increased exploration reward.
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_diy.conf.conf import Config


def bump(a1, b1, a2, b2):
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False

    dist = ((a1 - a2) ** 2 + (b1 - b2) ** 2) ** (0.5)

    return dist <= 500


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def convert_pos_to_grid_pos(x, z):
    x = (x + 2250) // 500
    z = (z + 5250) // 500
    x, z = z, x
    return x, z


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    remain_info,
    _remain_info,
    obs_data,
    _obs_data,
):
    reward = 0

    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    end_dist = _remain_info.get("end_pos").l2_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")

    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    prev_end_dist = remain_info.get("end_pos").l2_distance
    prev_treasure_dists = [
        pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")
    ]
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")

    is_treasures_remain = treasure_collected_count < treasure_count

    """
    Reward 1. Reward related to the end point
    """
    reward_end_dist = 0
    if prev_end_dist > 0 and treasure_collected_count > 0:
        reward_end_dist = 1 if end_dist < prev_end_dist else -1

    reward_win = 0
    if terminated:
        reward_win = treasure_collected_count

    """
    Reward 2. Rewards related to the treasure chest
    """
    reward_treasure_dist = 0
    if is_treasures_remain:
        visible_dists = [d for d in treasure_dists if d > 0]
        if visible_dists:
            min_dist = min(visible_dists)
            min_index = treasure_dists.index(min_dist)
            prev_min_dist = prev_treasure_dists[min_index]
            if min_dist < prev_min_dist or prev_min_dist < 0:
                reward_treasure_dist = 1
            else:
                reward_treasure_dist = -1

    reward_treasure = 0
    if treasure_collected_count > prev_treasure_collected_count:
        reward_treasure = 1

    """
    Reward 3. Rewards related to the buff
    """
    reward_buff_dist = 0
    reward_buff = 0

    """
    Reward 4. Rewards related to the flicker
    """
    reward_flicker = 0

    """
    Reward 5. Rewards for quick clearance
    """
    reward_step = 1

    reward_memory = 0
    memory_map = remain_info.get("memory_map")
    reward_memory = memory_map[len(memory_map) // 2]

    reward_bump = 0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    if is_bump:
        reward_bump = 1

    reward_exploration = 0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.5 * pass_times, -10)

    # DIY: timeout penalty to teach the agent that running out of time is bad
    reward_timeout = 0
    if truncated and not terminated:
        reward_timeout = 1

    reward_weight = {
        "reward_end_dist": 0.5,
        "reward_win": 0.5,
        "reward_buff_dist": 0,
        "reward_buff": 0,
        "reward_treasure_dists": 0.5,
        "reward_treasure": 1.0,
        "reward_flicker": 0,
        "reward_step": -0.0005,
        "reward_bump": -1.0,
        "reward_memory": -0.005,
        "reward_exploration": 0.1,
        "reward_timeout": -3.0,
    }

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_timeout * reward_weight["reward_timeout"],
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
    )


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = (2 * (Config.VIEW_SIZE) + 1) ** 2 * 4 + 404
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
