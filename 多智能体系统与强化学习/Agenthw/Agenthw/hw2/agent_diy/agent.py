#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from agent_diy.feature.definition import ObsData
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached


from agent_diy.algorithm.algorithm import Algorithm
from arena_proto.back_to_the_realm.custom_pb2 import (
    RelativeDirection,
)


def one_hot_encoding(grid_pos):
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1

    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        direction[rel_pos.direction - 1] = 1

    grid_distance = 1 if rel_pos.grid_distance < 0 else rel_pos.grid_distance / (128 * 128)
    feature = direction + [grid_distance]
    return feature


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.algorithm = Algorithm(device, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        # Sync target network after loading
        self.algorithm.target_model.load_state_dict(self.algorithm.model.state_dict())
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data):
        result = act_data.move_dir
        result += act_data.use_talent * 8
        return result

    def observation_process(self, raw_obs, preprocessor, state_env_info=None):
        feature, legal_act = [], []

        (
            norm_pos,
            grid_pos,
            start_pos,
            end_pos,
            buff_pos,
            treasure_pos_list,
            obstacle_map,
            memory_map,
            treasure_map,
            end_map,
            recent_position_map,
            treasure_collected_count,
            treasure_count,
        ) = preprocessor.process(raw_obs)

        one_hot_pos = one_hot_encoding(grid_pos)
        norm_pos = [norm_pos.x, norm_pos.z]
        end_pos_features = read_relative_position(end_pos)

        treasure_pos_features = []
        for treasure_pos in treasure_pos_list:
            treasure_pos_features = treasure_pos_features + list(read_relative_position(treasure_pos))

        buff_availability = 0
        if raw_obs:
            for organ in raw_obs.frame_state.organs:
                if organ.sub_type == 2:
                    buff_availability = organ.status

        talent_availability = 0
        if raw_obs:
            talent_availability = raw_obs.frame_state.heroes[0].talent.status

        # BUG FIX: baseline uses argmin on all distances including -1 (null),
        # which picks null treasures. Filter for valid (> 0) distances only.
        treasure_dists = [pos.grid_distance for pos in treasure_pos_list]
        valid_treasures = [(i, d) for i, d in enumerate(treasure_dists) if d > 0]
        if valid_treasures:
            nearest_idx = min(valid_treasures, key=lambda x: x[1])[0]
            end_pos_features = read_relative_position(treasure_pos_list[nearest_idx])

        feature_vec = (
            norm_pos + one_hot_pos + end_pos_features + treasure_pos_features + [buff_availability, talent_availability]
        )
        feature_map = obstacle_map + end_map + treasure_map + memory_map
        legal_act = list(raw_obs.legal_act)

        remain_info = {
            "memory_map": memory_map,
            "end_pos": end_pos,
            "buff_pos": buff_pos,
            "treasure_pos": treasure_pos_list,
            "recent_position_map": recent_position_map,
            "treasure_collected_count": treasure_collected_count,
            "treasure_count": treasure_count,
        }

        return ObsData(feature=feature_vec + feature_map, legal_act=legal_act), remain_info
