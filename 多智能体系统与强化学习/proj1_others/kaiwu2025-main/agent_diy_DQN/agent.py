#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import torch

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import create_cls, attached
from kaiwu_agent.agent.base_agent import BaseAgent

try:
    # Preferred when imported as package module.
    from agent_diy_DQN.conf.conf import Config
    from agent_diy_DQN.algorithm.algorithm import Algorithm
except ModuleNotFoundError:
    # Fallback for legacy flat PYTHONPATH layouts.
    from conf.conf import Config
    from algorithm.algorithm import Algorithm


ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger
        self.state_size = Config.OBSERVATION_SHAPE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = getattr(Config, 'BATCH_SIZE', 64)
        self.buffer_size = getattr(Config, 'BUFFER_SIZE', 10000)
        self.target_update = getattr(Config, 'TARGET_UPDATE', 100)
        self.algorithm = Algorithm(
            self.gamma, self.learning_rate, self.state_size, self.action_size,
            device=self.device, buffer_size=self.buffer_size, batch_size=self.batch_size, target_update=self.target_update
        )
        super().__init__(agent_type, self.device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        state = list_obs_data[0].feature
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        if torch.rand(1).item() <= self.epsilon:
            act = torch.randint(0, self.action_size, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.algorithm.policy_net(state_tensor)
                act = q_values.argmax(dim=1).item()
        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation["obs"], observation["extra_info"])
        state = obs_data.feature
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.algorithm.policy_net(state_tensor)
            act = q_values.argmax(dim=1).item()
        act_data = ActData(act=act)
        act = self.action_process(act_data)
        return act

    def _epsilon_greedy(self, state, epsilon=0.1):
        # 已合并到predict
        pass

    @learn_wrapper
    def learn(self, list_sample_data):
        # 需要保证sample有done字段
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs, extra_info):

        game_info = extra_info["game_info"]
        pos = [game_info["pos_x"], game_info["pos_z"]]
        # Feature #1: Current state of the agent (1-dimensional representation)
        # 特征#1: 智能体当前 state (1维表示)
        state = [int(pos[0] * 64 + pos[1])]
        # Feature #2: One-hot encoding of the agent's current position
        # 特征#2: 智能体当前位置信息的 one-hot 编码
        pos_row = [0] * 64
        pos_row[pos[0]] = 1
        pos_col = [0] * 64
        pos_col[pos[1]] = 1

        # Feature #3: Discretized distance of the agent's current position from the endpoint
        # 特征#3: 智能体当前位置相对于终点的距离(离散化)
        # Feature #4: Discretized distance of the agent's current position from the treasure
        # 特征#4: 智能体当前位置相对于宝箱的距离(离散化)
        end_treasure_dists = raw_obs["feature"]

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # 特征#5: 图特征生成(障碍物信息, 宝箱信息, 终点信息)
        local_view = [game_info["local_view"][i : i + 5] for i in range(0, len(game_info["local_view"]), 5)]
        obstacle_map, treasure_map, end_map = [], [], []
        for sub_list in local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])

        # Feature #6: Conversion of graph features into vector features
        # 特征#6: 图特征转换为向量特征
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        # Feature #7: Information of the map areas visited within the agent's current local view
        # 特征#7: 智能体当前局部视野中的走过的地图信息
        memory_flat = []
        view_size = game_info["view"] * 2 + 1
        memory = game_info["location_memory"]
        for i in range(view_size):
            row = pos[0] - game_info["view"] + i
            col_start = pos[1] - game_info["view"]
            row_vals = []
            for j in range(view_size):
                col = col_start + j
                if 0 <= row < 64 and 0 <= col < 64:
                    row_vals.append(memory[row * 64 + col])
                else:
                    # Pad out-of-bound cells with 0 to keep a stable local-view feature.
                    row_vals.append(0)
            memory_flat.extend(row_vals)

        tmp_treasure_status = [x if x != 2 else 0 for x in game_info["treasure_status"]]

        feature = np.concatenate(
            [
                state,
                pos_row,
                pos_col,
                end_treasure_dists,
                obstacle_flat,
                treasure_flat,
                end_flat,
                memory_flat,
                tmp_treasure_status,
            ]
        )
        feature = feature.astype(np.float32)
        if feature.shape[0] > self.state_size:
            feature = feature[: self.state_size]
        elif feature.shape[0] < self.state_size:
            feature = np.pad(feature, (0, self.state_size - feature.shape[0]), mode="constant")
        return ObsData(feature=feature)

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # 保存policy_net和target_net参数
        import os
        if path is None:
            path = '.'
        os.makedirs(path, exist_ok=True)
        policy_path = os.path.join(path, f"policy_net_{id}.pth")
        target_path = os.path.join(path, f"target_net_{id}.pth")
        torch.save(self.algorithm.policy_net.state_dict(), policy_path)
        torch.save(self.algorithm.target_net.state_dict(), target_path)
        if self.logger:
            self.logger.info(f"save model {policy_path} and {target_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # 加载policy_net和target_net参数
        import os
        if path is None:
            path = '.'
        policy_path = os.path.join(path, f"policy_net_{id}.pth")
        target_path = os.path.join(path, f"target_net_{id}.pth")
        try:
            self.algorithm.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            self.algorithm.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
            if self.logger:
                self.logger.info(f"load model {policy_path} and {target_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"File {policy_path} or {target_path} not found")
            exit(1)
