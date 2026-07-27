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
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from model.model import DuelingDQN


class Algorithm:
    def __init__(self, gamma, learning_rate, state_size, action_size, device='cpu',
                 buffer_size=10000, batch_size=64, target_update=100):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step = 0
        # 经验回放
        self.memory = deque(maxlen=buffer_size)
        # 主网络和目标网络
        self.policy_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32, device=self.device),
            torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32, device=self.device),
            torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    def learn(self, list_sample_data):
        # 存储样本
        for sample in list_sample_data:
            self.store(sample.state, sample.action, sample.reward, sample.next_state, sample.done)
        if len(self.memory) < self.batch_size:
            return  # 样本不足不学习
        state, action, reward, next_state, done = self.sample()
        # Q(s,a)
        q_values = self.policy_net(state).gather(1, action)
        # Q(s',a') for next state
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1, keepdim=True)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # target网络软更新
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
