#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Algorithm:
    """
    Lightweight compatibility layer for planner-first agent.

    The framework still calls `learn`, `save_model`, and `load_model` paths
    that were originally designed for Q-learning. We keep those interfaces
    available but only track planner metadata and runtime counters.
    """

    def __init__(self, gamma, learning_rate, state_size, action_size):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        self.end_state = None
        self.end_distance = {}
        self.learn_steps = 0
        self.last_reward = 0.0
        self.last_done = False

    def set_planner_data(self, end_state, end_distance, action_size=None):
        self.end_state = None if end_state is None else int(end_state)
        self.end_distance = {int(k): int(v) for k, v in (end_distance or {}).items()}
        if action_size is not None:
            self.action_size = int(action_size)

    def learn(self, list_sample_data):
        if not list_sample_data:
            return
        sample = list_sample_data[0]
        self.learn_steps += 1
        self.last_reward = float(getattr(sample, "reward", 0.0))
        self.last_done = bool(getattr(sample, "done", False))

    def export_payload(self):
        return {
            "mode": "planner_first",
            "end_state": self.end_state,
            "end_distance": self.end_distance,
            "action_size": int(self.action_size),
            "learn_steps": int(self.learn_steps),
            "last_reward": float(self.last_reward),
            "last_done": bool(self.last_done),
        }

    def import_payload(self, payload):
        payload = payload or {}
        self.end_state = payload.get("end_state", self.end_state)
        end_distance = payload.get("end_distance", {})
        self.end_distance = {int(k): int(v) for k, v in end_distance.items()}
        self.action_size = int(payload.get("action_size", self.action_size))
        self.learn_steps = int(payload.get("learn_steps", 0))
        self.last_reward = float(payload.get("last_reward", 0.0))
        self.last_done = bool(payload.get("last_done", False))
