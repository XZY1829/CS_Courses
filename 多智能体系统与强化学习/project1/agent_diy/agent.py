#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import json
import os
from collections import deque

import numpy as np
from kaiwu_agent.agent.base_agent import (
    exploit_wrapper,
    learn_wrapper,
    load_model_wrapper,
    predict_wrapper,
    save_model_wrapper,
)
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.utils.common_func import attached, create_cls

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_act=None)
ActData = create_cls("ActData", act=None)

MAP_SIZE = 64
DEFAULT_VIEW_RADIUS = 2
MAX_RECENT_STATES = 12
MAX_TABU_EDGES = 8
TREASURE_TILE = 4


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_start = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.episodes = Config.EPISODES

        self.algorithm = Algorithm(self.gamma, self.learning_rate, self.state_size, self.action_size)

        self.end_state = None
        self.graph = {}
        self.reverse_graph = {}
        self.end_distance = {}
        self.end_parent_action = {}
        self._runtime_obs = {}
        self._goal_state = None
        self._goal_kind = "end"
        self._recent_states = []
        self._recent_actions = []
        self._tabu_edges = deque(maxlen=MAX_TABU_EDGES)
        self._last_treasure_signature = ()

        super().__init__(agent_type, device, logger, monitor)

    def initialize_planner(self, env_conf=None):
        self.end_state = self._end_state_from_conf(env_conf)
        self.graph = self._load_map_graph()
        self.reverse_graph = self._build_reverse_graph(self.graph)
        self.end_distance, self.end_parent_action = self._single_target_bfs(self.end_state)
        self.algorithm.set_planner_data(
            end_state=self.end_state,
            end_distance=self.end_distance,
            action_size=self.action_size,
        )

    @predict_wrapper
    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        action = self._plan_action(obs_data.feature, obs_data.legal_act)
        return [ActData(act=action)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        obs_data = list_obs_data[0]
        action = self._plan_action(obs_data.feature, obs_data.legal_act)
        return [ActData(act=action)]

    def _plan_action(self, state, legal_act):
        legal = legal_act if legal_act else self._legal_actions_for_state(state)
        if not legal:
            return 0

        runtime = self._runtime_obs
        visible_treasures = runtime.get("visible_treasures", [])
        treasure_signature = tuple(sorted(visible_treasures))
        if treasure_signature != self._last_treasure_signature:
            self._goal_state = None
            self._goal_kind = "end"
            self._last_treasure_signature = treasure_signature

        goal_state = self._choose_goal_state(state, visible_treasures)
        action = self._choose_action_towards_goal(state, goal_state, legal)
        action = self._break_loops_if_needed(state, goal_state, legal, action)
        self._record_transition(state, action)
        return action

    def _choose_goal_state(self, state, visible_treasures):
        if self._goal_kind == "treasure" and self._goal_state == state:
            self._goal_state = None
            self._goal_kind = "end"

        if self._goal_kind == "treasure" and self._goal_state in visible_treasures:
            return self._goal_state

        selected_treasure = self._pick_worthwhile_treasure(state, visible_treasures)
        if selected_treasure is not None:
            self._goal_kind = "treasure"
            self._goal_state = selected_treasure
            return selected_treasure

        self._goal_kind = "end"
        self._goal_state = self.end_state
        return self.end_state

    def _pick_worthwhile_treasure(self, state, visible_treasures):
        if not visible_treasures:
            return None

        best_target = None
        best_cost = None
        base_cost = self.end_distance.get(state, float("inf"))
        for treasure_state in visible_treasures:
            if treasure_state == state:
                return treasure_state

            treasure_distance, _ = self._single_target_bfs(treasure_state)
            to_treasure = treasure_distance.get(state, float("inf"))
            to_end = self.end_distance.get(treasure_state, float("inf"))
            if np.isinf(to_treasure) or np.isinf(to_end):
                continue

            detour = (to_treasure + to_end) - base_cost
            if to_treasure <= 4 and detour <= 6:
                total_cost = to_treasure + detour
                if best_cost is None or total_cost < best_cost:
                    best_target = treasure_state
                    best_cost = total_cost

        return best_target

    def _choose_action_towards_goal(self, state, goal_state, legal):
        if goal_state == self.end_state and state in self.end_parent_action:
            preferred = self.end_parent_action[state]
            if preferred in legal and not self._is_tabu_edge(state, preferred):
                return preferred

        goal_distance, goal_parent_action = self._single_target_bfs(goal_state)
        preferred = goal_parent_action.get(state)
        if preferred in legal and not self._is_tabu_edge(state, preferred):
            return preferred

        ranked_actions = []
        for action in legal:
            if self._is_tabu_edge(state, action):
                continue
            next_state = self.graph[state][action]
            ranked_actions.append((goal_distance.get(next_state, float("inf")), action))

        if ranked_actions:
            ranked_actions.sort(key=lambda item: (item[0], item[1]))
            return int(ranked_actions[0][1])

        fallback = sorted(legal, key=lambda action: (self.end_distance.get(self.graph[state][action], float("inf")), action))
        return int(fallback[0])

    def _break_loops_if_needed(self, state, goal_state, legal, action):
        if not self._detect_loop(state):
            return int(action)

        self._tabu_edges.append((state, action))
        alternatives = []
        goal_distance, _ = self._single_target_bfs(goal_state)
        last_action = self._recent_actions[-1] if self._recent_actions else None
        for candidate in legal:
            if candidate == action or self._is_tabu_edge(state, candidate):
                continue
            next_state = self.graph[state][candidate]
            distance = goal_distance.get(next_state, float("inf"))
            revisit_penalty = self._recent_states.count(next_state)
            action_penalty = 1 if candidate == last_action else 0
            alternatives.append((distance, revisit_penalty, action_penalty, candidate))

        if alternatives:
            alternatives.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            return int(alternatives[0][3])

        return int(action)

    def _detect_loop(self, state):
        if len(self._recent_states) >= 3 and self._recent_states[-1] == state and self._recent_states[-3] == state:
            return True
        if len(self._recent_states) >= 4:
            recent = self._recent_states[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3]:
                return True
        if self._recent_states.count(state) >= 3:
            return True
        return False

    def _record_transition(self, state, action):
        self._recent_states.append(int(state))
        self._recent_actions.append(int(action))
        if len(self._recent_states) > MAX_RECENT_STATES:
            self._recent_states = self._recent_states[-MAX_RECENT_STATES:]
        if len(self._recent_actions) > MAX_RECENT_STATES:
            self._recent_actions = self._recent_actions[-MAX_RECENT_STATES:]

    def _is_tabu_edge(self, state, action):
        return (state, action) in self._tabu_edges

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def reset_episode_state(self):
        self._runtime_obs = {}
        self._goal_state = None
        self._goal_kind = "end"
        self._recent_states = []
        self._recent_actions = []
        self._tabu_edges.clear()
        self._last_treasure_signature = ()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def observation_process(self, raw_obs, game_info):
        pos_x = int(game_info.pos_x)
        pos_z = int(game_info.pos_z)
        state = pos_x * MAP_SIZE + pos_z
        legal_act = self._legal_actions_for_state(state)
        local_view = [game_info.local_view[i : i + 5] for i in range(0, len(game_info.local_view), 5)]
        visible_treasures = self._visible_treasure_states(pos_x, pos_z, local_view, getattr(game_info, "view", DEFAULT_VIEW_RADIUS))
        treasure_status = tuple(int(x) for x in getattr(game_info, "treasure_status", []))

        self._runtime_obs = {
            "pos": (pos_x, pos_z),
            "state": state,
            "raw_obs": np.asarray(raw_obs, dtype=np.float32).reshape(-1),
            "local_view": local_view,
            "visible_treasures": visible_treasures,
            "treasure_status": treasure_status,
        }

        return ObsData(feature=int(state), legal_act=legal_act)

    def _visible_treasure_states(self, pos_x, pos_z, local_view, view_radius):
        visible = []
        for row_idx, row in enumerate(local_view):
            for col_idx, tile in enumerate(row):
                if int(tile) != TREASURE_TILE:
                    continue
                global_x = pos_x + row_idx - view_radius
                global_z = pos_z + col_idx - view_radius
                if 0 <= global_x < MAP_SIZE and 0 <= global_z < MAP_SIZE:
                    visible.append(global_x * MAP_SIZE + global_z)
        return sorted(set(visible))

    def _legal_actions_for_state(self, state):
        transitions = self.graph.get(int(state), {})
        legal = [action for action, next_state in transitions.items() if next_state != state]
        return legal if legal else list(transitions.keys())

    def _load_map_graph(self):
        map_file_path = os.path.join(os.path.dirname(__file__), "..", "conf", "map_data", "F_level_1.json")
        map_file_path = os.path.abspath(map_file_path)
        with open(map_file_path, "r", encoding="utf-8") as file:
            raw_graph = json.load(file)

        graph = {}
        for state_str, action_map in raw_graph.items():
            state = int(state_str)
            graph[state] = {}
            for action_str, transition in action_map.items():
                graph[state][int(action_str)] = int(transition[0])
        return graph

    def _end_state_from_conf(self, env_conf):
        if env_conf is None:
            return 11 * MAP_SIZE + 55
        try:
            target = env_conf.get("env_conf", env_conf).get("end", [11, 55])
            return int(target[0]) * MAP_SIZE + int(target[1])
        except Exception:
            return 11 * MAP_SIZE + 55

    def _build_reverse_graph(self, graph):
        reverse_graph = {}
        for state, transitions in graph.items():
            reverse_graph.setdefault(state, [])
            for action, next_state in transitions.items():
                reverse_graph.setdefault(next_state, []).append((state, action))
        return reverse_graph

    def _single_target_bfs(self, target_state):
        distance = {int(target_state): 0}
        parent_action = {}
        queue = deque([int(target_state)])
        while queue:
            current = queue.popleft()
            for prev_state, action in self.reverse_graph.get(current, []):
                if prev_state in distance:
                    continue
                distance[prev_state] = distance[current] + 1
                parent_action[prev_state] = action
                queue.append(prev_state)
        return distance, parent_action

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        planner_payload = self.algorithm.export_payload()
        np.save(model_file_path, planner_payload, allow_pickle=True)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            planner_payload = np.load(model_file_path, allow_pickle=True).item()
            self.algorithm.import_payload(planner_payload)
            self.end_state = int(planner_payload.get("end_state", self.end_state if self.end_state is not None else 0))
            self.end_distance = {int(k): int(v) for k, v in planner_payload.get("end_distance", {}).items()}
            if self.graph and self.end_state is not None:
                self.end_distance, self.end_parent_action = self._single_target_bfs(self.end_state)
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"File {model_file_path} not found")
            exit(1)
