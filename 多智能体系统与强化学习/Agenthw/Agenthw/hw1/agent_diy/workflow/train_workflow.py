#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
import time
import copy
import numpy as np
from kaiwu_agent.utils.common_func import create_cls
import os
from agent_diy.conf.conf import Config
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None, done=False)


def sample_process(list_game_data, done=False, **kwargs):
    processed = []
    for i in list_game_data:
        payload = dict(i.__dict__)
        payload["done"] = done
        processed.append(SampleData(**payload))
    return processed


def reward_shaping(frame_no, delta_score, terminated, truncated, obs, _obs):
    reward = -0.02
    if len(obs) < 1 or len(_obs) < 1:
        return -1.0

    end_dist, next_end_dist = float(obs[0]), float(_obs[0])
    if next_end_dist < end_dist:
        reward += 0.08
    elif next_end_dist > end_dist:
        reward -= 0.05

    treasure_dists, next_treasure_dists = obs[1:11], _obs[1:11]
    valid_idx = [i for i, d in enumerate(treasure_dists) if d < 999]
    if valid_idx:
        nearest_idx = min(valid_idx, key=lambda i: treasure_dists[i])
        if next_treasure_dists[nearest_idx] < treasure_dists[nearest_idx]:
            reward += 0.02

    reward += max(-1.0, min(1.0, delta_score / 150.0))
    if terminated:
        reward += 8.0
    if truncated and not terminated:
        reward -= 8.0

    return float(max(-10.0, min(10.0, reward)))


def _same_state_feature(a, b):
    # Robust compare for both scalar state-id and vector state feature.
    # 兼容标量状态与向量状态的比较
    a_is_arr = isinstance(a, np.ndarray)
    b_is_arr = isinstance(b, np.ndarray)
    if a_is_arr or b_is_arr:
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    EPISODES = getattr(Config, "EPISODES", 10000)
    epsilon_start = getattr(Config, "EPSILON_START", 1.0)
    epsilon_min = getattr(Config, "EPSILON_MIN", 0.05)
    epsilon_decay = getattr(Config, "EPSILON_DECAY", 0.9995)
    no_progress_patience = getattr(Config, "NO_PROGRESS_PATIENCE", 18)
    force_random_steps_cfg = getattr(Config, "FORCE_RANDOM_STEPS", 4)

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    logger.info("Start Training ...")
    start_t = time.time()
    last_auto_save_time = start_t

    total_rew, win_cnt = (
        0,
        0,
    )

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    # Curriculum: learn reliable path first, then optimize treasure collection.
    # 课程学习：先学稳定通关，再学拿宝箱
    warmup_conf = copy.deepcopy(usr_conf)
    conf_key = "env_conf" if "env_conf" in warmup_conf else "diy"
    if conf_key in warmup_conf:
        warmup_conf[conf_key]["treasure_random"] = False
        warmup_conf[conf_key]["treasure_id"] = []
    warmup_episodes = int(EPISODES * 0.2)

    success_window = []
    success_window_size = 200

    opposite_action = {0: 1, 1: 0, 2: 3, 3: 2}

    for episode in range(EPISODES):
        # Retrieving training metrics
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        # Reset the environment and obtain the initial state
        # 重置环境, 并获取初始状态
        current_conf = warmup_conf if episode < warmup_episodes else usr_conf
        obs, state = env.reset(usr_conf=current_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # First frame processing
        # 首帧处理
        obs_data = agent.observation_process(obs, state)

        # Task loop
        # 任务循环
        done = False
        # Episode-based epsilon schedule is more stable than per-step reset/decay.
        # 按episode衰减epsilon，更稳定
        agent.epsilon = max(epsilon_min, epsilon_start * (epsilon_decay**episode))
        if hasattr(agent, "algorithm") and hasattr(agent.algorithm, "set_episode"):
            agent.algorithm.set_episode(episode, EPISODES)
        prev_score = 0
        last_state_feature = None
        last_action = None
        no_progress_steps = 0
        force_random_steps = 0
        terminated, truncated = False, False
        while not done:
            # Agent performs inference to obtain the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            if force_random_steps > 0:
                act = int(np.random.randint(0, agent.action_size))
                force_random_steps -= 1
            else:
                act_data, model_version = agent.predict(list_obs_data=[obs_data])
                act_data = act_data[0]
                # Unpacking ActData into actions
                # ActData 解包成动作
                act = agent.action_process(act_data)

            # Interact with the environment, perform actions, and obtain the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, state = env.step(act)
            if _obs is None:
                break
            delta_score = score - prev_score
            prev_score = score

            # Feature processing
            # 特征处理
            _obs_data = agent.observation_process(_obs, state)

            # Compute reward
            # 计算 reward
            reward = reward_shaping(frame_no, delta_score, terminated, truncated, obs, _obs)

            # Loop suppression:
            # 1) no state change => likely invalid move/collision
            # 2) s(t-1) -> s(t) -> s(t-1) => immediate oscillation
            # 3) opposite action with no gain => discourage ping-pong
            # 抑制循环：原地、两步往返、无收益反向动作
            curr_state_feature = obs_data.feature
            next_state_feature = _obs_data.feature
            if _same_state_feature(next_state_feature, curr_state_feature):
                reward -= 0.15
            if last_state_feature is not None and _same_state_feature(next_state_feature, last_state_feature):
                reward -= 0.12
            if (
                last_action is not None
                and opposite_action.get(last_action) == act
                and delta_score <= 0
                and _obs[0] >= obs[0]
            ):
                reward -= 0.08

            # Anti-stagnation trigger:
            # if many consecutive steps make no progress, force short random burst.
            # 反停滞触发：连续无进展时，短暂强制随机探索跳出局部环
            if delta_score <= 0 and _obs[0] >= obs[0]:
                no_progress_steps += 1
            else:
                no_progress_steps = 0
            if no_progress_steps >= no_progress_patience:
                force_random_steps = force_random_steps_cfg
                no_progress_steps = 0
                reward -= 0.05

            # Determine over and update the win count
            # 判断结束, 并更新胜利次数
            done = terminated or truncated
            if terminated and not truncated:
                win_cnt += 1

            # Updating data and generating frames for training
            # 数据更新, 生成训练需要的 frame
            sample = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=_obs_data.feature,
            )

            # Sample processing
            # 样本处理
            sample = sample_process([sample])

            # train
            # 训练
            agent.learn(sample)

            # Update total reward and state
            # 更新总奖励和状态
            total_rew += reward
            last_state_feature = curr_state_feature
            last_action = act
            obs = _obs
            obs_data = _obs_data

        # Reporting training progress
        # 上报训练进度
        now = time.time()
        if now - last_auto_save_time >= 180:
            agent.save_model()
            logger.info(f"Auto-saved model at Episode: {episode + 1}")
            last_auto_save_time = now

        if now - last_report_monitor_time > 60:
            logger.info(f"Episode: {episode + 1}, Reward: {total_rew}")
            logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
            alpha = getattr(getattr(agent, "algorithm", None), "alpha", None)
            if alpha is not None:
                logger.info(f"epsilon={agent.epsilon:.4f}, alpha={alpha:.4f}")
            else:
                logger.info(f"epsilon={agent.epsilon:.4f}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})

            total_rew = 0
            last_report_monitor_time = now

        # The model has converged, training is complete, and reporting monitoring metric
        # 模型收敛, 结束训练, 上报监控指标
        episode_success = 1 if terminated and not truncated else 0
        success_window.append(episode_success)
        if len(success_window) > success_window_size:
            success_window.pop(0)
        rolling_success = sum(success_window) / max(1, len(success_window))

        # Converged criterion: recent success stays high after curriculum phase.
        # 收敛判断：课程学习后，近期成功率足够高
        if episode % 50 == 0 and episode >= warmup_episodes:
            logger.info(
                f"Rolling Success@{len(success_window)}: {rolling_success:.3f}, "
                f"anti_stagnation={no_progress_patience}/{force_random_steps_cfg}"
            )

        if rolling_success > 0.97 and episode > max(1000, warmup_episodes):
            logger.info(f"Training Converged at Episode: {episode + 1}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            break

    end_t = time.time()
    logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    agent.episodes = episode + 1

    # model saving
    # 保存模型
    agent.save_model()

    return
