#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
from collections import deque

from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached

from agent_diy.feature.definition import reward_shaping, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf

SAVE_INTERVAL_SECONDS = 180
MIN_EPISODES_BEFORE_CONVERGENCE = 100
CONVERGENCE_WINDOW = 50
CONVERGENCE_SUCCESS_RATE = 0.95


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    monitor_data = {
        "reward": 0.0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }

    logger.info("Start planner training until convergence ...")
    start_t = time.time()
    last_save_model_time = start_t
    last_monitor_time = start_t

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    if hasattr(agent, "initialize_planner"):
        agent.initialize_planner(env_conf=usr_conf)
    if hasattr(agent, "reset_episode_state"):
        agent.reset_episode_state()

    total_reward = 0.0
    rollout_steps = 0
    success_count = 0
    episode_count = 0
    recent_success = deque(maxlen=CONVERGENCE_WINDOW)

    env_conf = usr_conf.get("env_conf", usr_conf) if isinstance(usr_conf, dict) else {}
    max_step = int(env_conf.get("max_step", 2000))

    while True:
        obs, state = env.reset(usr_conf=usr_conf)
        if obs is None:
            continue

        if hasattr(agent, "reset_episode_state"):
            agent.reset_episode_state()

        obs_data = agent.observation_process(obs, state)
        episode_count += 1
        episode_step = 0
        done = False
        episode_success = False

        while (not done) and episode_step < max_step:
            now = time.time()
            if now - last_save_model_time >= SAVE_INTERVAL_SECONDS:
                logger.info("Periodic checkpoint: saving model (every 3 minutes)")
                agent.save_model()
                last_save_model_time = now

            act_data, _ = agent.predict(list_obs_data=[obs_data])
            act = agent.action_process(act_data[0])
            frame_no, next_obs, score, terminated, truncated, next_state = env.step(act)
            if next_obs is None:
                break

            next_obs_data = agent.observation_process(next_obs, next_state)
            reward = reward_shaping(
                frame_no=frame_no,
                score=score,
                terminated=terminated,
                truncated=truncated,
                obs=obs,
                _obs=next_obs,
            )
            total_reward += float(reward)
            rollout_steps += 1
            episode_step += 1

            sample = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=next_obs_data.feature,
                done=bool(terminated or truncated),
            )
            agent.learn(sample_process([sample]))

            obs, obs_data = next_obs, next_obs_data
            done = bool(terminated or truncated)
            if terminated:
                success_count += 1
                episode_success = True

            if now - last_monitor_time >= 60:
                elapsed = now - start_t
                logger.info(
                    f"elapsed={elapsed:.1f}s episodes={episode_count} steps={rollout_steps} successes={success_count}"
                )
                last_monitor_time = now

        recent_success.append(1 if episode_success else 0)
        if (
            episode_count >= MIN_EPISODES_BEFORE_CONVERGENCE
            and len(recent_success) == CONVERGENCE_WINDOW
        ):
            recent_success_rate = sum(recent_success) / float(CONVERGENCE_WINDOW)
            if recent_success_rate >= CONVERGENCE_SUCCESS_RATE:
                logger.info(
                    f"Converged: recent {CONVERGENCE_WINDOW} episodes success_rate="
                    f"{recent_success_rate:.3f}"
                )
                break

    monitor_data["reward"] = total_reward
    monitor_data["diy_1"] = rollout_steps
    monitor_data["diy_2"] = int(success_count)
    monitor_data["diy_3"] = int(episode_count)
    if monitor:
        monitor.put_data({os.getpid(): monitor_data})

    training_metrics = get_training_metrics()
    if training_metrics:
        logger.info(f"training_metrics is {training_metrics}")

    agent.episodes = episode_count
    agent.save_model()

    end_t = time.time()
    logger.info(
        f"Planner training finished in {end_t - start_t:.2f} s, "
        f"episodes={episode_count}, steps={rollout_steps}, successes={success_count}"
    )
