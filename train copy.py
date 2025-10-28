
'''
# MADDPG（连续动作）
python train.py --algo maddpg

# QMIX（离散动作）
python train.py --algo qmix
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
# 全局忽略所有 DeprecationWarning（包括 tensorboardX 等第三方库）
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Conversion of an array with ndim > 0 to a scalar.*')

import os
import time
import argparse
import numpy as np
from datetime import datetime

import paddle
from parl.utils import logger, summary

from simple_model import MAModel
from simple_agent import MAAgent
from env import FacadeEnv
from wrapper import MultiAgentWrapper

from parl.algorithms import MADDPG
from parl.algorithms import QMIX
from mappo import MAPPO
from madqn import MADQN


# 全局超参
CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 1024
MAX_STEP = 20
EVAL_EPISODES = 3
STEP_SIZE = 0.05  # 离散/连续步长

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['maddpg', 'mappo', 'qmix', 'madqn'], default='madqn')
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--max_episodes', type=int, default=200000)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--device', type=str, default='gpu')
    return parser.parse_args()

def make_env(args):
    # 离散算法使用离散动作，否则连续
    discrete = args.algo in ['qmix', 'madqn']
    env = FacadeEnv(discrete_actions=discrete, max_steps=MAX_STEP)
    wrapper = MultiAgentWrapper(env, discrete_actions=discrete, step_size=STEP_SIZE)
    return wrapper

def select_algorithm(args, model, agent_index, act_space):
    if args.algo == 'maddpg':
        return MADDPG(model, agent_index=agent_index, act_space=act_space,
                      gamma=GAMMA, tau=TAU,
                      critic_lr=CRITIC_LR, actor_lr=ACTOR_LR)
    if args.algo == 'mappo':
        return MAPPO(model, agent_index=agent_index, act_space=act_space,
                     gamma=GAMMA, tau=TAU,
                     critic_lr=CRITIC_LR, actor_lr=ACTOR_LR)
    if args.algo == 'qmix' and QMIX:
        return QMIX(model, agent_index=agent_index, act_space=act_space,
                    gamma=GAMMA, lr=CRITIC_LR)
    if args.algo == 'madqn':
        return MADQN(model, agent_index=agent_index, act_space=act_space,
                     gamma=GAMMA, lr=ACTOR_LR)
    raise ValueError(f"Unsupported algorithm {args.algo}")

def run_episode(env, agents, args):
    obs_n = env.reset()
    done = False
    total_reward = 0.0
    agents_reward = [0.0 for _ in range(env.n)]
    steps = 0
    losses = []
    while not done and steps < MAX_STEP:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        # 存储与学习
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])
        obs_n = next_obs_n
        for i, agent in enumerate(agents):
            loss = agent.learn(agents)
            if loss is not None:
                losses.append(float(loss.numpy() if hasattr(loss, 'numpy') else loss))
        # 累计奖励
        total_reward += sum(reward_n)
        for i, r in enumerate(reward_n):
            agents_reward[i] += r
    return total_reward, agents_reward, losses, steps

def evaluate(env, agents, args):
    rewards = []
    for _ in range(EVAL_EPISODES):
        obs_n = env.reset()
        done = False
        total = 0.0
        while not done:
            act_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, r_n, done_n, _ = env.step(act_n)
            done = all(done_n)
            total += sum(r_n)
        rewards.append(total)
    return np.mean(rewards)

def main():
    args = parse_args()
    paddle.set_device(args.device)
    logger.set_dir(f"./train_log/{args.algo}")

    env = make_env(args)

    # 从环境中提取形状与动作空间
    obs_shape = env.obs_shape_n
    act_shape = env.act_shape_n
    act_space_n = [env.action_space[name] for name in env.agent_names]

    # 构建 agents
    agents = []
    for i in range(env.n):
        model = MAModel(
            obs_dim=obs_shape[i],
            act_dim=act_shape[i],
            obs_shape_n=obs_shape,
            act_shape_n=act_shape,
            continuous_actions=(args.algo in ['maddpg', 'mappo']),
            discrete_actions=(args.algo in ['qmix', 'madqn'])
        )
        alg = select_algorithm(args, model, i, act_space_n)
        agent = MAAgent(
            alg,
            agent_index=i,
            obs_dim_n=obs_shape,
            act_dim_n=act_shape,
            batch_size=BATCH_SIZE,
            discrete_actions=(args.algo in ['qmix', 'madqn']),
            step_size=STEP_SIZE
        )
        agents.append(agent)
        agent.alg.sync_target(decay=0)

    total_steps = 0
    for ep in range(1, args.max_episodes + 1):
        ep_reward, ep_agent_rewards, losses, steps = run_episode(env, agents, args)
        total_steps += steps

        # 记录 & 打印
        summary.add_scalar('train/episode_reward', float(ep_reward), ep)
        for i, r in enumerate(ep_agent_rewards):
            summary.add_scalar(f"train/agent_{i}_reward", float(r), ep)
        summary.add_scalar('train/critic_loss',
                           float(np.mean(losses)) if losses else 0.0, ep)
        logger.info(f"Episode {ep}, Total Reward: {ep_reward:.3f}, " +
                    ", ".join([f"Agent_{i}:{r:.3f}" for i, r in enumerate(ep_agent_rewards)]))

        # Eval & Save
        if ep % args.test_every == 0:
            eval_reward = evaluate(env, agents, args)
            logger.info(f"[Eval] Episode {ep}, Reward: {eval_reward:.3f}")
            summary.add_scalar('eval/episode_reward', float(eval_reward), ep)
            if not args.restore:
                os.makedirs(args.model_dir, exist_ok=True)
                for i, agent in enumerate(agents):
                    agent.save(os.path.join(args.model_dir, f"agent_{i}"))

    print("Training complete")

if __name__ == '__main__':
    main()
