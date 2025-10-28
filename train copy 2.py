
'''
# MADDPG（连续动作）
python train.py --algo maddpg

# QMIX（离散动作）
python train.py --algo qmix
'''
# train.py

import os
import time
import argparse
import numpy as np
import paddle
import warnings
from datetime import datetime

# 忽略各种 Deprecation 和 UserWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


from parl.utils import logger, summary
from simple_model import MAModel
from simple_agent import MAAgent
from env import FacadeEnv
from wrapper import MultiAgentWrapper

# 算法导入：MADDPG、QMIX 来自 parl；MAPPO、MADQN 为本地实现
from parl.algorithms import MADDPG, QMIX
from mappo import MAPPO
from madqn import MADQN

# 全局超参
CRITIC_LR             = 0.001
ACTOR_LR              = 0.0001
GAMMA                 = 0.95
TAU                   = 0.001
BATCH_SIZE            = 1024
MAX_STEP_PER_EPISODE  = 20
EVAL_EPISODES         = 3
STEP_SIZE             = 0.05  # 离散动作步长

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',
                        choices=['maddpg', 'mappo', 'qmix', 'madqn'],
                        default='madqn')
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--max_episodes', type=int, default=2000000)
    parser.add_argument('--test_every_episodes', type=int, default=1000)
    parser.add_argument('--device', type=str, default='gpu')
    return parser.parse_args()

def make_env(args):
    discrete = args.algo in ['qmix', 'madqn']
    env = FacadeEnv(discrete_actions=discrete,
                    max_steps=MAX_STEP_PER_EPISODE)
    return MultiAgentWrapper(env,
                             discrete_actions=discrete,
                             step_size=STEP_SIZE)

def select_algorithm(args, model, agent_index, act_space_n):
    if args.algo == 'maddpg':
        return MADDPG(model,
                      agent_index=agent_index,
                      act_space=act_space_n,
                      gamma=GAMMA,
                      tau=TAU,
                      critic_lr=CRITIC_LR,
                      actor_lr=ACTOR_LR)
    if args.algo == 'mappo':
        return MAPPO(model,
                     agent_index=agent_index,
                     act_space=act_space_n,
                     gamma=GAMMA,
                     tau=TAU,
                     critic_lr=CRITIC_LR,
                     actor_lr=ACTOR_LR)
    if args.algo == 'qmix':
        # QMIX 初始化只需 model, mixer, config
        config = {
            'lr': CRITIC_LR,
            'gamma': GAMMA,
            'double_q': False,
            'n_agents': len(act_space_n),
            'n_actions': act_space_n[agent_index].n,
            'obs_shape': model.obs_dim,
            'batch_size': BATCH_SIZE,
            'episode_limit': MAX_STEP_PER_EPISODE,
            'rnn_hidden_dim': 64,
            'clip_grad_norm': 10
        }
        return QMIX(model, model, config)
    if args.algo == 'madqn':
        return MADQN(model,
                     agent_index=agent_index,
                     act_space=act_space_n,
                     gamma=GAMMA,
                     lr=CRITIC_LR)
    raise ValueError(f"Unsupported algorithm {args.algo}")

def run_evaluate_episodes(env, agents, eval_episodes, args):
    eval_episode_rewards = []
    eval_episode_steps   = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            total_reward += sum(reward_n)
            if args.show:
                time.sleep(0.1)
                env.render()
        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps

def run_episode(env, agents, args):
    obs_n = env.reset()
    done = False
    total_reward  = 0.0
    agents_reward = [0.0] * env.n
    steps = 0
    losses = []
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        # 存经验
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i],
                                 action_n[i],
                                 reward_n[i],
                                 next_obs_n[i],
                                 done_n[i])
        obs_n = next_obs_n
        # 学习
        for i, agent in enumerate(agents):
            loss = agent.learn(agents)
            if loss is not None:
                losses.append(loss)
        # 累计奖励
        total_reward += sum(reward_n)
        for i, r in enumerate(reward_n):
            agents_reward[i] += r
    average_loss = np.mean(losses) if losses else 0.0
    return total_reward, agents_reward, steps, average_loss

def main():
    args = parse_args()
    paddle.set_device(args.device)

    # 日志目录
    logger.set_dir(f'./train_log/{args.algo}_{args.show}')

    env = make_env(args)

    # 多智能体属性
    n_agents    = env.n
    obs_shape_n = env.obs_shape_n
    act_shape_n = env.act_shape_n
    act_space_n = [env.action_space[name] for name in env.agent_names]

    # 构建 Agents
    agents = []
    for i in range(n_agents):
        model = MAModel(obs_dim=obs_shape_n[i],
                        act_dim=act_shape_n[i],
                        obs_shape_n=obs_shape_n,
                        act_shape_n=act_shape_n,
                        continuous_actions=(args.algo in ['maddpg','mappo']))
        alg   = select_algorithm(args, model, i, act_space_n)
        agent = MAAgent(alg,
                        agent_index=i,
                        obs_dim_n=obs_shape_n,
                        act_dim_n=act_shape_n,
                        batch_size=BATCH_SIZE,
                        step_size=STEP_SIZE)
        agents.append(agent)
        # 首次同步 target
        if hasattr(agent.alg, 'sync_target'):
            agent.alg.sync_target()

    total_steps = 0
    total_episodes = 0

    while total_episodes < args.max_episodes:
        # 训练一个回合
        ep_reward, ep_agent_rewards, steps, ep_loss = run_episode(env, agents, args)

        # 可选渲染
        if args.show:
            env.render()

        # summary & logger 保持原格式
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward, total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward, total_steps)
        for i, ar in enumerate(ep_agent_rewards):
            summary.add_scalar(f"Reward/Agent_{i}_Reward", ar, total_episodes)
        summary.add_scalar('Loss/Critic_Loss', ep_loss, total_episodes)

        logger.info(f'total_steps {total_steps}, episode {total_episodes}, '
                    f'reward {ep_reward}, agents rewards {ep_agent_rewards}, episode steps {steps}')

        total_steps += steps
        total_episodes += 1

        # 周期性评估
        if total_episodes % args.test_every_episodes == 0:
            eval_rewards, eval_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES, args)
            avg_eval = np.mean(eval_rewards)
            summary.add_scalar('eval/episode_reward', avg_eval, total_episodes)
            logger.info(f'Evaluation over: {EVAL_EPISODES} episodes, Reward: {avg_eval:.3f}')

            # 保存模型
            if not args.restore:
                os.makedirs(args.model_dir, exist_ok=True)
                for i, agent in enumerate(agents):
                    agent.save(os.path.join(args.model_dir, f'agent_{i}'))

    print("Training complete.")

if __name__ == '__main__':
    main()

