# train.py
# python train.py --algo maddpg
import os
import time
import argparse
import numpy as np
import paddle
import warnings
from gym import spaces
from datetime import datetime

# 忽略各种 Deprecation 和 UserWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="0D Tensor cannot be used as 'Tensor.numpy()\\[0\\]'")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboardX.summary")

from parl.utils import logger, summary
from parl.utils import ReplayMemory  # === QMIX PATCH: central replay 需要 ===
from simple_model import MAModel
from simple_agent import MAAgent
from env import FacadeEnv
from env_cartpole import CartPoleMultiEnv
from wrapper import MultiAgentWrapper

# 算法导入：MADDPG来自 parl；MAPPO、MADQN 为本地实现
from parl.algorithms import MADDPG
from qmix import QMIX
from qmix_mixer import QMixer
from qmix_model  import QMIXCompositeModel

from mappo import MAPPO
from madqn import MADQN

# 全局超参
CRITIC_LR             = 3e-4 #0.001 
ACTOR_LR              = 1e-4 #0.0001
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
                        default='maddpg')
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--max_episodes', type=int, default=500000)
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
        return MAPPO(
            model,
            agent_index=agent_index,
            act_space=act_space_n,
            gamma=GAMMA,
            policy_lr=ACTOR_LR,
            value_lr=CRITIC_LR,
            epsilon=0.2,
        )
    if args.algo == 'qmix':
        # 构建 Mixer + 组合模型
        mixer = QMixer(n_agents=len(act_space_n), mixing_hidden_dim=32, hyper_hidden_dim=64)
        composite_model = QMIXCompositeModel(agent_model=model, mixer=mixer)

        # 返回我们的 QMIX 算法外壳（接口与 MAAgent 对接一致）
        return QMIX(
            composite_model,
            agent_index=agent_index,
            act_space=act_space_n,
            gamma=GAMMA,
            lr=CRITIC_LR,
            double_q=True,   # 建议开
            tau=0.01         # 软更新；也可以 None 然后外部周期调 sync_target()
        )

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

def _flat_concat_obs(obs_list):
    """把每个体的观测扁平化后拼接，返回 1D 向量。更稳健地应对不同 shape。"""
    return np.concatenate([np.asarray(o).ravel() for o in obs_list], axis=-1).astype('float32')

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

        # === QMIX PATCH: 仅在 QMIX 时写“集中式回放 + 联合动作”；否则保持原 per-agent 存储 ===
        if getattr(agents[0], 'algo_name', None) == 'qmix' and getattr(agents[0], 'shared_rpm', None) is not None:
            # 1) 联合动作（按环境实际执行）
            joint_act = np.asarray([int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a) for a in action_n],dtype=np.int64)
            # 2) 拼接观测
            obs_concat  = _flat_concat_obs(obs_n)               # [sum_obs]
            next_concat = _flat_concat_obs(next_obs_n)          # [sum_obs]
            # 3) 团队奖励（总和）与终止
            team_reward = float(np.sum(reward_n))
            terminal    = bool(all(done_n))
            # 4) 仅写入集中式回放
            agents[0].shared_rpm.append(
                obs=obs_concat,
                act=joint_act,
                reward=team_reward,
                next_obs=next_concat,
                terminal=terminal,
            )
        else:
            # 非 QMIX：保持你原有的 per-agent 存经验
            for i, agent in enumerate(agents):
                agent.add_experience(obs_n[i],
                                     action_n[i],
                                     reward_n[i],
                                     next_obs_n[i],
                                     done_n[i])
        # === END QMIX PATCH ===

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
    obs_shape_n = env.obs_shape_n   # 这里每个元素可能是 int 或 tuple
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

    # === QMIX PATCH: 创建集中式回放并注入，仅在 algo==qmix 时生效 ===
    if args.algo == 'qmix':
        # 把每个体的 obs shape（可能是 int 或 tuple）扁平化后求和
        def _flat_dim(x):
            if isinstance(x, (list, tuple)):
                p = 1
                for v in x: p *= int(v)
                return int(p)
            return int(x)
        sum_obs_dim = int(sum(_flat_dim(s) for s in obs_shape_n))  # ★ 修复：不再直接 sum(tuple)
        central_rpm = ReplayMemory(
            max_size=200000,
            obs_dim=sum_obs_dim,   # concat 后的 obs
            act_dim=n_agents,      # 联合动作
        )
        for ag in agents:
            setattr(ag, 'algo_name', 'qmix')
            setattr(ag, 'shared_rpm', central_rpm)
            setattr(ag, 'n_agents', n_agents)
    # === END QMIX PATCH ===

    total_steps = 0
    total_episodes = 0

    while total_episodes < args.max_episodes:
        # 训练一个回合
        ep_reward, ep_agent_rewards, steps, ep_loss = run_episode(env, agents, args)

        # 可选渲染
        if args.show:
            env.render()

        # summary & logger 保持原格式
        summary.add_scalar('train/episode_reward_wrt_episode', float(ep_reward), total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', float(ep_reward), total_steps)
        for i, ar in enumerate(ep_agent_rewards):
            summary.add_scalar(f"Reward/Agent_{i}_Reward", float(ar), total_episodes)
        summary.add_scalar('Loss/Critic_Loss', float(ep_loss), total_episodes)


        logger.info(f'total_steps {total_steps}, episode {total_episodes}, '
                    f'reward {ep_reward}, agents rewards {ep_agent_rewards}, episode steps {steps}')

        total_steps += steps
        total_episodes += 1

        # 周期性评估
        if total_episodes % args.test_every_episodes == 0:
            eval_rewards, eval_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES, args)
            avg_eval = np.mean(eval_rewards)
            summary.add_scalar('eval/episode_reward', float(avg_eval), total_episodes)
            logger.info(f'Evaluation over: {EVAL_EPISODES} episodes, Reward: {avg_eval:.3f}')

            # 保存模型
            if not args.restore:
                os.makedirs(args.model_dir, exist_ok=True)
                for i, agent in enumerate(agents):
                    agent.save(os.path.join(args.model_dir, f'agent_{i}'))

    print("Training complete.")

if __name__ == '__main__':
    main()
