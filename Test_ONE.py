# Test_ONE.py
# 评估固定环境（env_ONE.FacadeEnv），从 max_episodes 次 episode 中挑选前 10 名，输出 CSV
# 用法示例：
# python Test_ONE.py --restore --algo madqn --max_episodes 10 --csv_out top10_results.csv

import os
import time
import argparse
import numpy as np
import pandas as pd
import paddle
import warnings

from gym import spaces

# 忽略各种 Warning（不影响 logger 输出与格式）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="0D Tensor cannot be used as 'Tensor.numpy()\\[0\\]'")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboardX.summary")

from parl.utils import logger, summary
from simple_model import MAModel
from simple_agent import MAAgent
from env_ONE import FacadeEnv
from wrapper import MultiAgentWrapper

# 算法导入：MADDPG 来自 PARL；MAPPO、MADQN、QMIX 为本地实现
from parl.algorithms import MADDPG
from qmix import QMIX
from qmix_mixer import QMixer
from qmix_model  import QMIXCompositeModel
from mappo import MAPPO
from madqn import MADQN

# ---------------- 全局超参 ----------------
CRITIC_LR             = 3e-4
ACTOR_LR              = 1e-4
GAMMA                 = 0.95
TAU                   = 0.001
BATCH_SIZE            = 1024
MAX_STEP_PER_EPISODE  = 20
STEP_SIZE             = 0.05  # 离散动作步长
TOP_K                 = 10
# ------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['maddpg', 'mappo', 'qmix', 'madqn'], default='madqn')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='评估模式：不训练、不保存模型（必须开启以导出CSV）')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--max_episodes', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--csv_out', type=str, default='top10_results.csv', help='导出 CSV 路径')
    return parser.parse_args()

def make_env(args):
    discrete = args.algo in ['qmix', 'madqn']
    # ★ 如需固定场景，请在此处传 scene_index=某索引（env_ONE 已支持）
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
                     policy_lr=ACTOR_LR,
                     value_lr=CRITIC_LR,
                     epsilon=0.2)
    if args.algo == 'qmix':
        mixer = QMixer(n_agents=len(act_space_n), mixing_hidden_dim=32, hyper_hidden_dim=64)
        composite_model = QMIXCompositeModel(agent_model=model, mixer=mixer)
        return QMIX(composite_model,
                    agent_index=agent_index,
                    act_space=act_space_n,
                    gamma=GAMMA,
                    lr=CRITIC_LR,
                    double_q=True,
                    tau=0.01)
    if args.algo == 'madqn':
        return MADQN(model,
                     agent_index=agent_index,
                     act_space=act_space_n,
                     gamma=GAMMA,
                     lr=CRITIC_LR)
    raise ValueError(f"Unsupported algorithm {args.algo}")

# —— 工具：从 MultiAgentWrapper “探底”拿到真实 FacadeEnv（以便访问 scaler/pressure_model/scene_index）
def _get_base_env(env_like):
    cand_attrs = ['env', '_env', 'base_env', 'unwrapped']
    e = env_like
    for _ in range(4):  # 最多向内探 4 层
        if hasattr(e, 'pressure_model') and hasattr(e, 'scaler'):
            return e
        found = False
        for name in cand_attrs:
            if hasattr(e, name):
                e = getattr(e, name)
                found = True
                break
        if not found:
            break
    return None

# —— 工具：解析观测中的 orig/current —— #
# 约定：obs = [orig(7), current(7), diff(7), self_act(1), other_act(6)]
def _parse_features_from_obs(obs_vec, num_agents=7):
    obs_vec = np.asarray(obs_vec, dtype=np.float32).ravel()
    orig   = obs_vec[0:num_agents]
    curr   = obs_vec[num_agents:2*num_agents]
    return orig.copy(), curr.copy()

# —— 安全取得“第一个智能体”的观测：兼容 dict/list —— #
def _get_first_obs_of_first_agent(obs_n, env):
    if isinstance(obs_n, dict):
        if hasattr(env, 'agent_names') and len(env.agent_names) > 0 and env.agent_names[0] in obs_n:
            return obs_n[env.agent_names[0]]
        first_key = next(iter(obs_n.keys()))
        return obs_n[first_key]
    elif isinstance(obs_n, (list, tuple)):
        return obs_n[0]
    else:
        return obs_n

# —— 计算 stress_index：优先用真实 env 的 scaler+model；否则返回 None —— #
def _compute_stress(base_env, features_7d):
    try:
        import torch
        x = np.asarray(features_7d, dtype=np.float32).reshape(1, -1)
        scaled = base_env.scaler.transform(x).astype(np.float32)
        t = torch.from_numpy(scaled).to(base_env.device)
        with torch.no_grad():
            v = float(base_env.pressure_model(t).item())
        return v
    except Exception:
        return None

def run_episode_eval(env, agents, args):
    """
    评估模式下一次 episode：
    - 仅使用 agent.predict(obs) 推理；
    - 返回：total_reward, agents_reward, steps, init_obs_n, last_obs_n
    """
    obs_n = env.reset()
    init_obs_n = obs_n  # 保存初始观测
    done = False
    total_reward  = 0.0
    agents_reward = [0.0] * env.n
    steps = 0
    last_obs_n = obs_n
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        total_reward += sum(reward_n)
        for i, r in enumerate(reward_n):
            agents_reward[i] += r
        obs_n = next_obs_n
        last_obs_n = obs_n
        if args.show:
            time.sleep(0.02)
            env.render()
    return total_reward, agents_reward, steps, init_obs_n, last_obs_n

def main():
    args = parse_args()
    if not args.restore:
        print("⚠ 本脚本用于评估导出 CSV，请加上 --restore 以避免训练流程。")
    paddle.set_device(args.device)

    # —— 日志目录 —— 
    logger.set_dir(f'./train_log/{args.algo}_{args.show}')

    env = make_env(args)
    base_env = _get_base_env(env)  # 以便计算 stress_index 与读取 scene_index

    # 多智能体属性
    n_agents    = env.n
    obs_shape_n = env.obs_shape_n
    act_shape_n = env.act_shape_n
    act_space_n = [env.action_space[name] for name in env.agent_names]

    # 构建 Agents 并加载参数
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
        if hasattr(agent.alg, 'sync_target'):
            agent.alg.sync_target()

    # —— 从 ./model（或 --model_dir）加载已训练好的参数 —— 
    ckpt_root = args.model_dir if args.model_dir else "./model"
    for i, agent in enumerate(agents):
        ckpt_path = os.path.join(ckpt_root, f"agent_{i}")
        if os.path.exists(ckpt_path):
            try:
                if hasattr(agent, "restore"):
                    agent.restore(ckpt_path)
                elif hasattr(agent, "load"):
                    agent.load(ckpt_path)
                else:
                    raise RuntimeError("agent has neither restore() nor load()")
                logger.info(f"[RESTORE] Loaded agent_{i} from {ckpt_path}")
            except Exception as e:
                logger.warning(f"[RESTORE] Failed to load agent_{i} from {ckpt_path}: {e}")
        else:
            logger.warning(f"[RESTORE] Checkpoint not found for agent_{i}: {ckpt_path}")

    # —— 开始评估并记录 —— #
    results = []  # 每项：dict，含 scene_index / orig_7 / final_7 / orig_stress / final_stress / total_reward
    total_steps = 0
    total_episodes = 0

    feature_names = ["WGR","BH_mean","BS_mean","BL_mean","WH_mean","WS_mean","WL_mean"]

    while total_episodes < args.max_episodes:
        ep_reward, ep_agent_rewards, steps, init_obs_n, last_obs_n = run_episode_eval(env, agents, args)

        # 取“第一个智能体”的初始/最终观测（各 agent 的 orig/current 一致）
        first_init_obs  = _get_first_obs_of_first_agent(init_obs_n, env)
        first_final_obs = _get_first_obs_of_first_agent(last_obs_n, env)

        # 解析 7 维特征（obs 排布：orig(7), current(7), diff(7), self_act(1), other_act(6)）
        orig_feats, _    = _parse_features_from_obs(first_init_obs,  num_agents=len(feature_names))
        _, final_feats   = _parse_features_from_obs(first_final_obs, num_agents=len(feature_names))

        # 计算 stress（若拿得到真实 env）
        if base_env is not None:
            orig_stress  = _compute_stress(base_env, orig_feats)
            final_stress = _compute_stress(base_env, final_feats)
            scene_idx = getattr(base_env, 'scene_index', None)
        else:
            orig_stress = None
            final_stress = None
            scene_idx = None

        # 记录结果
        row = {
            "algo": args.algo,
            "episode": total_episodes,
            "scene_index": -1 if scene_idx is None else int(scene_idx),
            "total_reward": float(ep_reward),
        }
        for i, n in enumerate(feature_names):
            row[f"orig_{n}"]  = float(orig_feats[i])
        row["orig_stress_index"] = None if orig_stress is None else float(orig_stress)
        for i, n in enumerate(feature_names):
            row[f"final_{n}"] = float(final_feats[i])
        row["final_stress_index"] = None if final_stress is None else float(final_stress)

        results.append(row)

        # —— 记录到 tensorboard 的可视化（保持原 key）——
        summary.add_scalar('train/episode_reward_wrt_episode', float(ep_reward), total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', float(ep_reward), total_steps)
        for i, ar in enumerate(ep_agent_rewards):
            summary.add_scalar(f"Reward/Agent_{i}_Reward", float(ar), total_episodes)
        summary.add_scalar('Loss/Critic_Loss', 0.0, total_episodes)

        logger.info(
            f'total_steps {total_steps}, episode {total_episodes}, '
            f'reward {ep_reward}, agents rewards {ep_agent_rewards}, episode steps {steps}'
        )

        total_steps += steps
        total_episodes += 1

    # —— 选 TOP_K 并导出 CSV —— #
    if len(results) == 0:
        print("没有评估结果可写出。")
        return

    # 排序：按 total_reward 从高到低
    results_sorted = sorted(results, key=lambda d: d["total_reward"], reverse=True)
    topk = results_sorted[:TOP_K]

    # 转为 DataFrame 并保存
    df = pd.DataFrame(topk)
    # 友好的列顺序
    cols = (
        ["algo", "episode", "scene_index"]
        + [f"orig_{n}" for n in feature_names] + ["orig_stress_index"]
        + [f"final_{n}" for n in feature_names] + ["final_stress_index"]
        + ["total_reward"]
    )
    df = df.reindex(columns=cols)

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")

    print(f"✅ 已保存 Top-{TOP_K} 结果到: {args.csv_out}")

if __name__ == '__main__':
    main()
