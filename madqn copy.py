# madqn.py

import warnings
warnings.simplefilter('default')

import copy
import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers

__all__ = ['MADQN']


class MADQN(Algorithm):
    """
    多智能体 DQN (MADQN) 算法，实现与单智能体 DQN 类似的接口，
    但可在多智能体情况下并行训练。
    """

    def __init__(self, model, agent_index=None, act_space=None, gamma=None, lr=None):
        """
        Args:
          model        (parl.Model): 定义 Q 网络的模型，需实现 policy(obs)->Q-values 和 value(obs)。
          agent_index  (int):       本智能体索引。
          act_space    (list):      所有智能体的 action space 列表，用于获取动作维度。
          gamma        (float):     折扣因子。
          lr           (float):     学习率。
        """
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.lr = lr

        # 动作维度
        space = act_space[agent_index]
        self.act_dim = space.n if hasattr(space, 'n') else int(space)

    def predict(self, obs):
        """
        评估时调用，返回 Q-values 向量。
        obs: Paddle Tensor, shape [B, obs_dim]
        返回: Paddle Tensor, shape [B, act_dim]
        """
        return self.model.policy(obs)

    def sample(self, obs, use_target_model=False):
        """
        训练时或构建 target 时调用，返回 Q-values 向量，
        具体动作由外部取 argmax 完成。
        obs: Paddle Tensor, shape [B, obs_dim]
        use_target_model: bool, 是否使用 target network
        返回: Paddle Tensor, shape [B, act_dim]
        """
        net = self.target_model if use_target_model else self.model
        return net.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate=None):
        """
        单步学习更新。
        Args:
          obs            Paddle Tensor, shape [B, obs_dim]
          action         Paddle Tensor (int64), shape [B]
          reward         Paddle Tensor, shape [B]
          next_obs       Paddle Tensor, shape [B, obs_dim]
          terminal       Paddle Tensor (float32), shape [B], 1 表示终止
          learning_rate  float, 可选，覆盖初始化 lr
        Returns:
          cost           Paddle Tensor 标量，当前 batch 的 loss
        """
        # 使用指定学习率或默认 lr
        if learning_rate is None:
            assert isinstance(self.lr, float)
            learning_rate = self.lr

        # 1. 预测当前 Q
        pred_q = self.model.value(obs)  # shape [B, act_dim]
        # 2. 预测下一步 Q，取最大值
        next_q = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_q, dim=1)  # shape [B]
        best_v.stop_gradient = True

        # 3. 构造 target: r + γ * (1 - done) * max_next_q
        target = reward + (1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v

        # 4. 选出当前动作对应的 Q 值
        act_onehot = layers.one_hot(action, self.act_dim)
        act_onehot = layers.cast(act_onehot, dtype='float32')
        pred_act_q = layers.reduce_sum(layers.elementwise_mul(pred_q, act_onehot), dim=1)

        # 5. MSE Loss
        cost = layers.square_error_cost(pred_act_q, target)
        cost = layers.reduce_mean(cost)

        # 6. 优化
        optimizer = fluid.optimizer.Adam(learning_rate=learning_rate, epsilon=1e-3)
        optimizer.minimize(cost)

        return cost

    def sync_target(self):
        """
        同步 model 到 target_model
        """
        self.model.sync_weights_to(self.target_model)
