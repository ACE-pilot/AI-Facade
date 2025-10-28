# qmix_mixer.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


__all__ = ['QMixer']

class QMixer(nn.Layer):
    """
    标准 QMIX 混合器（单隐层 + 两层超网络），保证对个体 Q 的单调性：
      Q_tot = |W2(s)| · ELU( |W1(s)| · Q + b1(s) ) + b2(s)

    输入:
      q_values: [B, n_agents, 1] 也可容忍 [B, n_agents] 或 list([B] / [B,1])，会自动整理
      states  : [B, state_dim]   （若是 [B] 会自动升维为 [B,1]）
    输出:
      [B, 1]
    """
    def __init__(self, n_agents, mixing_hidden_dim=32, hyper_hidden_dim=64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.hidden_dim = int(mixing_hidden_dim)
        self.hyper_hidden = int(hyper_hidden_dim)
        self._params_built = False  # 懒初始化参数（按 state_dim）
        # >>> DEBUG
        self._dbg_prints = 0
        self._last_l2 = None
        # <<< DEBUG
        
    # >>> DEBUG
    def _param_l2(self):
        total, count = 0.0, 0
        for p in self.parameters():
            try:
                v = p.detach()
                total += float((v * v).sum())    
                count += 1
            except Exception:
                pass
        return total, count

    # <<< DEBUG

    

    # ---------- 懒初始化若干线性层权重 ----------
    def _build_params(self, state_dim: int):
        # 生成 W1(s) 和 b1(s): states->[B, n_agents*H], [B, H]
        self.w1_weight = self.create_parameter(
            shape=(state_dim, self.n_agents * self.hidden_dim),
            default_initializer=nn.initializer.XavierUniform())
        self.w1_bias = self.create_parameter(
            shape=(self.n_agents * self.hidden_dim,),
            default_initializer=nn.initializer.Constant(0.0))

        self.b1_weight = self.create_parameter(
            shape=(state_dim, self.hidden_dim),
            default_initializer=nn.initializer.XavierUniform())
        self.b1_bias = self.create_parameter(
            shape=(self.hidden_dim,),
            default_initializer=nn.initializer.Constant(0.0))

        # 两层超网络生成 W2(s)、b2(s)
        self.w2h_weight = self.create_parameter(
            shape=(state_dim, self.hyper_hidden),
            default_initializer=nn.initializer.XavierUniform())
        self.w2h_bias = self.create_parameter(
            shape=(self.hyper_hidden,),
            default_initializer=nn.initializer.Constant(0.0))

        self.w2_weight = self.create_parameter(
            shape=(self.hyper_hidden, self.hidden_dim),
            default_initializer=nn.initializer.XavierUniform())
        self.w2_bias = self.create_parameter(
            shape=(self.hidden_dim,),
            default_initializer=nn.initializer.Constant(0.0))

        self.b2_weight = self.create_parameter(
            shape=(self.hyper_hidden, 1),
            default_initializer=nn.initializer.XavierUniform())
        self.b2_bias = self.create_parameter(
            shape=(1,),
            default_initializer=nn.initializer.Constant(0.0))

        self._params_built = True

        # >>> DEBUG
        try:
            n_params = sum(int(np.prod(p.shape)) for p in self.parameters())
        except Exception:
            n_params = None
        l2, cnt = self._param_l2()
        print(f"[QMixer.build] state_dim={state_dim}, params={n_params}, tensors={cnt}, L2={l2:.6f}")
        # <<< DEBUG
        
    # ---------- 辅助：线性层（x @ W + b） ----------
    def _linear(self, x, W, b):
        return paddle.matmul(x, W) + b

    # ---------- 形状整理（容错） ----------
    def _normalize_inputs(self, q_values, states):
        """
        期望：
          q_values -> [B, n_agents, 1]
          states   -> [B, state_dim]
        允许输入：
          - q_values: [B, n_agents] / [B, n_agents, 1] / list(len=n_agents, each [B] or [B,1])
          - states  : [B] / [B, state_dim]
        """
        # 处理 q_values
        if isinstance(q_values, (list, tuple)):
            if len(q_values) != self.n_agents:
                raise ValueError(f"q_values 为 list/tuple 时长度应为 n_agents={self.n_agents}，实际 {len(q_values)}")
            cols = []
            for i, q in enumerate(q_values):
                if not isinstance(q, paddle.Tensor):
                    q = paddle.to_tensor(q)
                if q.ndim == 1:           # [B]
                    q = q.unsqueeze(-1)   # [B,1]
                elif q.ndim == 2 and q.shape[-1] != 1:
                    raise ValueError(f"第 {i} 个 q 的形状期望 [B] 或 [B,1]，实际 {list(q.shape)}")
                cols.append(q.unsqueeze(1))  # -> [B,1,1]
            q_values = paddle.concat(cols, axis=1)  # [B, n_agents, 1]
        else:
            if not isinstance(q_values, paddle.Tensor):
                q_values = paddle.to_tensor(q_values)
            if q_values.ndim == 2:  # [B, n_agents]
                if q_values.shape[1] != self.n_agents:
                    raise ValueError(f"q_values 形状第二维应为 n_agents={self.n_agents}，实际 {list(q_values.shape)}")
                q_values = q_values.unsqueeze(-1)      # [B, n_agents, 1]
            elif q_values.ndim == 3:
                if q_values.shape[1] != self.n_agents:
                    raise ValueError(f"q_values 第二维应为 n_agents={self.n_agents}，实际 {list(q_values.shape)}")
                if q_values.shape[2] != 1:
                    # 若是 [B, n_agents, K]，但 K>1，尝试最后一维求和/保留第一列，这里取第一列更符合“选后的标量Q”
                    q_values = q_values[..., :1]
            else:
                raise ValueError(f"不支持的 q_values 维度：{q_values.ndim}，期望 2 或 3")

        # 处理 states
        if not isinstance(states, paddle.Tensor):
            states = paddle.to_tensor(states)
        if states.ndim == 1:  # [B] -> [B,1]
            states = states.unsqueeze(-1)
        elif states.ndim != 2:
            raise ValueError(f"states 形状应为 [B, state_dim]，实际 {list(states.shape)}")

        if q_values.dtype != paddle.float32:
            q_values = paddle.cast(q_values, paddle.float32)
        if states.dtype != paddle.float32:
            states = paddle.cast(states, paddle.float32)

        # >>> 确保对 mixer 参数的梯度计算不被意外阻断
        try:
            states.stop_gradient = False
        except Exception:
            pass
        # <<<

        return q_values, states


    # ---------- 对外混合函数 ----------

    def mix(self, q_values, states):
        """
        q_values: [B, n_agents, 1]（自动容错并转换）
        states  : [B, state_dim]
        return  : [B, 1]
        """
        q_values, states = self._normalize_inputs(q_values, states)
        # --- 单体直通（可选，但强烈建议在 n_agents==1 时打开） ---
        if self.n_agents == 1:
            # q_values: [B,1,1] or [B,1] or list([B] / [B,1])
            if isinstance(q_values, (list, tuple)):
                q = q_values[0]
            else:
                q = q_values
            if isinstance(q, paddle.Tensor):
                if q.ndim == 3 and q.shape[2] == 1:
                    q = q.squeeze(2)  # [B,1]
                if q.ndim == 2 and q.shape[1] == 1:
                    return q  # [B,1]
            # 兜底：整理成 [B,1]
            q, _ = self._normalize_inputs(q_values, states)
            return q.squeeze(2)

        B, _, _ = q_values.shape
        state_dim = int(states.shape[-1])

        if not self._params_built:
            self._build_params(state_dim)

        # hyper: W1, b1
        w1 = self._linear(states, self.w1_weight, self.w1_bias)          # [B, n_agents*H]
        w1 = paddle.reshape(w1, shape=[B, self.n_agents, self.hidden_dim])  # [B, n_agents, H]
        w1 = paddle.abs(w1)                                               # 单调性约束



        b1 = self._linear(states, self.b1_weight, self.b1_bias)          # [B, H]
        b1 = paddle.unsqueeze(b1, axis=1)                                # [B, 1, H]

        # hidden = ELU( q^T · W1 + b1 )
        qT = paddle.transpose(q_values, perm=[0, 2, 1])                  # [B, 1, n_agents]
        hidden = paddle.matmul(qT, w1) + b1                              # [B, 1, H]
        hidden = F.elu(hidden)

        # hyper: W2, b2（两层）
        h = F.relu(self._linear(states, self.w2h_weight, self.w2h_bias))  # [B, hyper_hidden]
        w2 = self._linear(h, self.w2_weight, self.w2_bias)                # [B, H]
        w2 = paddle.abs(w2)                                               # 单调性约束
        w2 = paddle.unsqueeze(w2, axis=2)                                 # [B, H, 1]

        b2 = self._linear(h, self.b2_weight, self.b2_bias)                # [B, 1]
        b2 = paddle.unsqueeze(b2, axis=1)                                 # [B, 1, 1]

        q_tot = paddle.matmul(hidden, w2) + b2
        out = paddle.squeeze(q_tot, axis=2)
        # mean_val = float(out.mean())  
        # std_val  = float(out.std())   
        # print(f"[QMixer.mix/out] out={list(out.shape)}, "
        #       f"mean={mean_val:.4f}, std={std_val:.4f}")
        return out

