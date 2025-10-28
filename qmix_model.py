# qmix_model.py
import parl
from paddle import fluid
F = fluid.layers

__all__ = ['QMIXCompositeModel']

class QMIXCompositeModel(parl.Model):
    """
    组合模型：复用你现有的 MAModel（个体 Q 网络），外加一个 QMixer。
    对外接口（与算法外壳约定）：
      - policy(obs) / value(obs): 返回本体 Q(o_i, ·)（沿用 MAModel）
      - mix(q_values, states)   : 返回 Q_tot（给 QMIX 算法调用）
    """
    def __init__(self, agent_model, mixer):
        super(QMIXCompositeModel, self).__init__()
        self.agent_model = agent_model
        self.mixer = mixer
        # >>> DEBUG
        self._dbg = {'pol': 0, 'val': 0, 'mix': 0}
        # <<< DEBUG

    def policy(self, obs):
        out = self.agent_model.policy(obs)
        # >>> DEBUG
        if self._dbg['pol'] < 3:
            try:
                print(f"[QMIXModel.policy] obs={list(getattr(obs,'shape',()))} "
                      f"-> out={list(getattr(out,'shape',()))}, dtype={getattr(out,'dtype',type(out))}")
            except Exception as e:
                print(f"[QMIXModel.policy dbg error] {e}")
            self._dbg['pol'] += 1
        # <<< DEBUG
        return out


    def value(self, obs):
        """QMIX 需要 Q(o,·) 向量：优先从 value(obs) 取，若无再回退到 policy(obs)。"""
        try:
            out = self.agent_model.value(obs)
        except Exception:
            out = self.agent_model.policy(obs)
        # >>> DEBUG
        if self._dbg['val'] < 3:
            try:
                print(f"[QMIXModel.value]  obs={list(getattr(obs,'shape',()))} "
                      f"-> out={list(getattr(out,'shape',()))}, dtype={getattr(out,'dtype',type(out))}")
            except Exception as e:
                print(f"[QMIXModel.value dbg error] {e}")
            self._dbg['val'] += 1
        # <<< DEBUG
        return out



    def mix(self, q_values, states):
        out = self.mixer.mix(q_values, states)
        # mean_val = float(out.mean())  
        # std_val  = float(out.std())   
        # print(f"[QMIXModel.mix/out] q_tot shape={list(out.shape)}, "
        #       f"mean={mean_val:.4f}, std={std_val:.4f}")
        return out



