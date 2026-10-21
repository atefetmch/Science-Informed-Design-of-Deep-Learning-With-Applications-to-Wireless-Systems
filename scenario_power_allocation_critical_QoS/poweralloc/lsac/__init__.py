"""
Lyapunov-based safe actor-critic (LSAC) with a UUB condition on the
critical-user constraint, after Han et al., Automatica 129 (2021) 109689.

  state s_t        features of the slot state, see `StateEncoder`
  action a_t       u in (-1,1)^(K+1), mapped to p(t) so that (c1), (c2) hold
  reward r_t       objective minus the (c3) penalty; (c4) is not in the reward
  cost c(s_t,a_t)  max_{k in K_c} [R_k^crit - R_k]^+, see `qos_cost`
  edge set Delta   {(s,a) : c >= eta}   (eta = epsilon in the paper)
  Lyapunov critic  L(s,a) >= 0 trained on c + gamma L'(s', pi(s'))
  UUB condition    E[ 1_Delta(s) (L(s', pi(s')) - L(s,a) + alpha3 c) ] <= 0

LSACConfig(lyapunov=False) is the SAC baseline: no Lyapunov critic and no UUB
term, with a (c4) penalty in the reward instead.
"""

from .action import ActionMapper
from .agent import LSACAgent, LSACAllocator
from .config import LSACConfig
from .cost import qos_cost
from .state import StateEncoder

__all__ = ["ActionMapper", "LSACAgent", "LSACAllocator", "LSACConfig", "StateEncoder", "qos_cost"]
