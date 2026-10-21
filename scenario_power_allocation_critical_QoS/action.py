"""Maps the squashed policy output to a power vector satisfying (c1) and (c2)."""

from __future__ import annotations

import numpy as np

from ..config import SystemConfig

SPLIT_FLOOR = 1e-3                 # keeps the split weights strictly positive


class ActionMapper:
    """
    u in (-1,1)^(K+1):
        frac  = (u_0 + 1)/2                       total-power fraction
        w_k   = (u_k + 1)/2 + floor               split weights
        p_k   = P_max * frac * w_k / sum_j w_j    => p_k >= 0, sum_k p_k <= P_max
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.dim = cfg.num_users + 1

    def __call__(self, u: np.ndarray) -> np.ndarray:
        # float64 so that a total at P_max does not overshoot it through rounding
        u = np.asarray(u, dtype=np.float64)
        frac = (u[0] + 1.0) / 2.0
        w = (u[1:] + 1.0) / 2.0 + SPLIT_FLOOR
        p = self.cfg.p_max * frac * w / w.sum()
        total = p.sum()
        return p * (self.cfg.p_max / total) if total > self.cfg.p_max else p
