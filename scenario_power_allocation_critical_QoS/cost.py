"""Constraint cost c(s_t, a_t) used by the Lyapunov critic."""

from __future__ import annotations

import numpy as np

from ..config import SystemConfig


def critical_shortfall(info: dict, cfg: SystemConfig, margin: float = 0.0) -> np.ndarray:
    """
    Per-user shortfall [R_k^crit + margin - R_k]^+ for k in K_c (0 for regular users).

    margin > 0 tightens the critical constraint during training only (constraint
    back-off): the cost becomes positive while a critical user is still above
    R_crit but within `margin` Mbps of it. With margin = 0 this equals
    info["critical_violation"], i.e. the true (c4) shortfall.
    """
    if margin == 0.0:
        return info["critical_violation"]
    v = np.maximum(cfg.r_crit + margin - info["rates"], 0.0)
    v[cfg.regular_mask] = 0.0
    return v


def qos_cost(info: dict, cfg: SystemConfig, margin: float = 0.0) -> float:
    """c = max_{k in K_c} [R_k^crit + margin - R_k]^+ for the slot the environment just evaluated."""
    crit = critical_shortfall(info, cfg, margin)[cfg.critical_mask]
    return float(np.max(crit)) if crit.size else 0.0
