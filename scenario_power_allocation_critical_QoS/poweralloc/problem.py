"""
Per-slot power-allocation problem: objective and constraints (c1)-(c4).

    max_{p(t)}  sum_{k in K_r} log(1 + Rbar_k(p(t))) + lambda sum_k Rbar_k(p(t))
    s.t.        p_k(t) >= 0,  sum_k p_k(t) <= P_max
                Rbar_k(p(t)) >= R_k^min                    for k in K_r
                R_k(p(t))    >= R_k^crit                   for k in K_c

Sliding-window average over the last W slots (current slot included):
    Rbar_k(p(t)) = (1/W) sum_{tau=t-W+1}^{t} R_k(p(tau))
                 = (1/W) [ S_k(t) + R_k(p(t)) ],   S_k(t) = sum_{tau=t-W+1}^{t-1} R_k(tau)
S_k(t) is fixed by past decisions, so only R_k(p(t)) depends on the current p(t).
During warm-up (t < W) the average is taken over the min(t, W) slots available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import SystemConfig
from .phy import rates


@dataclass
class SlotState:
    """Everything the per-slot problem at slot t depends on."""
    t: int                      # current slot index (0-based)
    H: np.ndarray               # (K, M) current channel
    W: np.ndarray               # (M, K) MRT beamformers
    G: np.ndarray               # (K, K) effective gains |h_k^H w_j|^2
    past_window_sum: np.ndarray # (K,) S_k(t) = sum of rates in the previous W-1 slots
    past_window_count: int      # number of past slots in S_k(t), = min(t, W-1)
    past_window_rates: np.ndarray  # (count, K) those rates, oldest first
    speeds: np.ndarray          # (K,) v_k(t) in m/s
    positions: np.ndarray       # (K, 2) x_k(t) in m, BS at the origin
    headings: np.ndarray        # (K,) phi_k(t) in rad
    critical_mask: Optional[np.ndarray] = None  # (K,) True for k in K_c at this slot


OUTAGE_TOL = 1e-9


def in_outage(r: np.ndarray, target: np.ndarray, tol: float = OUTAGE_TOL) -> np.ndarray:
    """
    Boolean mask of rates that miss their target by more than `tol`.

    `r` and `target` broadcast, so this works for one slot or a whole rollout.
    """
    return np.maximum(np.asarray(target) - np.asarray(r), 0.0) > tol


class PowerAllocationProblem:
    """
    All methods take a candidate p(t) and the current SlotState, so a solver can
    evaluate any candidate before committing to it.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def window_denominator(self, state: SlotState) -> int:
        return state.past_window_count + 1            # = min(t+1, W)

    def window_avg_rates(self, r: np.ndarray, state: SlotState) -> np.ndarray:
        """Rbar_k(p(t)) = (S_k(t) + R_k(p(t))) / min(t+1, W)."""
        return (state.past_window_sum + r) / self.window_denominator(state)

    def fairness_term(self, avg_rates: np.ndarray) -> float:
        """sum_{k in K_r} log(1 + Rbar_k)."""
        return float(np.sum(np.log1p(avg_rates[self.cfg.regular_mask])))

    def throughput_term(self, avg_rates: np.ndarray) -> float:
        """lambda sum_k Rbar_k, over all users (critical and regular)."""
        return float(self.cfg.tradeoff_weight * avg_rates.sum())

    def objective(self, p: np.ndarray, state: SlotState) -> float:
        avg = self.window_avg_rates(rates(p, state.G, self.cfg), state)
        return self.fairness_term(avg) + self.throughput_term(avg)

    # Constraint violations (>= 0; zero means satisfied)
    def power_violation(self, p: np.ndarray) -> dict:
        """(c1) p_k >= 0 and (c2) sum_k p_k <= P_max."""
        return {
            "nonneg": float(np.sum(np.maximum(-p, 0.0))),
            "budget": float(max(p.sum() - self.cfg.p_max, 0.0)),
        }

    def regular_violation(self, avg_rates: np.ndarray) -> np.ndarray:
        """(c3): max(R_k^min - Rbar_k, 0) for k in K_r (0 for critical users)."""
        v = np.maximum(self.cfg.r_min - avg_rates, 0.0)
        v[self.cfg.critical_mask] = 0.0
        return v

    def critical_violation(self, r: np.ndarray) -> np.ndarray:
        """(c4): max(R_k^crit - R_k, 0) for k in K_c (0 for regular users)."""
        v = np.maximum(self.cfg.r_crit - r, 0.0)
        v[self.cfg.regular_mask] = 0.0
        return v

    def evaluate(self, p: np.ndarray, state: SlotState, tol: float = 1e-9) -> dict:
        """Objective, rates and all constraint violations for candidate p(t)."""
        p = np.asarray(p, dtype=float)
        r = rates(p, state.G, self.cfg)
        avg = self.window_avg_rates(r, state)
        pw = self.power_violation(p)
        reg_v = self.regular_violation(avg)
        crit_v = self.critical_violation(r)
        return {
            "objective": self.fairness_term(avg) + self.throughput_term(avg),
            "fairness_term": self.fairness_term(avg),
            "throughput_term": self.throughput_term(avg),
            "rates": r,
            "window_avg_rates": avg,
            "power_violation": pw,
            "regular_violation": reg_v,
            "critical_violation": crit_v,
            "feasible": (pw["nonneg"] <= tol and pw["budget"] <= tol
                         and np.all(reg_v <= tol) and np.all(crit_v <= tol)),
        }
