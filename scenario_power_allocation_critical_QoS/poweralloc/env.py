"""Slot-by-slot simulator that poses the per-slot problem to a solver."""

from __future__ import annotations

from collections import deque

import numpy as np

from .channel import ChannelModel
from .config import SystemConfig
from .phy import effective_gains, mrt_beamformers
from .problem import PowerAllocationProblem, SlotState


class PowerAllocationEnv:
    """
    At every slot the per-slot problem is posed with the current channel and the
    sliding-window history S_k(t).

        env = PowerAllocationEnv(cfg)
        state = env.reset()
        while not done:
            p = allocator.allocate(state)      # solver decides p(t)
            state, info, done = env.step(p)
        summary = env.summary()

    No reward is defined here; the solver decides how to build one from `info`.
    Slots are counted from 0, so slot t here is slot t+1 in the paper.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.channel = ChannelModel(cfg, self.rng)
        self.problem = PowerAllocationProblem(cfg)
        self._reset_buffers()

    def _reset_buffers(self):
        self.t = 0
        # Last W-1 rate vectors, i.e. the terms of S_k(t)
        self.window: deque = deque(maxlen=max(self.cfg.window_length - 1, 0))
        self.log: list[dict] = []

    def _state(self, H: np.ndarray) -> SlotState:
        W = mrt_beamformers(H)
        past = (np.array(self.window) if len(self.window)
                else np.zeros((0, self.cfg.num_users)))
        ch = self.channel
        return SlotState(self.t, H, W, effective_gains(H, W), past.sum(axis=0),
                         len(self.window), past, ch.speeds.copy(),
                         ch.positions.copy(), ch.headings.copy(),
                         self.cfg.critical_mask.copy())

    def reset(self) -> SlotState:
        self._reset_buffers()
        self.state = self._state(self.channel.reset())
        return self.state

    def step(self, p: np.ndarray):
        p = np.asarray(p, dtype=float)
        if p.shape != (self.cfg.num_users,):
            raise ValueError(f"p must have shape ({self.cfg.num_users},)")

        info = self.problem.evaluate(p, self.state)
        info["t"] = self.t
        info["p"] = p.copy()
        self.log.append(info)

        if self.window.maxlen:
            self.window.append(info["rates"])
        self.t += 1

        done = self.t >= self.cfg.num_slots
        if not done:
            self.state = self._state(self.channel.step())
        return self.state, info, done

    def summary(self, skip_warmup: bool = True, tol: float = 1e-9) -> dict:
        """
        Time-averaged performance over the episode. With skip_warmup=True the
        first W-1 slots (incomplete window) are excluded from the statistics.
        """
        start = self.cfg.window_length - 1 if skip_warmup else 0
        L = self.log[start:]
        if not L:
            raise ValueError("No slots after warm-up; increase num_slots.")
        cm, rm = self.cfg.critical_mask, self.cfg.regular_mask
        R = np.array([d["rates"] for d in L])                   # (N, K)
        Rbar = np.array([d["window_avg_rates"] for d in L])     # (N, K)
        reg_v = np.array([d["regular_violation"] for d in L])
        crit_v = np.array([d["critical_violation"] for d in L])
        return {
            "slots_evaluated": len(L),
            "avg_objective": float(np.mean([d["objective"] for d in L])),
            "avg_fairness_term": float(np.mean([d["fairness_term"] for d in L])),
            "avg_throughput_term": float(np.mean([d["throughput_term"] for d in L])),
            "avg_sum_rate": float(R.sum(axis=1).mean()),
            "avg_rates": R.mean(axis=0),
            "avg_window_rates": Rbar.mean(axis=0),
            "power_feasible": all(d["power_violation"]["nonneg"] <= tol
                                  and d["power_violation"]["budget"] <= tol for d in L),
            "critical_outage_prob": float(np.mean(crit_v[:, cm] > tol)) if cm.any() else 0.0,
            "critical_outage_per_user": np.mean(crit_v > tol, axis=0),
            "regular_violation_prob": float(np.mean(reg_v[:, rm] > tol)) if rm.any() else 0.0,
            "regular_violation_per_user": np.mean(reg_v > tol, axis=0),
            "all_constraints_slot_rate": float(np.mean([d["feasible"] for d in L])),
        }
