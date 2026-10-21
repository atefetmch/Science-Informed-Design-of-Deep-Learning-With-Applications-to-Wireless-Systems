"""Solver interface and simple reference allocators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import SystemConfig
from .env import PowerAllocationEnv
from .problem import SlotState


class PowerAllocator(ABC):
    @abstractmethod
    def allocate(self, state: SlotState) -> np.ndarray:
        """Return p(t) with shape (K,)."""


class EqualPowerAllocator(PowerAllocator):
    """Baseline for sanity checks: p_k = P_max / K."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def allocate(self, state: SlotState) -> np.ndarray:
        return np.full(self.cfg.num_users, self.cfg.p_max / self.cfg.num_users)


def run_episode(env: PowerAllocationEnv, allocator: PowerAllocator) -> dict:
    state, done = env.reset(), False
    while not done:
        state, _, done = env.step(allocator.allocate(state))
    return env.summary()
