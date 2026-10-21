"""Replay buffers D and D_Delta."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Stores (s, a, r, c, s', 1_Delta(s))."""

    def __init__(self, s_dim, a_dim, size, rng):
        self.s = np.zeros((size, s_dim), np.float32)
        self.a = np.zeros((size, a_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.c = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, s_dim), np.float32)
        self.e = np.zeros(size, np.float32)
        self.size, self.ptr, self.n, self.rng = size, 0, 0, rng

    def add(self, s, a, r, c, s2, e):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.c[i], self.s2[i], self.e[i] = s, a, r, c, s2, e
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch, device):
        idx = self.rng.integers(0, self.n, size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return dict(s=t(self.s), a=t(self.a), r=t(self.r), c=t(self.c), s2=t(self.s2), e=t(self.e))
