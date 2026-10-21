"""Actor, Q critic and Lyapunov critic networks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(inp: int, out: int, hidden: tuple) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """pi_theta: a = tanh(mu + sigma * eps), eps ~ N(0, I)."""

    def __init__(self, s_dim, a_dim, hidden):
        super().__init__()
        self.body = mlp(s_dim, 2 * a_dim, hidden)
        self.a_dim = a_dim

    def forward(self, s, deterministic=False, with_logprob=True):
        mu, log_std = self.body(s).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        z = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        logp = None
        if with_logprob:
            logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
            logp = logp - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(-1)   # tanh correction
        return a, logp


class QNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super().__init__()
        self.net = mlp(s_dim + a_dim, 1, hidden)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class LyapunovCritic(nn.Module):
    """L(s,a) >= 0 as the squared norm of the last layer."""

    def __init__(self, s_dim, a_dim, hidden):
        super().__init__()
        self.net = mlp(s_dim + a_dim, hidden[-1], hidden)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).pow(2).sum(-1)
