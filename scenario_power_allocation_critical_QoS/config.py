"""LSAC hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LSACConfig:
    hidden: tuple = (256, 256)

    batch_size: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    lr_lyapunov: float = 3e-4
    lr_entropy: float = 3e-4                 # entropy temperature beta
    lr_lyap_multiplier: float = 1e-2         # mu (adam update only)
    gamma: float = 0.99
    tau: float = 0.005                       # target smoothing
    target_entropy: Optional[float] = None   # default -dim(a)
    init_entropy_temp: float = 0.1

    # Lyapunov critic and UUB condition on the critical-user cost; False gives
    # the SAC baseline, which carries the (c4) penalty in its reward instead.
    lyapunov: bool = True
    alpha3: float = 1.0
    eta: float = 0.1                         # edge-set level epsilon (Mbps)
    init_lyap_multiplier: float = 1.0
    max_lyap_multiplier: float = 100.0
    # mu update: "adam" = Adam on log(mu) with lr_lyap_multiplier;
    #            "sgd"  = projected dual ascent
    #                     mu <- clip(mu + lr_mu_sgd * UUB, min_lyap_multiplier, max_lyap_multiplier).
    # The paper uses "sgd" with max_lyap_multiplier = 1000.
    mu_update: str = "adam"
    lr_mu_sgd: float = 1.0
    min_lyap_multiplier: float = 1e-6

    # Optional training-only safety margin (Mbps): the cost uses R_crit + crit_margin;
    # evaluation always uses R_crit. Not used in the paper (0).
    crit_margin: float = 0.0

    # Optional variable critical set (not used in the paper): critical_flag adds a
    # per-user 0/1 critical indicator to the state; random_critical (implies
    # critical_flag) draws a random critical set, with a count in
    # [min_critical, max_critical] (default max K-1), in every training episode.
    critical_flag: bool = False
    random_critical: bool = False
    min_critical: int = 1
    max_critical: Optional[int] = None

    # Reward: objective - reg_weight * sum_{k in K_r} [R_min - Rbar_k]^+ / R_min
    #   (SAC only: - crit_weight * max_{k in K_c} [R_crit - R_k]^+ / R_crit)
    reg_weight: float = 50.0
    crit_weight: float = 200.0
    reward_scale: float = 0.1

    episode_len: int = 200                   # slots per training episode
    num_episodes: int = 500
    random_episodes: int = 5                 # uniform random actions for exploration
    updates_per_step: float = 1.0            # gradient updates per collected slot
    buffer_size: int = 200_000

    # Validation, used to keep the best checkpoint
    eval_every: int = 25                     # episodes
    eval_slots: int = 1000
    eval_layouts: int = 10
    seed: int = 0
    device: str = "cpu"
