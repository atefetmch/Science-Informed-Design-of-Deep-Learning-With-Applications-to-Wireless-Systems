"""System parameters for the downlink power-allocation case study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SPEED_OF_LIGHT = 3e8


@dataclass
class SystemConfig:
    # Network
    num_antennas: int = 8                      # M
    num_users: int = 6                         # K
    critical_users: tuple = (0, 1)             # indices of K_c (0-based)

    # Power / noise / bandwidth
    p_max_dbm: float = 40.0                    # P_max
    bandwidth_hz: float = 10e6                 # BW
    noise_psd_dbm_hz: float = -174.0           # thermal noise PSD
    noise_figure_db: float = 9.0

    # Large-scale path loss: beta_k = beta_0 (d_0 / d_k)^eta
    beta0_db: float = -30.0                    # gain at reference distance d_0
    d0_m: float = 1.0
    pathloss_exp: float = 3.0                  # eta
    user_distances_m: Optional[np.ndarray] = None   # initial d_k(0); if None, drawn uniformly
    d_min_m: float = 20.0                      # users stay in the annulus [d_min, d_max]
    d_max_m: float = 200.0

    # Temporal correlation: rho_k(t) = J0(2 pi f_c v_k(t) / c * T_s)
    carrier_freq_hz: float = 3.5e9             # f_c
    slot_duration_s: float = 1e-3              # T_s
    user_speeds_mps: Optional[np.ndarray] = None    # mean speeds vbar_k; if None, drawn uniformly
    v_min_mps: float = 0.5
    v_max_mps: float = 15.0

    # Mobility (Gauss-Markov mobility model, updated once per slot)
    #   v_k(t)   = a v_k(t-1)   + (1-a) vbar_k   + sqrt(1-a^2) sigma_v   n
    #   phi_k(t) = a phi_k(t-1) + (1-a) phibar_k + sqrt(1-a^2) sigma_phi n'
    #   x_k(t)   = x_k(t-1) + v_k(t-1) T_s [cos phi_k(t-1), sin phi_k(t-1)]
    mobility_model: str = "gauss_markov"       # "gauss_markov" or "static"
    mobility_memory: float = 0.99              # a in [0, 1]; 1 = constant velocity
    speed_std_mps: float = 1.0                 # sigma_v
    direction_std_rad: float = 0.3             # sigma_phi

    # QoS requirements (Mbps)
    r_min: Optional[np.ndarray] = None         # R_k^min for regular users (len K)
    r_crit: Optional[np.ndarray] = None        # R_k^crit for critical users (len K)
    default_r_min: float = 5.0                 # Mbps
    default_r_crit: float = 10.0               # Mbps

    # Objective / averaging
    tradeoff_weight: float = 0.01              # lambda
    window_length: int = 50                    # W: sliding-window length for Rbar_k

    # Slots per episode
    num_slots: int = 1000

    # Rate unit (1e6 -> Mbps)
    rate_unit: float = 1e6

    seed: Optional[int] = 0

    def __post_init__(self):
        K = self.num_users
        self.critical_users = tuple(sorted(set(self.critical_users)))
        if any(k < 0 or k >= K for k in self.critical_users):
            raise ValueError("critical_users indices must lie in [0, K-1].")
        if self.window_length < 1:
            raise ValueError("window_length (W) must be >= 1.")
        if self.mobility_model not in ("gauss_markov", "static"):
            raise ValueError("mobility_model must be 'gauss_markov' or 'static'.")
        if not 0.0 <= self.mobility_memory <= 1.0:
            raise ValueError("mobility_memory must lie in [0, 1].")
        if self.r_min is None:
            self.r_min = np.full(K, self.default_r_min)
        if self.r_crit is None:
            self.r_crit = np.full(K, self.default_r_crit)
        self.r_min = np.asarray(self.r_min, dtype=float)
        self.r_crit = np.asarray(self.r_crit, dtype=float)

    @property
    def p_max(self) -> float:
        """P_max in watts."""
        return 10 ** ((self.p_max_dbm - 30) / 10)

    @property
    def noise_power(self) -> float:
        """sigma^2 in watts."""
        dbm = self.noise_psd_dbm_hz + 10 * np.log10(self.bandwidth_hz) + self.noise_figure_db
        return 10 ** ((dbm - 30) / 10)

    @property
    def critical_mask(self) -> np.ndarray:
        m = np.zeros(self.num_users, dtype=bool)
        m[list(self.critical_users)] = True
        return m

    @property
    def regular_mask(self) -> np.ndarray:
        return ~self.critical_mask
