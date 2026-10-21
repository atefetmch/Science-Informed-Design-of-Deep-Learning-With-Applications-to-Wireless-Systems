"""User mobility, path loss and Gauss-Markov small-scale fading."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import j0

from .config import SPEED_OF_LIGHT, SystemConfig


class ChannelModel:
    """
    User mobility + path loss + first-order Gauss-Markov Rayleigh fading.

    Per slot t (quantities held constant within the slot):
        x_k(t)     position of user k (BS at the origin), d_k(t) = ||x_k(t)||
        v_k(t)     speed, phi_k(t) heading
        beta_k(t)  = beta_0 (d_0 / d_k(t))^eta
        rho_k(t)   = J_0(2 pi f_c v_k(t) T_s / c)
        h~_k(t)    = rho_k(t) h~_k(t-1) + sqrt(1 - rho_k(t)^2) e_k(t)
        h_k(t)     = sqrt(beta_k(t)) h~_k(t)
    """

    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.h_tilde: Optional[np.ndarray] = None   # (K, M)

    def _cn(self, *shape) -> np.ndarray:
        """i.i.d. CN(0, 1) samples."""
        return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / np.sqrt(2)

    def _update_large_scale(self):
        """Recompute d_k, beta_k, f_{D,k}, rho_k from the current positions and speeds."""
        cfg = self.cfg
        self.distances = np.linalg.norm(self.positions, axis=1)
        beta0 = 10 ** (cfg.beta0_db / 10)
        self.beta = beta0 * (cfg.d0_m / self.distances) ** cfg.pathloss_exp
        self.doppler = self.speeds * cfg.carrier_freq_hz / SPEED_OF_LIGHT
        self.rho = j0(2 * np.pi * self.doppler * cfg.slot_duration_s)

    def _drop_users(self):
        """Initial positions, speeds and headings for a new episode."""
        cfg, rng, K = self.cfg, self.rng, self.cfg.num_users
        d0 = (np.asarray(cfg.user_distances_m, dtype=float)
              if cfg.user_distances_m is not None
              else rng.uniform(cfg.d_min_m, cfg.d_max_m, K))
        angle = rng.uniform(0, 2 * np.pi, K)
        self.positions = np.stack([d0 * np.cos(angle), d0 * np.sin(angle)], axis=1)  # (K, 2)

        self.mean_speeds = (np.asarray(cfg.user_speeds_mps, dtype=float)
                            if cfg.user_speeds_mps is not None
                            else rng.uniform(cfg.v_min_mps, cfg.v_max_mps, K))
        self.speeds = self.mean_speeds.copy()
        self.headings = rng.uniform(0, 2 * np.pi, K)
        self.mean_headings = self.headings.copy()

    def _move_users(self):
        """One Gauss-Markov mobility step; users are reflected at the annulus edges."""
        cfg, rng, K = self.cfg, self.rng, self.cfg.num_users
        if cfg.mobility_model == "static":
            return

        # Position update uses the velocity held during the previous slot
        step = self.speeds * cfg.slot_duration_s
        self.positions = self.positions + step[:, None] * np.stack(
            [np.cos(self.headings), np.sin(self.headings)], axis=1)

        # Velocity update for the new slot
        a = cfg.mobility_memory
        s = np.sqrt(1 - a ** 2)
        self.speeds = (a * self.speeds + (1 - a) * self.mean_speeds
                       + s * cfg.speed_std_mps * rng.standard_normal(K))
        self.speeds = np.clip(self.speeds, 0.0, cfg.v_max_mps)
        self.headings = (a * self.headings + (1 - a) * self.mean_headings
                         + s * cfg.direction_std_rad * rng.standard_normal(K))

        # Keep users in d_min <= d_k <= d_max: put them back on the boundary and
        # reflect the heading off it (radial velocity component flips sign).
        d = np.linalg.norm(self.positions, axis=1)
        out = (d > cfg.d_max_m) | (d < cfg.d_min_m)
        if np.any(out):
            radial = self.positions[out] / d[out, None]                    # unit normal
            target = np.where(d[out] > cfg.d_max_m, cfg.d_max_m, cfg.d_min_m)
            self.positions[out] = radial * target[:, None]
            vel = np.stack([np.cos(self.headings[out]), np.sin(self.headings[out])], axis=1)
            vel = vel - 2 * np.sum(vel * radial, axis=1, keepdims=True) * radial
            new_heading = np.arctan2(vel[:, 1], vel[:, 0])
            self.headings[out] = new_heading
            self.mean_headings[out] = new_heading      # stop drifting back into the wall

    def reset(self) -> np.ndarray:
        """New user drop; h~(0) from the stationary distribution CN(0, I)."""
        self._drop_users()
        self._update_large_scale()
        self.h_tilde = self._cn(self.cfg.num_users, self.cfg.num_antennas)
        return self.channel()

    def step(self) -> np.ndarray:
        """Advance to the next slot: move users, update beta/rho, evolve fading."""
        self._move_users()
        self._update_large_scale()
        e = self._cn(self.cfg.num_users, self.cfg.num_antennas)
        rho = self.rho[:, None]
        self.h_tilde = rho * self.h_tilde + np.sqrt(1 - rho ** 2) * e
        return self.channel()

    def channel(self) -> np.ndarray:
        """H(t) with rows h_k(t)^T, shape (K, M)."""
        return np.sqrt(self.beta)[:, None] * self.h_tilde
