"""MRT beamforming, SINR and achievable rates."""

from __future__ import annotations

import numpy as np

from .config import SystemConfig


def mrt_beamformers(H: np.ndarray) -> np.ndarray:
    """Unit-norm MRT beamformers. Returns W of shape (M, K), column k = w_k."""
    return (H / np.linalg.norm(H, axis=1, keepdims=True)).T


def effective_gains(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """G[k, j] = |h_k^H w_j|^2, shape (K, K)."""
    return np.abs(H.conj() @ W) ** 2


def sinr(p: np.ndarray, G: np.ndarray, noise_power: float) -> np.ndarray:
    """gamma_k = p_k G_kk / (sum_{j != k} p_j G_kj + sigma^2)."""
    signal = np.diag(G) * p
    interference = G @ p - signal
    return signal / (interference + noise_power)


def rates(p: np.ndarray, G: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """R_k = BW log2(1 + gamma_k), in units of cfg.rate_unit."""
    return cfg.bandwidth_hz * np.log2(1 + sinr(p, G, cfg.noise_power)) / cfg.rate_unit
