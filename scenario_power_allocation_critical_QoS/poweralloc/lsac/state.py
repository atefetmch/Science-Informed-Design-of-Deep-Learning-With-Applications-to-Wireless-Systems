"""Maps a SlotState to the flat feature vector s_t fed to the networks."""

from __future__ import annotations

import numpy as np

from ..config import SystemConfig
from ..problem import SlotState


class StateEncoder:
    """
    s_t = [H(t), m(t), R(t)] as in the case study:

      H(t)  channel state, given through the MRT effective gains
            log10(1 + |h_k^H w_j|^2 / sigma^2) / 10 for all k, j   (K*K values).
            The rates depend on H(t) only through these gains, and unlike the raw
            h_k they do not carry the arbitrary per-user phase.
      m(t)  mobility state per user: x_k / d_max, y_k / d_max, v_k / v_max,
            cos(phi_k), sin(phi_k)                                   (5*K values)
      R(t)  rate vectors of the previous W-1 slots, oldest first, divided by the
            user's QoS threshold; missing slots in the warm-up are zero ((W-1)*K values)

    plus the window fill fraction n / (W-1), so warm-up zeros can be told apart
    from real zero rates.

    With critical_flag=True (optional), K more values are appended (1 if user k
    is critical at this slot, else 0), and the rate history is normalised by the
    target of the user's current role.
    """

    def __init__(self, cfg: SystemConfig, critical_flag: bool = False):
        self.cfg = cfg
        K = cfg.num_users
        self.critical_flag = critical_flag
        self.hist_len = max(cfg.window_length - 1, 0)
        self.dim = K * K + 5 * K + self.hist_len * K + 1 + (K if critical_flag else 0)
        self.qos_ref = self._qos_ref(cfg.critical_mask)

    def _qos_ref(self, mask: np.ndarray) -> np.ndarray:
        ref = np.where(mask, self.cfg.r_crit, self.cfg.r_min).astype(float)
        return np.maximum(ref, 1e-6)

    def __call__(self, state: SlotState) -> np.ndarray:
        cfg, K = self.cfg, self.cfg.num_users
        g = np.log10(1.0 + state.G / cfg.noise_power) / 10.0                    # (K, K)

        mobility = np.concatenate([
            state.positions / cfg.d_max_m,
            (state.speeds / max(cfg.v_max_mps, 1e-6))[:, None],
            np.cos(state.headings)[:, None],
            np.sin(state.headings)[:, None],
        ], axis=1)                                                               # (K, 5)

        if self.critical_flag:
            mask = (state.critical_mask if state.critical_mask is not None
                    else cfg.critical_mask)
            qos_ref = self._qos_ref(mask)
        else:                       # roles fixed by user index
            qos_ref = self.qos_ref

        n = state.past_window_count
        hist = np.zeros((self.hist_len, K))
        if n:
            hist[self.hist_len - n:] = state.past_window_rates / qos_ref

        feats = [g, mobility, hist, [n / max(self.hist_len, 1)]]
        if self.critical_flag:
            feats.append(mask.astype(float))
        return np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])
