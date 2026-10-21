"""
Successive convex approximation (SCA) for the per-slot problem.

At every slot, given the rates of the previous slots, the problem is re-solved
from the current channel as a sequence of convex problems.

The rate of user k is a difference of two concave functions of p,

    R_k(p) = c0 [ ln T_k(p) - ln I_k(p) ],
    T_k(p) = sum_j p_j G_kj + sigma^2         (everything user k receives)
    I_k(p) = T_k(p) - p_k G_kk                (interference plus noise)

with c0 = BW / (ln 2 * rate_unit). The first term is concave; the second is the
negative of a concave function. Replacing ln I_k by its tangent at the current
point p^n gives the standard concave minorant

    R~_k(p; p^n) = c0 [ ln T_k(p) - ln I_k(p^n) - (I_k(p) - I_k(p^n)) / I_k(p^n) ]

which is tight at p^n (R~_k(p^n) = R_k(p^n)) and below R_k everywhere else. With
it, every piece of the problem becomes convex:

    objective   sum_{K_r} log(1 + Rbar~_k) + lambda sum_k Rbar~_k      concave
    (c4)        R~_k >= R_k^crit                     k in K_c           convex set
    (c3)        (S_k + R~_k) / D >= R_k^min          k in K_r           convex set
    (c1), (c2)  0 <= p_k <= P_max,  sum_k p_k <= P_max                  linear

where Rbar~_k = (S_k + R~_k) / D is the sliding-window average (S_k is fixed by
past slots). Because R~_k <= R_k, any point feasible for the convex subproblem is
feasible for the original one, so the iterates stay feasible and the objective
never decreases; the sequence converges to a KKT point of the original problem.

Each convex subproblem is solved with SLSQP using exact gradients. If no
starting point satisfies (c3) and (c4), a first phase maximises the worst
normalised slack by the same scheme; if that cannot reach zero the slot is
infeasible and the best-effort allocation is returned.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .baselines import PowerAllocator
from .config import SystemConfig
from .phy import rates
from .problem import PowerAllocationProblem, SlotState


class SCAAllocator(PowerAllocator):
    def __init__(self, cfg: SystemConfig, max_iter: int = 20, tol: float = 1e-4,
                 inner_maxiter: int = 100, margin: float = 1e-6):
        self.cfg = cfg
        self.problem = PowerAllocationProblem(cfg)
        self.max_iter = max_iter
        self.tol = tol
        self.inner_maxiter = inner_maxiter
        self.margin = margin          # Mbps inside each rate constraint, against rounding
        self.c0 = cfg.bandwidth_hz / (np.log(2.0) * cfg.rate_unit)
        self._warm = None

    def _parts(self, p, G):
        s2 = self.cfg.noise_power
        T = G @ p + s2
        I = T - np.diag(G) * p
        return T, I

    def _surrogate(self, p, G, In):
        """R~_k(p; p^n) and its (K, K) Jacobian; In = I(p^n)."""
        T, I = self._parts(p, G)
        R = self.c0 * (np.log(T) - np.log(In) - (I - In) / In)
        Gint = G - np.diag(np.diag(G))
        J = self.c0 * (G / T[:, None] - Gint / In[:, None])
        return R, J

    def _window(self, state):
        return state.past_window_sum, float(self.problem.window_denominator(state))

    def _needs(self, state):
        """Per-slot rate each constrained user needs: (index, target, scale)."""
        cfg = self.cfg
        S, D = self._window(state)
        m = self.margin
        rows = [(k, cfg.r_crit[k] + m, cfg.r_crit[k]) for k in cfg.critical_users]
        for k in np.flatnonzero(cfg.regular_mask):
            need = D * cfg.r_min[k] - S[k]                # R_k needed for (c3)
            if need > 0:                                  # inactive otherwise
                rows.append((int(k), need + D * m, cfg.r_min[k] * D))
        return rows

    def _true_objective(self, p, state):
        r = rates(p, state.G, self.cfg)
        avg = self.problem.window_avg_rates(r, state)
        return self.problem.fairness_term(avg) + self.problem.throughput_term(avg)

    def _slack(self, p, state, rows):
        """Normalised slack of every active constraint at p (>= 0 is feasible)."""
        if not rows:
            return np.array([np.inf])
        r = rates(p, state.G, self.cfg)
        return np.array([(r[k] - need) / scale for k, need, scale in rows])

    def _phase2(self, p0, state, rows):
        """Maximise the objective from a feasible p0 by SCA."""
        cfg, G = self.cfg, state.G
        S, D = self._window(state)
        Kr = cfg.regular_mask
        lam = cfg.tradeoff_weight
        bounds = [(0.0, cfg.p_max)] * cfg.num_users
        pn = p0.copy()
        f_prev = self._true_objective(pn, state)
        for _ in range(self.max_iter):
            _, In = self._parts(pn, G)

            def neg_f(p):
                R, J = self._surrogate(p, G, In)
                x = S + R
                val = np.sum(np.log1p(x[Kr] / D)) + lam * np.sum(x) / D
                w = np.where(Kr, 1.0 / (D + x), 0.0) + lam / D
                return -val, -(w @ J)

            cons = [{"type": "ineq", "fun": lambda p: cfg.p_max - p.sum(),
                     "jac": lambda p: -np.ones_like(p)}]
            for k, need, _ in rows:
                cons.append({"type": "ineq",
                             "fun": (lambda p, k=k, need=need:
                                     self._surrogate(p, G, In)[0][k] - need),
                             "jac": (lambda p, k=k: self._surrogate(p, G, In)[1][k])})
            res = minimize(neg_f, pn, jac=True, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": self.inner_maxiter, "ftol": 1e-10})
            p_new = self._project(res.x)
            # keep only steps that stay feasible and do not lower the objective
            if np.min(self._slack(p_new, state, rows)) < -1e-9:
                break
            f_new = self._true_objective(p_new, state)
            if f_new < f_prev - 1e-12:
                break
            pn = p_new
            if abs(f_new - f_prev) <= self.tol * max(1.0, abs(f_prev)):
                break
            f_prev = f_new
        return pn

    def _phase1(self, p0, state, rows):
        """Maximise the worst normalised slack by SCA; returns (p, min slack)."""
        cfg, G = self.cfg, state.G
        K = cfg.num_users
        bounds = [(0.0, cfg.p_max)] * K + [(None, None)]
        pn = p0.copy()
        for _ in range(self.max_iter):
            _, In = self._parts(pn, G)
            t0 = float(np.min(self._slack(pn, state, rows)))
            cons = [{"type": "ineq", "fun": lambda z: cfg.p_max - z[:K].sum(),
                     "jac": lambda z: np.r_[-np.ones(K), 0.0]}]
            for k, need, scale in rows:
                cons.append({"type": "ineq",
                             "fun": (lambda z, k=k, need=need, scale=scale:
                                     (self._surrogate(z[:K], G, In)[0][k] - need) / scale
                                     - z[K]),
                             "jac": (lambda z, k=k, scale=scale:
                                     np.r_[self._surrogate(z[:K], G, In)[1][k] / scale, -1.0])})
            res = minimize(lambda z: (-z[K], np.r_[np.zeros(K), -1.0]),
                           np.r_[pn, t0], jac=True, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": self.inner_maxiter, "ftol": 1e-10})
            p_new = self._project(res.x[:K])
            t_new = float(np.min(self._slack(p_new, state, rows)))
            if t_new <= t0 + 1e-10:
                break
            pn = p_new
            if t_new >= 0:
                break
        return pn, float(np.min(self._slack(pn, state, rows)))

    def _project(self, p):
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
        tot = p.sum()
        return p * (self.cfg.p_max / tot) if tot > self.cfg.p_max else p

    def allocate(self, state: SlotState) -> np.ndarray:
        cfg = self.cfg
        K = cfg.num_users
        rows = self._needs(state)

        heavy = np.full(K, 0.05)
        heavy[list(cfg.critical_users)] = 1.0
        starts = [np.full(K, cfg.p_max / K), cfg.p_max * heavy / heavy.sum()]
        if self._warm is not None:
            starts.insert(0, self._warm)

        feasible = [p for p in starts if np.min(self._slack(p, state, rows)) >= 0]
        if feasible:
            p0 = max(feasible, key=lambda p: self._true_objective(p, state))
        else:
            p0, t = max((self._phase1(p, state, rows) for p in starts[-2:]),
                        key=lambda pt: pt[1])
            if t < 0:                          # infeasible slot: best effort
                self._warm = p0.copy()
                return p0

        p = self._phase2(p0, state, rows)
        self._warm = p.copy()
        return p
