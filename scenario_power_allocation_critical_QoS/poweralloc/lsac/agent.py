"""LSAC agent: update rule, episode collection, training loop and evaluation."""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch

from ..baselines import PowerAllocator, run_episode
from ..config import SystemConfig
from ..env import PowerAllocationEnv
from ..problem import SlotState
from .action import ActionMapper
from .buffer import ReplayBuffer
from .config import LSACConfig
from .cost import critical_shortfall, qos_cost
from .networks import GaussianPolicy, LyapunovCritic, QNetwork
from .state import StateEncoder


class LSACAgent:
    def __init__(self, cfg: SystemConfig, lcfg: LSACConfig):
        self.cfg, self.lcfg = cfg, lcfg
        if lcfg.mu_update not in ("adam", "sgd"):
            raise ValueError(f"mu_update must be 'adam' or 'sgd', got {lcfg.mu_update!r}")
        torch.manual_seed(lcfg.seed)
        self.rng = np.random.default_rng(lcfg.seed)
        self.dev = torch.device(lcfg.device)

        self.encoder = StateEncoder(cfg, critical_flag=lcfg.critical_flag or lcfg.random_critical)
        self.mapper = ActionMapper(cfg)
        s_dim, a_dim, h = self.encoder.dim, self.mapper.dim, lcfg.hidden

        self.pi = GaussianPolicy(s_dim, a_dim, h).to(self.dev)
        self.q1, self.q2 = QNetwork(s_dim, a_dim, h).to(self.dev), QNetwork(s_dim, a_dim, h).to(self.dev)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.lc = LyapunovCritic(s_dim, a_dim, h).to(self.dev)
        self.lc_t = copy.deepcopy(self.lc)
        for net in (self.q1_t, self.q2_t, self.lc_t):
            net.requires_grad_(False)

        # Multipliers in log space so beta, mu >= 0
        self.log_beta = torch.tensor(np.log(lcfg.init_entropy_temp), device=self.dev, requires_grad=True)
        self.log_mu = torch.tensor(np.log(lcfg.init_lyap_multiplier), device=self.dev, requires_grad=True)
        self.target_entropy = (lcfg.target_entropy if lcfg.target_entropy is not None
                               else -float(a_dim))

        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lcfg.lr_actor)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                      lr=lcfg.lr_critic)
        self.opt_lc = torch.optim.Adam(self.lc.parameters(), lr=lcfg.lr_lyapunov)
        self.opt_beta = torch.optim.Adam([self.log_beta], lr=lcfg.lr_entropy)
        self.opt_mu = torch.optim.Adam([self.log_mu], lr=lcfg.lr_lyap_multiplier)

        self.D = ReplayBuffer(s_dim, a_dim, lcfg.buffer_size, self.rng)
        self.D_edge = ReplayBuffer(s_dim, a_dim, lcfg.buffer_size, self.rng)

    @property
    def beta(self):
        return self.log_beta.exp()

    @property
    def mu(self):
        return self.log_mu.exp()

    def reward(self, info: dict, cfg: Optional[SystemConfig] = None) -> float:
        """
        r_t = objective - reg_weight * sum_{k in K_r} [R_min - Rbar_k]^+ / R_min; the
        SAC baseline also subtracts
        crit_weight * max_{k in K_c} [R_crit + crit_margin - R_k]^+ / R_crit.
        `cfg` gives the critical set of the slot (default: the agent's own).
        """
        lc, cfg = self.lcfg, (cfg if cfg is not None else self.cfg)
        r = info["objective"]
        rm = cfg.regular_mask
        r -= lc.reg_weight * float(np.sum(info["regular_violation"][rm] / np.maximum(cfg.r_min[rm], 1e-9)))
        if not lc.lyapunov and cfg.critical_mask.any():
            cm = cfg.critical_mask
            short = critical_shortfall(info, cfg, lc.crit_margin)
            r -= lc.crit_weight * float(np.max(short[cm] / np.maximum(cfg.r_crit[cm], 1e-9)))
        return r

    def act(self, s_vec: np.ndarray, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            a, _ = self.pi(torch.as_tensor(s_vec, device=self.dev).unsqueeze(0),
                           deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self) -> dict:
        """One gradient step on every network and multiplier."""
        lc = self.lcfg
        b = self.D.sample(lc.batch_size, self.dev)
        # drawn for SAC too, so both methods consume the same random stream
        be = self.D_edge.sample(lc.batch_size, self.dev) if self.D_edge.n > 0 else None
        use_uub = lc.lyapunov and be is not None
        beta, mu = self.beta.detach(), self.mu.detach()

        # (a) Q critics: 1/2 [r + gamma (min_j Q_j'(s',a') - beta log pi(a'|s')) - Q_i(s,a)]^2
        with torch.no_grad():
            a2, logp2 = self.pi(b["s2"])
            q_t = torch.min(self.q1_t(b["s2"], a2), self.q2_t(b["s2"], a2)) - beta * logp2
            y = lc.reward_scale * b["r"] + lc.gamma * q_t
        loss_q = 0.5 * ((self.q1(b["s"], b["a"]) - y) ** 2).mean() \
               + 0.5 * ((self.q2(b["s"], b["a"]) - y) ** 2).mean()
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        # (b) Lyapunov critic: 1/2 (L(s,a) - c - gamma L'(s', pi(s')))^2
        with torch.no_grad():
            a2l, _ = self.pi(b["s2"], with_logprob=False)
            l_tgt = b["c"] + lc.gamma * self.lc_t(b["s2"], a2l)
        loss_lc = 0.5 * ((self.lc(b["s"], b["a"]) - l_tgt) ** 2).mean()
        if lc.lyapunov:
            self.opt_lc.zero_grad(); loss_lc.backward(); self.opt_lc.step()

        # (c) Actor: -min Q(s, pi(s)) + beta log pi(.|s)
        #            + mu * mean[1_Delta(s) (L(s', pi(s')) - L(s,a) + alpha3 c)]
        a_new, logp = self.pi(b["s"])
        q_new = torch.min(self.q1(b["s"], a_new), self.q2(b["s"], a_new))
        loss_pi = (-q_new + beta * logp).mean()
        uub = torch.tensor(0.0, device=self.dev)
        if use_uub:
            a2e, _ = self.pi(be["s2"], with_logprob=False)
            l_next = self.lc(be["s2"], a2e) * be["e"]
            with torch.no_grad():
                l_now = (self.lc(be["s"], be["a"]) - lc.alpha3 * be["c"]) * be["e"]
            uub = (l_next - l_now).mean()
            loss_pi = loss_pi + mu * uub
        self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()

        # (d) Multipliers: beta towards the target entropy, mu up while the
        #     decrease condition is violated, within its bounds
        loss_beta = -(self.log_beta * (logp.detach() + self.target_entropy)).mean()
        self.opt_beta.zero_grad(); loss_beta.backward(); self.opt_beta.step()
        if use_uub and lc.mu_update == "sgd":
            with torch.no_grad():
                mu_new = (self.log_mu.exp() + lc.lr_mu_sgd * uub.detach()).clamp(
                    min=lc.min_lyap_multiplier, max=lc.max_lyap_multiplier)
                self.log_mu.copy_(mu_new.log())
        elif use_uub:
            loss_mu = -(self.log_mu * uub.detach())
            self.opt_mu.zero_grad(); loss_mu.backward(); self.opt_mu.step()
            with torch.no_grad():
                self.log_mu.clamp_(max=np.log(lc.max_lyap_multiplier))

        # (e) Target networks: phi' <- tau phi + (1 - tau) phi'
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t), (self.lc, self.lc_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - lc.tau).add_(lc.tau * p)

        return dict(loss_q=loss_q.item(), loss_lc=loss_lc.item(), loss_pi=loss_pi.item(),
                    uub=uub.item(), beta=self.beta.item(), mu=self.mu.item())

    def critical_range(self) -> tuple[int, int]:
        K, lc = self.cfg.num_users, self.lcfg
        hi = K - 1 if lc.max_critical is None else lc.max_critical
        if not 0 <= lc.min_critical <= hi <= K:
            raise ValueError(f"need 0 <= min_critical <= max_critical <= K, got "
                             f"{lc.min_critical}, {hi}, K={K}")
        return lc.min_critical, hi

    def draw_critical_set(self, rng: np.random.Generator) -> tuple:
        lo, hi = self.critical_range()
        n = int(rng.integers(lo, hi + 1))
        return tuple(sorted(int(k) for k in rng.choice(self.cfg.num_users, n, replace=False)))

    def collect_episode(self, env: PowerAllocationEnv, random_actions: bool) -> dict:
        """
        Run one episode, store its transitions in D, and the prefix t <= N in
        D_Delta, where N is the last slot whose state is in the edge set.
        With random_critical the episode's critical set is drawn first; the
        environment, reward and cost then all use that set.
        """
        lc, cfg = self.lcfg, env.cfg
        if lc.random_critical:
            if env.cfg is self.cfg:
                raise ValueError("random_critical needs an environment with its own config")
            cfg.critical_users = self.draw_critical_set(self.rng)
        state, done = env.reset(), False
        S, A, R, C = [], [], [], []
        while not done:
            s_vec = self.encoder(state)
            u = (self.rng.uniform(-1, 1, self.mapper.dim) if random_actions
                 else self.act(s_vec))
            state, info, done = env.step(self.mapper(u))
            S.append(s_vec); A.append(u)
            R.append(self.reward(info, cfg)); C.append(qos_cost(info, cfg, lc.crit_margin))

        C = np.asarray(C)
        edge = (C >= lc.eta).astype(np.float32)          # 1_Delta(s_t)
        L = len(S)
        for t in range(L - 1):                            # the last slot has no successor
            self.D.add(S[t], A[t], R[t], C[t], S[t + 1], edge[t])
        edge_idx = np.flatnonzero(edge)
        N = int(edge_idx.max()) if edge_idx.size else -1
        for t in range(min(N + 1, L - 1)):
            self.D_edge.add(S[t], A[t], R[t], C[t], S[t + 1], edge[t])

        return dict(ep_return=float(np.sum(R)), ep_cost=float(np.sum(C)),
                    edge_frac=float(edge.mean()), N=N, steps=L - 1,
                    num_critical=len(cfg.critical_users))

    def train(self, on_eval=None) -> list[dict]:
        """`on_eval(episode, eval_dict)` runs after every validation."""
        lc = self.lcfg
        train_cfg = copy.copy(self.cfg)
        train_cfg.num_slots = lc.episode_len
        env = PowerAllocationEnv(train_cfg)

        history, t0 = [], time.time()
        for ep in range(lc.num_episodes):
            ep_info = self.collect_episode(env, random_actions=ep < lc.random_episodes)
            upd = {}
            if self.D.n >= lc.batch_size:
                n_upd = int(round(lc.updates_per_step * ep_info["steps"]))
                logs = [self.update() for _ in range(n_upd)]
                upd = {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}
            rec = dict(episode=ep, **ep_info, **upd)

            if (ep + 1) % lc.eval_every == 0 or ep == lc.num_episodes - 1:
                ev = self.evaluate()
                rec.update({f"eval_{k}": v for k, v in ev.items()
                            if isinstance(v, (int, float, bool))})
                if on_eval is not None:
                    on_eval(ep, ev)
                print(f"ep {ep+1:4d} | ret {ep_info['ep_return']:8.1f} | cost {ep_info['ep_cost']:7.2f} "
                      f"| edge {ep_info['edge_frac']:.2f} | UUB {upd.get('uub', np.nan):+.3f} "
                      f"| mu {upd.get('mu', np.nan):.3f} | beta {upd.get('beta', np.nan):.3f} "
                      f"|| val obj {ev['avg_objective']:.3f} crit-out {ev['critical_outage_prob']:.3f} "
                      f"reg-viol {ev['regular_violation_prob']:.3f} | {time.time()-t0:5.0f}s",
                      flush=True)
            history.append(rec)
        return history

    def allocator(self) -> "LSACAllocator":
        return LSACAllocator(self)

    def evaluate(self, num_slots: Optional[int] = None, seed: int = 12345,
                 layouts: Optional[int] = None) -> dict:
        """
        Average of env.summary() over several user layouts (validation family by
        default). With random_critical the layouts are spread evenly over the
        critical counts, each with a set drawn from a generator fixed by the layout seed,
        so checkpoint selection is not tied to one critical set.
        """
        num_slots = num_slots or self.lcfg.eval_slots
        layouts = layouts or self.lcfg.eval_layouts
        allocator = self.allocator()
        runs = []
        if self.lcfg.random_critical:
            lo, hi = self.critical_range()
            counts = list(range(lo, hi + 1))
        for i in range(layouts):
            cfg = copy.copy(self.cfg)
            cfg.num_slots, cfg.seed = num_slots, seed + i
            if self.lcfg.random_critical:
                # spread the layouts evenly over the whole count range
                n = counts[int(round(i * (len(counts) - 1) / max(layouts - 1, 1)))]
                pick = np.random.default_rng(seed + i).choice(cfg.num_users, n, replace=False)
                cfg.critical_users = tuple(sorted(int(k) for k in pick))
            runs.append(run_episode(PowerAllocationEnv(cfg), allocator))
        out = {}
        for k, v in runs[0].items():
            if isinstance(v, (bool, np.bool_)):
                out[k] = float(np.mean([r[k] for r in runs]))
            elif isinstance(v, (int, float, np.floating, np.ndarray)):
                out[k] = np.mean([r[k] for r in runs], axis=0)
                if np.ndim(out[k]) == 0:
                    out[k] = float(out[k])
        return out

    def save(self, path: str):
        blob = {k: getattr(self, k).state_dict() for k in ("pi", "q1", "q2", "lc")}
        blob |= {"log_beta": self.log_beta.detach(), "log_mu": self.log_mu.detach()}
        torch.save(blob, path)

    def load(self, path: str):
        blob = torch.load(path, map_location=self.dev, weights_only=False)
        for k in ("pi", "q1", "q2", "lc"):
            getattr(self, k).load_state_dict(blob[k])
        with torch.no_grad():
            self.log_beta.copy_(blob["log_beta"])
            self.log_mu.copy_(blob["log_mu"])


class LSACAllocator(PowerAllocator):
    """Deterministic trained policy (tanh of the mean), usable with `run_episode`."""

    def __init__(self, agent: LSACAgent):
        self.agent = agent

    def allocate(self, state: SlotState) -> np.ndarray:
        return self.agent.mapper(self.agent.act(self.agent.encoder(state), deterministic=True))
