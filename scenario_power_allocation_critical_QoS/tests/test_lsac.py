import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from poweralloc import EqualPowerAllocator, PowerAllocationEnv, PowerAllocationProblem, SystemConfig
from poweralloc.lsac import ActionMapper, LSACAgent, LSACConfig, StateEncoder, qos_cost
from poweralloc.lsac.networks import GaussianPolicy, LyapunovCritic
from poweralloc.lsac.rebuild import lsac_config


def test_action_mapper_keeps_power_feasible():
    cfg = SystemConfig()
    mapper = ActionMapper(cfg)
    rng = np.random.default_rng(0)
    for u in [rng.uniform(-1, 1, mapper.dim) for _ in range(200)] + [
            -np.ones(mapper.dim), np.ones(mapper.dim)]:
        p = mapper(u)
        assert p.shape == (cfg.num_users,)
        assert np.all(p >= 0)
        assert p.sum() <= cfg.p_max * (1 + 1e-12)


def test_cost_is_max_critical_shortfall():
    cfg = SystemConfig(num_users=4, critical_users=(0, 2))
    info = {"critical_violation": np.array([1.5, 0.0, 3.0, 0.0])}
    assert qos_cost(info, cfg) == 3.0
    info = {"critical_violation": np.zeros(4)}
    assert qos_cost(info, cfg) == 0.0


def test_cost_ignores_regular_users():
    cfg = SystemConfig(seed=0, num_slots=5)
    env = PowerAllocationEnv(cfg)
    state = env.reset()
    _, info, _ = env.step(np.zeros(cfg.num_users))      # nobody served
    assert np.all(info["regular_violation"][cfg.regular_mask] > 0)
    assert qos_cost(info, cfg) == max(cfg.r_crit[cfg.critical_mask])


def test_state_encoder_layout():
    cfg = SystemConfig(window_length=4, num_slots=10, seed=1)
    enc = StateEncoder(cfg)
    env = PowerAllocationEnv(cfg)
    alloc = EqualPowerAllocator(cfg)
    K = cfg.num_users
    state = env.reset()
    v = enc(state)
    assert v.shape == (enc.dim,) == (K * K + 5 * K + 3 * K + 1,)
    assert v[-1] == 0.0
    past = []
    for _ in range(5):
        state, info, _ = env.step(alloc.allocate(state))
        past.append(info["rates"])
    v = enc(state)
    hist = v[K * K + 5 * K:-1].reshape(3, K)
    assert np.allclose(hist, np.array(past[-3:]) / enc.qos_ref, atol=1e-5)
    mob = v[K * K:K * K + 5 * K].reshape(K, 5)
    assert np.allclose(mob[:, 0:2], state.positions / cfg.d_max_m, atol=1e-6)
    assert np.allclose(mob[:, 3] ** 2 + mob[:, 4] ** 2, 1.0, atol=1e-5)
    assert v[-1] == 1.0


def test_policy_logprob_matches_tanh_gaussian():
    torch.manual_seed(0)
    pi = GaussianPolicy(5, 3, (16,))
    s = torch.randn(64, 5)
    torch.manual_seed(1)
    a, logp = pi(s)
    mu, log_std = pi.body(s).chunk(2, dim=-1)
    log_std = log_std.clamp(-20, 2)
    z = torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6))
    base = torch.distributions.Normal(mu, log_std.exp()).log_prob(z).sum(-1)
    ref = base - torch.log(1 - a ** 2 + 1e-12).sum(-1)
    assert torch.allclose(logp, ref, atol=1e-2)


def test_lyapunov_critic_nonnegative():
    lc = LyapunovCritic(5, 3, (16, 16))
    assert torch.all(lc(torch.randn(100, 5), torch.randn(100, 3)) >= 0)


def small(**kw):
    cfg = SystemConfig(window_length=10, default_r_crit=15.0, seed=3)
    lcfg = LSACConfig(hidden=(32, 32), batch_size=32, episode_len=40, num_episodes=3,
                      random_episodes=1, eval_every=99, eval_slots=30, eval_layouts=1, **kw)
    return LSACAgent(cfg, lcfg)


def params(net):
    return [p.detach().clone() for p in net.parameters()]


def test_short_training_run():
    agent = small()
    hist = agent.train()
    assert len(hist) == 3
    assert "uub" in hist[-1] and np.isfinite(hist[-1]["loss_q"])
    assert agent.mu.item() <= agent.lcfg.max_lyap_multiplier * (1 + 1e-6)
    assert agent.evaluate(num_slots=30, layouts=1)["power_feasible"] == 1.0


def test_lsac_trains_the_lyapunov_critic():
    agent = small()
    lc0 = params(agent.lc)
    agent.train()
    assert not all(torch.equal(a, b) for a, b in zip(lc0, params(agent.lc)))


def test_sac_never_touches_the_lyapunov_critic_or_multiplier():
    agent = small(lyapunov=False)
    lc0, mu0 = params(agent.lc), agent.mu.item()
    agent.train()
    assert all(torch.equal(a, b) for a, b in zip(lc0, params(agent.lc)))
    assert agent.mu.item() == mu0


def test_only_the_sac_baseline_has_the_c4_penalty_in_its_reward():
    lsac, sac = small(), small(lyapunov=False)
    cfg = lsac.cfg
    env = PowerAllocationEnv(cfg)
    env.reset()
    p = np.zeros(cfg.num_users); p[-1] = cfg.p_max          # starve the critical users
    _, info, _ = env.step(p)
    rm, cm = cfg.regular_mask, cfg.critical_mask
    c3 = 50.0 * np.sum(info["regular_violation"][rm] / cfg.r_min[rm])
    c4 = 200.0 * np.max(info["critical_violation"][cm] / cfg.r_crit[cm])
    assert np.isclose(lsac.reward(info), info["objective"] - c3)
    assert np.isclose(sac.reward(info), info["objective"] - c3 - c4)


def test_rebuild_maps_method():
    assert lsac_config({"method": "lsac"}).lyapunov
    assert not lsac_config({"method": "sac"}).lyapunov


def test_action_mapper_never_exceeds_budget_from_float32_policy_output():
    """float32 policy outputs must not push the total over P_max through rounding."""
    from poweralloc import PowerAllocationProblem
    cfg = SystemConfig()
    m = ActionMapper(cfg)
    prob = PowerAllocationProblem(cfg)
    rng = np.random.default_rng(3)
    for u in rng.uniform(-1, 1, (20000, m.dim)).astype(np.float32):
        p = m(u)
        assert p.dtype == np.float64 and np.all(p >= 0)
        assert p.sum() <= cfg.p_max * (1 + 1e-12)
        assert prob.power_violation(p)["budget"] <= 1e-9


# ---- safety margin on the critical constraint (training only) ----------------

def _slot_info(cfg, p=None):
    env = PowerAllocationEnv(cfg)
    env.reset()
    _, info, _ = env.step(np.full(cfg.num_users, cfg.p_max / cfg.num_users) if p is None else p)
    return info


def test_zero_margin_gives_the_true_cost():
    from poweralloc.lsac.cost import critical_shortfall
    for seed in range(20):
        cfg = SystemConfig(seed=seed, default_r_crit=12.0)
        info = _slot_info(cfg)
        assert np.array_equal(critical_shortfall(info, cfg, 0.0), info["critical_violation"])
        assert qos_cost(info, cfg, 0.0) == qos_cost(info, cfg)


def test_margin_tightens_the_critical_target_only():
    from poweralloc.lsac.cost import critical_shortfall
    cfg = SystemConfig(seed=0, default_r_crit=12.0)
    info = _slot_info(cfg)
    for delta in (0.5, 1.0, 2.0):
        v = critical_shortfall(info, cfg, delta)
        want = np.maximum(cfg.r_crit + delta - info["rates"], 0.0)
        assert np.allclose(v[cfg.critical_mask], want[cfg.critical_mask])
        assert np.all(v[cfg.regular_mask] == 0.0)
        assert qos_cost(info, cfg, delta) >= qos_cost(info, cfg)
    # a user just above R_crit has zero true cost but a positive tightened cost
    r = info["rates"].copy(); r[0] = cfg.r_crit[0] + 0.3
    fake = {**info, "rates": r,
            "critical_violation": np.where(cfg.critical_mask, np.maximum(cfg.r_crit - r, 0), 0.0)}
    assert critical_shortfall(fake, cfg, 0.0)[0] == 0.0
    assert np.isclose(critical_shortfall(fake, cfg, 1.0)[0], 0.7)


def test_margin_enters_the_sac_penalty_but_not_the_lsac_reward():
    lsac0, lsac1 = small(), small(crit_margin=2.0)
    sac0, sac1 = small(lyapunov=False), small(lyapunov=False, crit_margin=2.0)
    cfg = lsac0.cfg
    info = _slot_info(cfg)
    cm = cfg.critical_mask
    assert lsac0.reward(info) == lsac1.reward(info)          # LSAC: margin acts via the cost
    base = info["objective"] - 50.0 * np.sum(info["regular_violation"][cfg.regular_mask]
                                             / cfg.r_min[cfg.regular_mask])
    tight = np.max(np.maximum(cfg.r_crit + 2.0 - info["rates"], 0)[cm] / cfg.r_crit[cm])
    assert np.isclose(sac1.reward(info), base - 200.0 * tight)
    assert sac1.reward(info) <= sac0.reward(info)


def test_margin_reaches_the_stored_lyapunov_cost():
    agent = small(crit_margin=3.0)
    env = PowerAllocationEnv(agent.cfg)
    agent.collect_episode(env, random_actions=True)
    log = env.log[:-1]                                     # the last slot is not stored
    want = [np.max(np.maximum(agent.cfg.r_crit + 3.0 - d["rates"], 0)[agent.cfg.critical_mask])
            for d in log]
    assert np.allclose(agent.D.c[:len(log)], want, atol=1e-5)


def test_trainer_records_and_rebuild_restores_the_margin():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_lsac.py"
    spec = importlib.util.spec_from_file_location("train_lsac", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    a = mod.build_parser().parse_args(["--crit-margin", "1.5", "--method", "sac"])
    assert lsac_config(vars(a)).crit_margin == 1.5
    b = mod.build_parser().parse_args([])
    assert lsac_config(vars(b)).crit_margin == 0.0           # default: no margin


# ---- variable critical set: per-user flag + random critical sets ---------------

def flag_agent(**kw):
    cfg = SystemConfig(window_length=10, default_r_crit=12.0, seed=3)
    lcfg = LSACConfig(hidden=(32, 32), batch_size=32, episode_len=40, num_episodes=3,
                      random_episodes=1, eval_every=99, eval_slots=30, eval_layouts=4,
                      random_critical=True, **kw)
    return LSACAgent(cfg, lcfg)


def test_state_carries_the_current_critical_set():
    import copy
    cfg = copy.copy(SystemConfig(seed=0))
    env = PowerAllocationEnv(cfg)
    assert np.array_equal(env.reset().critical_mask, cfg.critical_mask)
    cfg.critical_users = (1, 4, 5)
    st = env.reset()
    assert np.array_equal(np.flatnonzero(st.critical_mask), [1, 4, 5])


def test_encoder_without_flag_is_unchanged_and_ignores_the_slot_set():
    cfg = SystemConfig(window_length=4, num_slots=10, seed=1)
    enc = StateEncoder(cfg)
    env = PowerAllocationEnv(cfg)
    st = env.reset()
    for _ in range(4):
        st, _, _ = env.step(np.full(cfg.num_users, cfg.p_max / cfg.num_users))
    v = enc(st)
    assert v.shape == (cfg.num_users ** 2 + 5 * cfg.num_users + 3 * cfg.num_users + 1,)
    import dataclasses
    other = dataclasses.replace(st, critical_mask=np.ones(cfg.num_users, bool))
    assert np.array_equal(enc(other), v)          # roles stay fixed by index


def test_encoder_flag_reports_the_set_and_normalises_by_role():
    import dataclasses
    cfg = SystemConfig(window_length=4, num_slots=10, seed=1, default_r_crit=12.0, default_r_min=5.0)
    enc = StateEncoder(cfg, critical_flag=True)
    K = cfg.num_users
    assert enc.dim == K * K + 5 * K + 3 * K + 1 + K
    env = PowerAllocationEnv(cfg)
    st = env.reset()
    past = []
    for _ in range(4):
        st, info, _ = env.step(np.full(K, cfg.p_max / K)); past.append(info["rates"])
    mask = np.zeros(K, bool); mask[[2, 5]] = True
    v = enc(dataclasses.replace(st, critical_mask=mask))
    assert np.array_equal(v[-K:], mask.astype(np.float32))
    hist = v[K * K + 5 * K:K * K + 5 * K + 3 * K].reshape(3, K)
    ref = np.where(mask, 12.0, 5.0)
    assert np.allclose(hist, np.array(past[-3:]) / ref, atol=1e-5)


def test_random_critical_draws_sets_and_uses_them_everywhere():
    import copy
    agent = flag_agent()
    cfg0 = agent.cfg.critical_users
    env_cfg = copy.copy(agent.cfg); env_cfg.num_slots = 40
    env = PowerAllocationEnv(env_cfg)
    seen = set()
    for _ in range(40):
        n0 = agent.D.n
        info = agent.collect_episode(env, random_actions=True)
        cs = env.cfg.critical_users
        seen.add(cs)
        assert 1 <= len(cs) <= agent.cfg.num_users - 1 and info["num_critical"] == len(cs)
        # stored costs = max shortfall over THIS episode's critical set
        log = env.log[:-1]
        want = [np.max(np.maximum(env_cfg.r_crit - d["rates"], 0)[list(cs)]) for d in log]
        assert np.allclose(agent.D.c[n0:n0 + len(log)], want, atol=1e-5)
        assert np.array_equal(np.flatnonzero(agent.D.s[n0, -agent.cfg.num_users:]), list(cs))
    assert agent.cfg.critical_users == cfg0           # the agent's own config untouched
    assert len(seen) > 10 and len({len(c) for c in seen}) > 2


def test_random_critical_refuses_to_mutate_the_agent_config():
    agent = flag_agent()
    with pytest.raises(ValueError):
        agent.collect_episode(PowerAllocationEnv(agent.cfg), random_actions=True)


def test_reward_follows_the_given_critical_set():
    import copy
    agent = small(lyapunov=False)
    cfg = copy.copy(agent.cfg); cfg.critical_users = (3,)
    info = _slot_info(agent.cfg)
    p = PowerAllocationProblem(cfg)       # recompute violations for the other set
    env = PowerAllocationEnv(cfg); st = env.reset()
    _, info2, _ = env.step(np.full(cfg.num_users, cfg.p_max / cfg.num_users))
    base = info2["objective"] - 50.0 * np.sum(info2["regular_violation"][cfg.regular_mask]
                                              / cfg.r_min[cfg.regular_mask])
    pen = 200.0 * max(cfg.r_crit[3] - info2["rates"][3], 0) / cfg.r_crit[3]
    assert np.isclose(agent.reward(info2, cfg), base - pen)


def test_flag_agent_trains_validates_over_counts_and_round_trips(tmp_path):
    agent = flag_agent()
    hist = agent.train()
    assert len(hist) == 3 and np.isfinite(hist[-1]["loss_q"])
    ev = agent.evaluate(num_slots=30, layouts=4)
    assert ev["power_feasible"] == 1.0
    path = str(tmp_path / "a.pt"); agent.save(path)
    args = {"method": "lsac", "random_critical": 1, "seed": 0}
    lcfg = lsac_config(args); lcfg.hidden = agent.lcfg.hidden
    b = LSACAgent(agent.cfg, lcfg); b.load(path)
    assert b.encoder.critical_flag and b.encoder.dim == agent.encoder.dim
    st = PowerAllocationEnv(agent.cfg).reset()
    assert np.allclose(agent.act(agent.encoder(st), True), b.act(b.encoder(st), True))


def test_rebuild_maps_random_critical_and_defaults_stay_off():
    assert lsac_config({"random_critical": 1}).critical_flag
    assert lsac_config({"random_critical": "1"}).random_critical
    off = lsac_config({})
    assert not off.critical_flag and not off.random_critical


# ---- mu update rule, alpha3, cap, SAC penalty weight -------------------------

def test_sgd_mu_update_is_proportional_and_bounded():
    agent = small(mu_update="sgd", lr_mu_sgd=10.0, max_lyap_multiplier=1000.0)
    agent.train()                       # fills the buffers
    for target_uub, check in ((0.5, "up"), (-0.5, "down")):
        mu0 = agent.mu.item()
        # emulate one dual step exactly as update() does
        with torch.no_grad():
            new = (agent.log_mu.exp() + agent.lcfg.lr_mu_sgd * target_uub).clamp(
                min=agent.lcfg.min_lyap_multiplier, max=agent.lcfg.max_lyap_multiplier)
        want = float(np.clip(mu0 + 10.0 * target_uub, 1e-6, 1000.0))
        assert np.isclose(new.item(), want, rtol=1e-5)
    # through update(): mu stays within its bounds
    for _ in range(50):
        agent.update()
        assert agent.lcfg.min_lyap_multiplier * 0.999 <= agent.mu.item() <= 1000.0 * 1.001


def test_sgd_mu_moves_with_the_sign_of_uub():
    agent = small(mu_update="sgd", lr_mu_sgd=5.0, max_lyap_multiplier=1e6)
    agent.train()
    mu_before = agent.mu.item()
    log = agent.update()
    step = agent.mu.item() - mu_before
    if mu_before > 1e-5:                # not sitting on the floor
        assert np.sign(step) == np.sign(log["uub"]) or abs(log["uub"]) < 1e-12
        assert np.isclose(step, 5.0 * log["uub"], rtol=1e-3, atol=1e-6)


def test_alpha3_enters_the_uub_term():
    torch.manual_seed(0)
    a1, a2 = small(alpha3=1.0), small(alpha3=0.1)
    a1.train()
    a2.load_state = None
    # same networks and same batch -> uub differs only through alpha3 * c
    a2.pi.load_state_dict(a1.pi.state_dict()); a2.lc.load_state_dict(a1.lc.state_dict())
    be = a1.D_edge.sample(64, a1.dev) if a1.D_edge.n else None
    if be is None:
        return
    with torch.no_grad():
        a2e, _ = a1.pi(be["s2"], with_logprob=False)
        nxt = a1.lc(be["s2"], a2e) * be["e"]
        now = a1.lc(be["s"], be["a"]) * be["e"]
        u1 = (nxt - (now - 1.0 * be["c"] * be["e"])).mean()
        u2 = (nxt - (now - 0.1 * be["c"] * be["e"])).mean()
    assert np.isclose((u1 - u2).item(), 0.9 * (be["c"] * be["e"]).mean().item(), atol=1e-5)


def test_cli_options_reach_the_config_and_defaults_are_unchanged():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_lsac.py"
    spec = importlib.util.spec_from_file_location("train_lsac_cli", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    a = mod.build_parser().parse_args(["--mu-update", "sgd", "--mu-lr", "2", "--mu-cap", "1000",
                                       "--alpha3", "0.5", "--crit-weight", "800"])
    c = lsac_config(vars(a))
    assert (c.mu_update, c.lr_mu_sgd, c.max_lyap_multiplier, c.alpha3, c.crit_weight) == \
           ("sgd", 2.0, 1000.0, 0.5, 800.0)
    d = lsac_config(vars(mod.build_parser().parse_args([])))
    assert (d.mu_update, d.max_lyap_multiplier, d.alpha3, d.crit_weight) == ("adam", 100.0, 1.0, 200.0)


def test_crit_weight_scales_the_sac_penalty():
    s1, s2 = small(lyapunov=False), small(lyapunov=False, crit_weight=800.0)
    info = _slot_info(s1.cfg)
    cm = s1.cfg.critical_mask
    pen = np.max(info["critical_violation"][cm] / s1.cfg.r_crit[cm])
    assert np.isclose(s1.reward(info) - s2.reward(info), 600.0 * pen)


def test_bad_mu_update_is_rejected():
    with pytest.raises(ValueError):
        small(mu_update="rmsprop")
