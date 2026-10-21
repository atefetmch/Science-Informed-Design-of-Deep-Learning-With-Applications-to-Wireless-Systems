import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from poweralloc import (EqualPowerAllocator, PowerAllocationEnv, PowerAllocationProblem,
                        SystemConfig, in_outage, rates)
from poweralloc.sca import SCAAllocator


def a_state(r_crit=10.0, seed=90000, slots=60, warm=55):
    """A slot state with a filled window, reached under equal power."""
    cfg = SystemConfig(num_slots=slots, seed=seed, default_r_crit=r_crit)
    env = PowerAllocationEnv(cfg)
    eq = EqualPowerAllocator(cfg)
    st = env.reset()
    for _ in range(warm):
        st, _, _ = env.step(eq.allocate(st))
    return cfg, st


def test_minorant_is_tight_at_the_expansion_point_and_below_elsewhere():
    cfg, st = a_state()
    sca = SCAAllocator(cfg)
    rng = np.random.default_rng(0)
    pn = rng.uniform(0.1, 1.0, cfg.num_users) * cfg.p_max / cfg.num_users
    _, In = sca._parts(pn, st.G)
    Rn, _ = sca._surrogate(pn, st.G, In)
    assert np.allclose(Rn, rates(pn, st.G, cfg), rtol=1e-10)
    for _ in range(200):
        p = rng.uniform(0, 1, cfg.num_users) * cfg.p_max / cfg.num_users
        R, _ = sca._surrogate(p, st.G, In)
        assert np.all(R <= rates(p, st.G, cfg) + 1e-9)


def test_minorant_gradient_matches_finite_differences():
    cfg, st = a_state()
    sca = SCAAllocator(cfg)
    pn = np.full(cfg.num_users, cfg.p_max / cfg.num_users)
    _, In = sca._parts(pn, st.G)
    p = pn * np.linspace(0.6, 1.4, cfg.num_users)
    _, J = sca._surrogate(p, st.G, In)
    h = 1e-7
    for j in range(cfg.num_users):
        e = np.zeros(cfg.num_users); e[j] = h
        fd = (sca._surrogate(p + e, st.G, In)[0] - sca._surrogate(p - e, st.G, In)[0]) / (2 * h)
        assert np.allclose(J[:, j], fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("r_crit", [10.0, 15.0])
def test_sca_is_feasible_within_budget_and_beats_equal_power(r_crit):
    cfg, st = a_state(r_crit=r_crit)
    sca = SCAAllocator(cfg)
    prob = PowerAllocationProblem(cfg)
    p = sca.allocate(st)
    assert np.all(p >= 0) and p.sum() <= cfg.p_max * (1 + 1e-12)
    ev = prob.evaluate(p, st)
    kc = list(cfg.critical_users)
    if np.min(sca._slack(p, st, sca._needs(st))) >= 0:
        assert not in_outage(ev["rates"][kc], cfg.r_crit[kc]).any()
    eq = prob.evaluate(np.full(cfg.num_users, cfg.p_max / cfg.num_users), st)
    eq_ok = not in_outage(eq["rates"][kc], cfg.r_crit[kc]).any()
    if eq_ok:     # equal power feasible here, so SCA starting from it cannot do worse
        assert ev["objective"] >= eq["objective"] - 1e-9


def test_sca_meets_critical_target_over_an_episode():
    cfg = SystemConfig(num_slots=120, seed=90003, default_r_crit=15.0)
    env = PowerAllocationEnv(cfg)
    sca = SCAAllocator(cfg)
    st, done, miss, n = env.reset(), False, 0, 0
    kc = list(cfg.critical_users)
    while not done:
        st, info, done = env.step(sca.allocate(st))
        miss += int(in_outage(info["rates"][kc], cfg.r_crit[kc]).any()); n += 1
    # the physical floor at 15 Mbps is ~0.4% of slots
    assert miss / n < 0.05


def test_sca_never_counts_as_an_outage_where_the_target_is_easy():
    """With one critical and one regular user every slot is feasible."""
    from poweralloc.lsac.rebuild import system_config
    cfg = system_config({"num_users": 2, "num_critical": 1, "r_crit": 13.0},
                        num_slots=300, seed=70000)
    env = PowerAllocationEnv(cfg)
    sca = SCAAllocator(cfg)
    st, done, miss = env.reset(), False, 0
    while not done:
        st, info, done = env.step(sca.allocate(st))
        miss += int(in_outage(info["rates"][:1], cfg.r_crit[:1]).any())
    assert miss == 0
