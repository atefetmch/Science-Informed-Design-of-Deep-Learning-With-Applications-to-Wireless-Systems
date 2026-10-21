import os
import sys

import numpy as np
from scipy.special import j0

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from poweralloc import (EqualPowerAllocator, PowerAllocationEnv, SystemConfig,
                        effective_gains, mrt_beamformers, rates, sinr)
from poweralloc.config import SPEED_OF_LIGHT


def random_channel(K=4, M=8, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) * 1e-4


def test_mrt_unit_norm_and_gains():
    H = random_channel()
    W = mrt_beamformers(H)
    assert np.allclose(np.linalg.norm(W, axis=0), 1.0)
    G = effective_gains(H, W)
    for k in range(H.shape[0]):
        for j in range(H.shape[0]):
            assert np.isclose(G[k, j], abs(np.vdot(H[k], W[:, j])) ** 2)
    # with MRT the own gain is ||h_k||^2
    assert np.allclose(np.diag(G), np.linalg.norm(H, axis=1) ** 2)


def test_sinr_matches_definition():
    H = random_channel()
    G = effective_gains(H, mrt_beamformers(H))
    p = np.array([1.0, 2.0, 0.5, 3.0])
    noise = 1e-9
    g = sinr(p, G, noise)
    for k in range(4):
        interf = sum(p[j] * G[k, j] for j in range(4) if j != k)
        assert np.isclose(g[k], p[k] * G[k, k] / (interf + noise))


def test_rates_unit():
    cfg = SystemConfig(num_users=4)
    H = random_channel()
    G = effective_gains(H, mrt_beamformers(H))
    p = np.full(4, cfg.p_max / 4)
    r = rates(p, G, cfg)
    assert np.allclose(r, cfg.bandwidth_hz * np.log2(1 + sinr(p, G, cfg.noise_power)) / 1e6)


def test_pathloss_and_correlation():
    cfg = SystemConfig(mobility_model="static", seed=3)
    env = PowerAllocationEnv(cfg)
    env.reset()
    ch = env.channel
    beta0 = 10 ** (cfg.beta0_db / 10)
    assert np.allclose(ch.beta, beta0 * (cfg.d0_m / ch.distances) ** cfg.pathloss_exp)
    fd = ch.speeds * cfg.carrier_freq_hz / SPEED_OF_LIGHT
    assert np.allclose(ch.rho, j0(2 * np.pi * fd * cfg.slot_duration_s))


def test_gauss_markov_fading_step():
    cfg = SystemConfig(mobility_model="static", seed=4)
    env = PowerAllocationEnv(cfg)
    env.reset()
    ch = env.channel
    h_prev = ch.h_tilde.copy()
    rng_copy = np.random.default_rng()
    rng_copy.bit_generator.state = ch.rng.bit_generator.state
    ch.step()
    K, M = cfg.num_users, cfg.num_antennas
    e = (rng_copy.standard_normal((K, M)) + 1j * rng_copy.standard_normal((K, M))) / np.sqrt(2)
    rho = ch.rho[:, None]
    assert np.allclose(ch.h_tilde, rho * h_prev + np.sqrt(1 - rho ** 2) * e)


def test_users_stay_in_annulus():
    cfg = SystemConfig(num_slots=3000, seed=5, v_max_mps=15.0, speed_std_mps=5.0)
    env = PowerAllocationEnv(cfg)
    env.reset()
    for _ in range(3000):
        env.channel.step()
        d = env.channel.distances
        assert np.all(d >= cfg.d_min_m - 1e-9) and np.all(d <= cfg.d_max_m + 1e-9)


def test_window_average_and_objective():
    cfg = SystemConfig(window_length=5, num_slots=20, seed=6)
    env = PowerAllocationEnv(cfg)
    alloc = EqualPowerAllocator(cfg)
    state, done = env.reset(), False
    while not done:
        state, info, done = env.step(alloc.allocate(state))
    R = np.array([d["rates"] for d in env.log])
    for t in range(len(env.log)):
        lo = max(0, t - cfg.window_length + 1)
        rbar = R[lo:t + 1].mean(axis=0)
        info = env.log[t]
        assert np.allclose(info["window_avg_rates"], rbar)
        obj = np.sum(np.log1p(rbar[cfg.regular_mask])) + cfg.tradeoff_weight * rbar.sum()
        assert np.isclose(info["objective"], obj)
        assert np.allclose(info["regular_violation"][cfg.regular_mask],
                           np.maximum(cfg.r_min - rbar, 0)[cfg.regular_mask])
        assert np.allclose(info["critical_violation"][cfg.critical_mask],
                           np.maximum(cfg.r_crit - R[t], 0)[cfg.critical_mask])


def test_window_available_denominator():
    cfg = SystemConfig(window_length=5, num_slots=10, seed=7)
    env = PowerAllocationEnv(cfg)
    alloc = EqualPowerAllocator(cfg)
    state, done = env.reset(), False
    while not done:
        state, info, done = env.step(alloc.allocate(state))
    R = np.array([d["rates"] for d in env.log])
    for t in range(len(env.log)):
        lo = max(0, t - cfg.window_length + 1)
        assert np.allclose(env.log[t]["window_avg_rates"], R[lo:t + 1].mean(axis=0))


def test_in_outage_tolerates_sitting_exactly_on_the_target():
    """A solver that sits exactly on the threshold must not count as in outage."""
    from poweralloc import OUTAGE_TOL, in_outage
    target = 15.0
    exact = np.full(500, target)
    assert not in_outage(exact, target).any()
    # values a hair under the target, the way a solver returns them
    jitter = target - np.random.default_rng(0).uniform(0, 1e-12, 500)
    assert not in_outage(jitter, target).any()
    # a real shortfall is still caught
    assert in_outage(np.array([target - 1e-6]), target).all()
    assert in_outage(np.array([target - OUTAGE_TOL * 10]), target).all()


def test_in_outage_matches_env_summary():
    """The shared helper must agree with what env.summary() reports."""
    from poweralloc import in_outage
    cfg = SystemConfig(window_length=10, num_slots=60, seed=3)
    env = PowerAllocationEnv(cfg)
    alloc = EqualPowerAllocator(cfg)
    state, done = env.reset(), False
    while not done:
        state, _, done = env.step(alloc.allocate(state))
    summary = env.summary()
    start = cfg.window_length - 1
    R = np.array([d["rates"] for d in env.log[start:]])
    cm = cfg.critical_mask
    assert np.isclose(in_outage(R[:, cm], cfg.r_crit[cm]).mean(),
                      summary["critical_outage_prob"])
