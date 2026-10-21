import json
import os
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import sweeps
from poweralloc import EqualPowerAllocator
from poweralloc.lsac import LSACAgent, LSACConfig
from poweralloc.lsac.rebuild import lsac_config, system_config


@pytest.mark.parametrize("name", list(sweeps.SWEEPS))
def test_every_planned_run_builds_the_intended_system(name, capsys):
    """Each training line, parsed by the real trainer, yields that point's system."""
    import train_lsac
    sweeps.cmd_plan(type("A", (), {"sweep": name})())
    lines = capsys.readouterr().out.strip().splitlines()
    pts = sweeps.SWEEPS[name]["points"]
    assert len(lines) == len(pts) * len(sweeps.SEEDS) * len(sweeps.METHODS)
    seen = set()
    for ln in lines:
        args = train_lsac.build_parser().parse_args(shlex.split(ln))
        seen.add(args.tag)
        cfg, lcfg = system_config(vars(args)), lsac_config(vars(args))
        method = "SAC" if args.tag.startswith("sac_") else "LSAC"
        i = next(j for j, p in enumerate(pts) if sweeps.tag(name, p["x"], args.seed, method) == args.tag)
        want = sweeps.point_args(name, i)
        assert cfg.num_users == want.get("num_users", 6)
        assert len(cfg.critical_users) == want.get("num_critical", 2)
        assert cfg.r_crit[0] == want.get("r_crit", 10.0)
        assert cfg.r_min[-1] == want.get("r_min", 5.0)
        assert np.isclose(cfg.p_max_dbm, want.get("p_max_dbm", 40.0))
        assert cfg.num_antennas == want.get("num_antennas", 8)
        assert lcfg.lyapunov == (method == "LSAC")
    assert len(seen) == len(lines)                     # tags are unique


def test_users_sweep_splits_equally():
    for p in sweeps.SWEEPS["users"]["points"]:
        assert p["num_critical"] == p["num_users"] // 2


@pytest.mark.parametrize("name", list(sweeps.SWEEPS))
def test_every_point_has_more_antennas_than_users(name):
    for i in range(len(sweeps.SWEEPS[name]["points"])):
        a = sweeps.point_args(name, i)
        assert a.get("num_antennas", 8) > a.get("num_users", 6)


def test_rollout_agrees_with_env_summary_and_is_sane():
    """rollout() asserts per-(slot, user) agreement with env.summary() itself."""
    args = {"num_users": 4, "num_critical": 2, "r_crit": 13.0, "p_max_dbm": 40.0}
    d = sweeps.rollout(lambda c: EqualPowerAllocator(c), args, layouts=2, slots=80, seed=70000)
    assert d["slots"] == 2 * (80 - 49)
    assert 0 <= d["crit_pair_pct"] <= d["crit_slot_pct"] <= 100
    assert 0 <= d["allok_pct"] <= 100 and d["ms_per_slot"] >= 0


def fake_runs(root, name, i, args=None):
    """An untrained checkpoint per method and seed, recorded like the trainer does."""
    p = sweeps.SWEEPS[name]["points"][i]
    args = args or sweeps.point_args(name, i)
    for m in sweeps.METHODS:
        for s in sweeps.SEEDS:
            d = root / sweeps.tag(name, p["x"], s, m)
            d.mkdir()
            rec = {**args, "seed": s, "method": m.lower()}
            agent = LSACAgent(system_config(rec), lsac_config(rec))
            agent.save(str(d / "best.pt"))
            json.dump({"args": rec, "best_episode": 124}, open(d / "result.json", "w"))
    return p


def run_sweeps(*a):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts/sweeps.py"), *a],
                          capture_output=True, text=True, env=env)


def test_evaluate_and_table_end_to_end(tmp_path):
    name, i = "users", 0                               # K = 2
    p = fake_runs(tmp_path, name, i)
    r = run_sweeps("evaluate", name, "--point", str(i), "--runs", str(tmp_path),
                   "--layouts", "1", "--slots", "60")
    assert r.returncode == 0, r.stderr
    ev = json.load(open(tmp_path / f"eval_{name}_{p['x']}.json"))
    assert set(ev) >= {"equal power", "SCA", "LSAC", "SAC"}
    assert all(len(ev[m]) == len(sweeps.SEEDS) for m in sweeps.METHODS)
    # table needs every point of the sweep
    for x in [q["x"] for q in sweeps.SWEEPS[name]["points"]][1:]:
        json.dump({**ev, "x": x}, open(tmp_path / f"eval_{name}_{x}.json", "w"))
    r = run_sweeps("table", name, "--runs", str(tmp_path), "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "Uniform PA" in (tmp_path / f"table_{name}.txt").read_text()


def test_evaluate_refuses_a_run_trained_with_other_settings(tmp_path):
    name, i = "users", 0
    fake_runs(tmp_path, name, i, args={**sweeps.point_args(name, i), "r_crit": 10.0})
    r = run_sweeps("evaluate", name, "--point", str(i), "--runs", str(tmp_path),
                   "--layouts", "1", "--slots", "60")
    assert r.returncode != 0 and "different settings" in (r.stdout + r.stderr)
