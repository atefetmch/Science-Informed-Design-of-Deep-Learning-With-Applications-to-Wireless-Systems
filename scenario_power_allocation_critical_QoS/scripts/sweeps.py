"""
Simulation settings (sweeps): training plan, evaluation and tables.

    python scripts/sweeps.py list
    python scripts/sweeps.py plan rcrit > cfg.txt        # one training line per run
    python scripts/sweeps.py evaluate rcrit --point 1 --runs runs
    python scripts/sweeps.py table rcrit --runs runs --out results/sweeps

Protocol, the same at every point:
  methods     Uniform PA (equal power), SCA, SAC and LSAC
  training    500 episodes, seeds 0-4; the checkpoint kept is the best on 10
              validation layouts x 1000 slots (seed-12345 family)
  test        50 layouts x 1000 slots from the seed-70000 family, the same for
              every method; the first W-1 slots of each layout are warm-up
  timing      wall-clock of allocate() only, one thread for every method
  violations  per slot: (c4) if any critical user misses R_crit, (c3) if any
              regular user's window average misses R_min
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

from poweralloc import EqualPowerAllocator, PowerAllocationEnv, in_outage
from poweralloc.problem import OUTAGE_TOL

# Learned methods: run-folder prefix and trainer arguments
METHODS = {"LSAC": ("", "--method lsac"), "SAC": ("sac_", "--method sac")}
SEEDS = (0, 1, 2, 3, 4)

SWEEPS = {
    "rcrit": {
        "fixed": {"num_users": 6, "num_critical": 2, "p_max_dbm": 40.0},
        "points": [{"x": r, "r_crit": float(r)} for r in (8, 10, 12, 15, 18)],
    },
    "rmin": {
        "fixed": {"num_users": 6, "num_critical": 2, "r_crit": 12.0, "p_max_dbm": 40.0},
        "points": [{"x": r, "r_min": float(r)} for r in (5, 8, 11, 14, 17, 20)],
    },
    # number of critical users N_c with K = 6 and M = 8
    "ncrit": {
        "fixed": {"num_users": 6, "r_crit": 12.0, "p_max_dbm": 40.0, "num_antennas": 8},
        "points": [{"x": n, "num_critical": n} for n in (1, 2, 3, 4, 5)],
    },
    # heavy load: 3 critical users among K users, M = 18
    "heavy": {
        "fixed": {"num_critical": 3, "r_crit": 12.0, "p_max_dbm": 40.0, "num_antennas": 18},
        "points": [{"x": K, "num_users": K} for K in (12, 16)],
    },
    # optional: one policy per K for any critical set (random critical set per episode)
    "anycrit": {
        "fixed": {"r_crit": 12.0, "p_max_dbm": 40.0, "random_critical": 1},
        "points": [{"x": 6, "num_users": 6, "num_critical": 2, "num_antennas": 8},
                   {"x": 16, "num_users": 16, "num_critical": 8, "num_antennas": 24}],
    },
    # critical and regular users split equally
    "users": {
        "fixed": {"r_crit": 12.0, "p_max_dbm": 40.0},
        "points": [{"x": K, "num_users": K, "num_critical": K // 2, "num_antennas": M}
                   for K, M in ((2, 8), (4, 8), (6, 8), (8, 12), (12, 16), (16, 24))],
    },
}

TEST_SEED, TEST_LAYOUTS, TEST_SLOTS = 70000, 50, 1000

FLAG = {"num_users": "--num-users", "num_critical": "--num-critical", "r_crit": "--r-crit",
        "r_min": "--r-min", "p_max_dbm": "--p-max-dbm", "num_antennas": "--num-antennas",
        "random_critical": "--random-critical"}


def point_args(sweep: str, i: int) -> dict:
    s = SWEEPS[sweep]
    pt = {k: v for k, v in s["points"][i].items() if k != "x"}
    return {**s["fixed"], **pt}


def tag(sweep: str, x, seed: int, method: str = "LSAC") -> str:
    return f"{METHODS[method][0]}sw_{sweep}_{x}_seed{seed}"


def eval_path(runs, sweep, x):
    return os.path.join(runs, f"eval_{sweep}_{x}.json")


def system_cfg(args: dict, **over):
    from poweralloc.lsac.rebuild import system_config
    return system_config(args, **over)


def rollout(make_alloc, args: dict, layouts: int, slots: int, seed: int) -> dict:
    """
    Per-slot outcomes for one method on the test layouts.

    Returns per-slot and per-(slot, user) constraint rates, the objective and the
    decision time; the per-(slot, user) rates are checked against env.summary().
    """
    crit_slot, reg_slot, crit_pair, reg_pair, obj, allok = [], [], [], [], [], []
    spent, n_dec = 0.0, 0
    for i in range(layouts):
        cfg = system_cfg(args, num_slots=slots, seed=seed + i)
        kc, kr = cfg.critical_mask, cfg.regular_mask
        alloc = make_alloc(cfg)
        env = PowerAllocationEnv(cfg)
        st, done, t = env.reset(), False, 0
        while not done:
            t0 = time.perf_counter()
            p = alloc.allocate(st)
            spent += time.perf_counter() - t0
            n_dec += 1
            st, info, done = env.step(p)
            if t >= cfg.window_length - 1:                   # after warm-up
                cm = in_outage(info["rates"][kc], cfg.r_crit[kc]) if kc.any() else np.zeros(0, bool)
                rm = (info["regular_violation"][kr] > OUTAGE_TOL) if kr.any() else np.zeros(0, bool)
                # a group with no users has no rate, not a 0% rate
                if kc.any():
                    crit_slot.append(cm.any())
                if kr.any():
                    reg_slot.append(rm.any())
                crit_pair.extend(cm.tolist()); reg_pair.extend(rm.tolist())
                obj.append(info["objective"]); allok.append(bool(info["feasible"]))
            t += 1
        summ = env.summary()
        start = cfg.window_length - 1
        n = len(env.log) - start
        lay_crit = np.mean(crit_pair[-n * kc.sum():]) if kc.any() else 0.0
        if kc.any():
            assert np.isclose(lay_crit, summ["critical_outage_prob"]), "critical rate disagrees"
        if kr.any():
            assert np.isclose(np.mean(reg_pair[-n * kr.sum():]),
                              summ["regular_violation_prob"]), "regular rate disagrees"
    pct = lambda v: 100.0 * float(np.mean(v)) if len(v) else float("nan")
    return {"crit_slot_pct": pct(crit_slot), "reg_slot_pct": pct(reg_slot),
            "crit_pair_pct": pct(crit_pair), "reg_pair_pct": pct(reg_pair),
            "allok_pct": pct(allok), "objective": float(np.mean(obj)),
            "ms_per_slot": 1e3 * spent / n_dec, "slots": len(obj)}


def cmd_list(_):
    for name, s in SWEEPS.items():
        print(f"{name:6} x = {[p['x'] for p in s['points']]}   fixed {s['fixed']}")


def cmd_count(a):
    print(len(SWEEPS[a.sweep]["points"]))


def cmd_plan(a):
    for m, (_, margs) in METHODS.items():
        for i, p in enumerate(SWEEPS[a.sweep]["points"]):
            extra = " ".join(f"{FLAG[k]} {v}" for k, v in point_args(a.sweep, i).items())
            for s in SEEDS:
                t = tag(a.sweep, p["x"], s, m)
                if getattr(a, "missing", False) and os.path.exists(
                        os.path.join(getattr(a, "runs", "runs"), t, "result.json")):
                    continue
                print(f"--tag {t} --seed {s} {extra} {margs}")


def cmd_evaluate(a):
    import torch
    torch.set_num_threads(1)
    from poweralloc.lsac.rebuild import agent_from_run
    from poweralloc.sca import SCAAllocator

    p = SWEEPS[a.sweep]["points"][a.point]
    args = point_args(a.sweep, a.point)
    out = {"sweep": a.sweep, "x": p["x"], "args": args,
           "protocol": {"test_seed": TEST_SEED, "layouts": a.layouts, "slots": a.slots}}
    for name, make in (("equal power", EqualPowerAllocator), ("SCA", SCAAllocator)):
        out[name] = rollout(lambda c: make(c), args, a.layouts, a.slots, TEST_SEED)
        print(name, json.dumps(out[name]), flush=True)
    for m in METHODS:
        out[m] = []
        for s in SEEDS:
            d = os.path.join(a.runs, tag(a.sweep, p["x"], s, m))
            saved = json.load(open(os.path.join(d, "result.json")))
            # the run must be this point's: run folders are named by x only
            bad = {k: (saved["args"].get(k), v) for k, v in args.items()
                   if saved["args"].get(k) is None or not np.isclose(float(saved["args"][k]), float(v))}
            if bad:
                raise SystemExit(f"{d} was trained with different settings (recorded, wanted): {bad}")
            rec = {"seed": s, "best_episode": saved["best_episode"]}
            agent = agent_from_run(saved["args"], os.path.join(d, "best.pt"))
            rec["best"] = rollout(lambda c: agent.allocator(), args, a.layouts, a.slots, TEST_SEED)
            out[m].append(rec)
            print(f"{m} seed {s} best ep {rec['best_episode']}", json.dumps(rec["best"]), flush=True)
    json.dump(out, open(eval_path(a.runs, a.sweep, p["x"]), "w"), indent=1)
    print("wrote", eval_path(a.runs, a.sweep, p["x"]))


def cmd_table(a):
    keys = ("crit_slot_pct", "reg_slot_pct", "allok_pct", "objective")
    hdr = (f"{'x':>6} {'method':12} {'crit %slot':>14} {'reg %slot':>14} "
           f"{'all-ok %':>14} {'objective':>14} {'ms/slot':>9}")
    lines = [hdr]
    for p in SWEEPS[a.sweep]["points"]:
        r = json.load(open(eval_path(a.runs, a.sweep, p["x"])))
        for m in ("equal power", "SCA"):
            d = r[m]
            lines.append(f"{r['x']:>6} {'Uniform PA' if m == 'equal power' else m:12} "
                         + " ".join(f"{d[k]:14.2f}" for k in keys) + f" {d['ms_per_slot']:9.3f}")
        for m in METHODS:
            v = {k: np.array([s["best"][k] for s in r[m]]) for k in keys + ("ms_per_slot",)}
            lines.append(f"{r['x']:>6} {m:12} "
                         + " ".join(f"{v[k].mean():7.2f}+/-{v[k].std(ddof=1):4.2f}" for k in keys)
                         + f" {v['ms_per_slot'].mean():9.3f}")
    print("\n".join(lines))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, f"table_{a.sweep}.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("plan", "count", "evaluate", "table"):
        p = sub.add_parser(name)
        p.add_argument("sweep", choices=list(SWEEPS))
        if name == "evaluate":
            p.add_argument("--point", type=int, required=True)
            p.add_argument("--runs", default="runs")
            p.add_argument("--layouts", type=int, default=TEST_LAYOUTS)
            p.add_argument("--slots", type=int, default=TEST_SLOTS)
        if name == "table":
            p.add_argument("--runs", default="runs")
            p.add_argument("--out", default=None)
        if name == "plan":
            p.add_argument("--runs", default="runs")
            p.add_argument("--missing", action="store_true",
                           help="only the runs that have no result.json under --runs yet")
    a = ap.parse_args()
    {"list": cmd_list, "plan": cmd_plan, "count": cmd_count, "evaluate": cmd_evaluate,
     "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
