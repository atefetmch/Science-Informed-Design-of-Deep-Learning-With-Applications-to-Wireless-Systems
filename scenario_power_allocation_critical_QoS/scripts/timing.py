"""
Decision time per slot, measured fairly: every method and every sweep point in
one process, on one node, on one CPU thread.

    python scripts/timing.py users --runs runs --out results/timing/timing_users.json

Only the allocate() call is timed (for LSAC: state encoding, one policy forward
pass and the action-to-power mapping; for SCA: every convex subproblem). Methods
are interleaved layout by layout and the first WARMUP calls of every rollout are
discarded.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import platform
import sys
import time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweeps
from poweralloc import EqualPowerAllocator, PowerAllocationEnv

WARMUP = 50


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def times_for(make_alloc, cfg) -> np.ndarray:
    alloc = make_alloc(cfg)
    env = PowerAllocationEnv(cfg)
    st, done, out = env.reset(), False, []
    while not done:
        t0 = time.perf_counter()
        p = alloc.allocate(st)
        out.append(time.perf_counter() - t0)
        st, _, done = env.step(p)
    return np.array(out[WARMUP:]) * 1e3          # ms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", choices=list(sweeps.SWEEPS))
    ap.add_argument("--runs", default="runs", help="where the sweep's trained runs are")
    ap.add_argument("--layouts", type=int, default=5)
    ap.add_argument("--slots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=sweeps.TEST_SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from poweralloc.lsac.rebuild import agent_from_run
    from poweralloc.sca import SCAAllocator

    result = {"sweep": a.sweep, "cpu": cpu_model(), "host": platform.node(),
              "threads": 1, "warmup_discarded": WARMUP,
              "layouts": a.layouts, "slots": a.slots,
              "versions": {"torch": torch.__version__, "numpy": np.__version__,
                           "python": platform.python_version()},
              "points": []}
    print("CPU:", result["cpu"], flush=True)
    for i, p in enumerate(sweeps.SWEEPS[a.sweep]["points"]):
        args = sweeps.point_args(a.sweep, i)
        methods = {"equal power": lambda c: EqualPowerAllocator(c),
                   "SCA": lambda c: SCAAllocator(c)}
        for m in sweeps.METHODS:
            d = os.path.join(a.runs, sweeps.tag(a.sweep, p["x"], sweeps.SEEDS[0], m))
            if not os.path.exists(os.path.join(d, "result.json")):
                continue
            saved = json.load(open(os.path.join(d, "result.json")))["args"]
            agent = agent_from_run(saved, os.path.join(d, "best.pt"))
            methods[m] = lambda c, agent=agent: agent.allocator()
        samples = {m: [] for m in methods}
        for L in range(a.layouts):
            cfg = sweeps.system_cfg(args, num_slots=a.slots, seed=a.seed + L)
            for m, make in methods.items():           # interleaved per layout
                samples[m].append(times_for(make, cfg))
        row = {"x": p["x"], "args": args}
        for m, v in samples.items():
            v = np.concatenate(v)
            row[m] = {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
                      "p5_ms": float(np.percentile(v, 5)), "p95_ms": float(np.percentile(v, 95)),
                      "max_ms": float(v.max()), "n": int(v.size)}
        result["points"].append(row)
        print(f"x={p['x']}: " + "  ".join(f"{m} {row[m]['median_ms']:.4f} ms (p95 {row[m]['p95_ms']:.4f})"
                                         for m in methods), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
