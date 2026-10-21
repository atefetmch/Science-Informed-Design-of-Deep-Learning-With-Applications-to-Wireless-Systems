"""
Train LSAC (or the SAC baseline) for one system setting and write result.json,
best.pt (checkpoint with the best validation score) and last.pt.

    python scripts/train_lsac.py --r-crit 12 --seed 0 --mu-update sgd --mu-cap 1000 --tag lsac_r12_s0
    python scripts/train_lsac.py --r-crit 12 --seed 0 --method sac --tag sac_r12_s0
"""

import argparse
import json
import os
import platform
import sys
import time

# The BLAS thread count changes the order of floating-point sums, so it is fixed
# before numpy and torch load; runs are then reproducible bit for bit.
BLAS_THREADS = os.environ.get("POWERALLOC_BLAS_THREADS", "1")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = BLAS_THREADS

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from poweralloc.lsac import LSACAgent
from poweralloc.lsac.rebuild import lsac_config, system_config

# Checkpoint score on the validation layouts
CRIT_WEIGHT, REG_WEIGHT = 50.0, 50.0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["lsac", "sac"], default="lsac")
    ap.add_argument("--seed", type=int, default=0)
    # system setting
    ap.add_argument("--r-crit", type=float, default=None, help="critical target in Mbps")
    ap.add_argument("--r-min", type=float, default=None, help="regular users' minimum average rate in Mbps")
    ap.add_argument("--num-users", type=int, default=None)
    ap.add_argument("--num-critical", type=int, default=None)
    ap.add_argument("--p-max-dbm", type=float, default=None)
    ap.add_argument("--num-antennas", type=int, default=None)
    # optional, training only: R_crit + margin in the cost/penalty (not used in the paper)
    ap.add_argument("--crit-margin", type=float, default=None,
                    help="safety margin on R_crit in Mbps used during training (default 0)")
    # optional: per-user critical flag in the state and a random critical set per
    # training episode (not used in the paper)
    ap.add_argument("--random-critical", type=int, choices=[0, 1], default=None)
    # LSAC multiplier and decrease condition (paper: --mu-update sgd --mu-cap 1000)
    ap.add_argument("--mu-update", choices=["adam", "sgd"], default=None,
                    help="adam or sgd: mu <- clip(mu + mu_lr * UUB, mu_min, mu_cap)")
    ap.add_argument("--mu-lr", type=float, default=None, help="step size of the sgd mu update (default 1.0)")
    ap.add_argument("--mu-cap", type=float, default=None, help="upper bound on mu (default 100)")
    ap.add_argument("--alpha3", type=float, default=None,
                    help="required decrease rate in the UUB condition (default 1.0)")
    # SAC baseline: weight of the critical-user penalty in the reward (default 200)
    ap.add_argument("--crit-weight", type=float, default=None)
    ap.add_argument("--min-critical", type=int, default=None, help="default 1")
    ap.add_argument("--max-critical", type=int, default=None, help="default K-1")
    # schedule and validation
    ap.add_argument("--num-episodes", type=int, default=None)
    ap.add_argument("--episode-len", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-layouts", type=int, default=None)
    ap.add_argument("--eval-slots", type=int, default=None)
    # output
    ap.add_argument("--tag", type=str, default="lsac")
    ap.add_argument("--out-dir", type=str, default="runs")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch intra-op threads; fixed because the count changes the numerics")
    return ap


def score(ev: dict) -> float:
    """Objective, heavily penalised by critical outage (c4) and regular violation (c3)."""
    return (ev["avg_objective"] - CRIT_WEIGHT * ev["critical_outage_prob"]
            - REG_WEIGHT * ev["regular_violation_prob"])


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def main():
    args = build_parser().parse_args()
    torch.set_num_threads(max(1, args.threads))

    cfg = system_config(vars(args))
    lcfg = lsac_config(vars(args))
    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    agent = LSACAgent(cfg, lcfg)
    print(f"[{args.tag}] {args.method}, state dim {agent.encoder.dim}, threads {args.threads}, "
          f"host {platform.node()}", flush=True)

    best = {"score": -np.inf, "episode": -1}
    ckpt = os.path.join(out_dir, "best.pt")

    def keep_best(ep, ev):
        s = score(ev)
        if s > best["score"]:
            best.update(score=s, episode=ep)
            agent.save(ckpt)

    history = agent.train(on_eval=keep_best)
    agent.save(os.path.join(out_dir, "last.pt"))

    out = {
        "tag": args.tag,
        "args": vars(args),
        "elapsed_s": time.time() - t0,
        "host": platform.node(),
        "numerics": {"torch_threads": args.threads, "blas_threads": BLAS_THREADS,
                     "torch": torch.__version__, "numpy": np.__version__},
        "best_episode": best["episode"],
        "history": history,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    print(f"[{args.tag}] best episode {best['episode']}, done in {out['elapsed_s']/60:.1f} min "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
