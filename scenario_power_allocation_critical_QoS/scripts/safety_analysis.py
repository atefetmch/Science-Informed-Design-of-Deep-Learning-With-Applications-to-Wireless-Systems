"""
Safety analysis of the critical-user constraint (c4).

Four commands, all run on the same sweep points, seeds and test layouts as
scripts/sweeps.py:

    # 1) per-slot rollouts on the test layouts (once per sweep point)
    python scripts/safety_analysis.py rollout  rcrit --point 1 --runs runs --out results/safety
    # 2) severity + duration/recovery metrics, paired tests, LaTeX table, CCDF plots
    python scripts/safety_analysis.py metrics  rcrit --point 1 --out results/safety
    # 3) critical cost during training (from result.json, no rollouts needed)
    python scripts/safety_analysis.py training rcrit --point 1 --runs runs --out results/safety
    # 4) does the Lyapunov critic L(s,a) predict upcoming violations?
    python scripts/safety_analysis.py lyapauc  rcrit --point 1 --out results/safety

Definitions (all on the slots after the W-1 warm-up, as in sweeps.py):
  c_t          = max_{k in K_c} [R_k^crit - R_k(t)]^+   (Mbps), the LSAC cost
  violation    c_t > OUTAGE_TOL  (identical to the "crit %slot" metric)
  severity     mean / p95 / p99 / max of c_t over the violating slots;
               CVaR_0.95 = mean of the worst 5% of all c_t (frequency x severity)
  run          maximal block of consecutive violating slots in one layout;
               its length is the recovery time (slots until c_t returns to 0).
               Runs cut off by the end of a test layout are kept (censored).
  Lyapunov AUC ROC-AUC of L(s_t, a_t) for y_t = 1{violation in t+1..t+n},
               on all slots and on currently safe slots (c_t = 0, onset
               prediction), against two non-learned scores: the current cost
               c_t and the current worst critical rate margin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOL = 1e-9                     # = poweralloc.problem.OUTAGE_TOL
LEARNED = ("SAC", "LSAC")
COLORS = {"Uniform PA": "tab:gray", "SCA": "tab:green", "SAC": "tab:orange", "LSAC": "tab:blue"}


# Metric helpers
def run_lengths(viol: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of True in a 1-D boolean array."""
    v = np.asarray(viol, dtype=int)
    if not v.any():
        return np.zeros(0, dtype=int)
    d = np.diff(np.r_[0, v, 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def cvar(x: np.ndarray, level: float = 0.95) -> float:
    """Mean of the worst (1 - level) fraction of x."""
    x = np.sort(np.ravel(x))
    # round first: (1 - 0.95) * 100 = 5.000000000000004 would otherwise give 6
    n_tail = max(1, int(np.ceil(round((1 - level) * x.size, 9))))
    return float(x[-n_tail:].mean())


def future_label(viol_row: np.ndarray, n: int) -> np.ndarray:
    """y_t = 1{any violation in slots t+1..t+n}, for t = 0..len-n-1."""
    cs = np.r_[0, np.cumsum(np.asarray(viol_row, dtype=int))]
    t = np.arange(len(viol_row) - n)
    return (cs[t + 1 + n] - cs[t + 1]) > 0


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties count 1/2)."""
    from scipy.stats import rankdata
    label = np.asarray(label, dtype=bool)
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def severity(cost: np.ndarray) -> dict:
    """cost: (layouts, slots) array of c_t in Mbps."""
    flat = np.ravel(cost)
    v = flat > TOL
    pos = flat[v]
    out = {"viol_pct": 100.0 * float(v.mean()), "n_viol": int(v.sum()),
           "mean_cost": float(flat.mean()), "cvar95": cvar(flat, 0.95)}
    if pos.size:
        out.update(mean_short=float(pos.mean()), p95_short=float(np.percentile(pos, 95)),
                   p99_short=float(np.percentile(pos, 99)), max_short=float(pos.max()))
    else:   # never violated: statistics conditional on a violation are undefined
        out.update(mean_short=np.nan, p95_short=np.nan, p99_short=np.nan, max_short=np.nan)
    return out


def durations(cost: np.ndarray) -> dict:
    runs = np.concatenate([run_lengths(row > TOL) for row in np.atleast_2d(cost)])
    out = {"n_runs": int(runs.size)}
    if runs.size:
        out.update(mean_run=float(runs.mean()), p95_run=float(np.percentile(runs, 95)),
                   max_run=int(runs.max()), single_slot_pct=100.0 * float(np.mean(runs == 1)))
    else:
        out.update(mean_run=np.nan, p95_run=np.nan, max_run=np.nan, single_slot_pct=np.nan)
    return out


def mean_std(values) -> tuple[float, float, int]:
    """Mean and sample std over seeds, ignoring NaN (seeds with no violation)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


# Test-time mobility shifts (training: mean speeds U[0.5, 15] m/s, mobility memory
# 0.99, speed std 1 m/s, heading std 0.3 rad). "erratic" is stop-and-go in the paper.
SHIFTS = {
    "fast": {"v_min_mps": 15.0, "v_max_mps": 30.0},
    "erratic": {"mobility_memory": 0.5, "speed_std_mps": 8.0, "direction_std_rad": 1.0},
    "highway": {"v_min_mps": 30.0, "v_max_mps": 60.0},
}
FLAG_TO_FIELD = {"v_min": "v_min_mps", "v_max": "v_max_mps", "mobility_memory": "mobility_memory",
                 "speed_std": "speed_std_mps", "direction_std": "direction_std_rad"}


def mobility_overrides(a) -> dict:
    """SystemConfig overrides for the test environment from --shift (one or more
    presets, merged left to right) and explicit flags (applied last)."""
    over = {}
    for name in (getattr(a, "shift", None) or []):
        over.update(SHIFTS[name])
    for flag, field in FLAG_TO_FIELD.items():
        v = getattr(a, flag, None)
        if v is not None:
            over[field] = v
    # test-time critical set: --test-critical N -> users 0..N-1; --test-critical-set ids
    if getattr(a, "test_critical_set", None):
        over["critical_users"] = tuple(sorted(set(a.test_critical_set)))
    elif getattr(a, "test_critical", None) is not None:
        over["critical_users"] = tuple(range(a.test_critical))
    return over


def point_info(sweep: str, i: int):
    import sweeps
    return sweeps.SWEEPS[sweep]["points"][i]["x"], sweeps.point_args(sweep, i)


def npz_path(out: str, sweep: str, x) -> str:
    return os.path.join(out, f"safety_{sweep}_{x}.npz")


def load_agent(runs: str, sweep: str, x, seed: int, method: str, args: dict):
    import sweeps
    from poweralloc.lsac.rebuild import agent_from_run
    d = os.path.join(runs, sweeps.tag(sweep, x, seed, method))
    ckpt = os.path.join(d, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing checkpoint {ckpt}")
    res = os.path.join(d, "result.json")
    if os.path.exists(res):
        saved = json.load(open(res))["args"]
        bad = {k: (saved.get(k), v) for k, v in args.items()
               if saved.get(k) is None or not np.isclose(float(saved[k]), float(v))}
        if bad:
            raise SystemExit(f"{d} was trained with different settings (recorded, wanted): {bad}")
    else:
        # No result.json: rebuild the agent from this sweep point's settings
        saved = {**args, "seed": seed, "method": method.lower()}
        print(f"  note: {d} has no result.json; using this sweep point's settings", flush=True)
    return agent_from_run(saved, ckpt)


def rollout_one(args: dict, layouts: int, slots: int, make_alloc=None, agent=None,
                seed_base: int | None = None, overrides: dict | None = None):
    """
    Per-slot c_t, worst critical rate margin min_{k in K_c}(R_k - R_k^crit), and,
    for LSAC, L(s_t, a_t). Learned policies act exactly as LSACAllocator does.
    Also the per-slot objective and whether any regular user violates (c3).
    `overrides` change the simulated system only (test-condition shifts); a learned
    agent keeps its training configuration, so its state features are normalised
    exactly as in training (e.g. speed / training v_max).
    Returns five (layouts, slots - T + 1) arrays.
    """
    import torch
    import sweeps
    from poweralloc import PowerAllocationEnv
    from poweralloc.lsac import qos_cost

    seed_base = sweeps.TEST_SEED if seed_base is None else seed_base
    C, Mg, L, OB, RG = [], [], [], [], []
    for i in range(layouts):
        cfg = sweeps.system_cfg(args, num_slots=slots, seed=seed_base + i, **(overrides or {}))
        kr = cfg.regular_mask
        kc = cfg.critical_mask
        env = PowerAllocationEnv(cfg)
        alloc = make_alloc(cfg) if agent is None else None
        st, done, t = env.reset(), False, 0
        c_l, m_l, l_l, o_l, r_l = [], [], [], [], []
        while not done:
            lval = np.nan
            if agent is None:
                p = alloc.allocate(st)
            else:
                s_vec = agent.encoder(st)
                u = agent.act(s_vec, deterministic=True)
                p = agent.mapper(u)
                if agent.lcfg.lyapunov:
                    with torch.no_grad():
                        lval = float(agent.lc(
                            torch.as_tensor(s_vec, device=agent.dev).unsqueeze(0),
                            torch.as_tensor(u, device=agent.dev).unsqueeze(0)))
            st, info, done = env.step(p)
            if t >= cfg.window_length - 1:                       # after warm-up
                c_l.append(qos_cost(info, cfg))
                m_l.append(float(np.min(info["rates"][kc] - cfg.r_crit[kc])))
                l_l.append(lval)
                o_l.append(info["objective"])
                r_l.append(bool(np.any(info["regular_violation"][kr] > TOL)) if kr.any() else False)
            t += 1
        C.append(c_l); Mg.append(m_l); L.append(l_l); OB.append(o_l); RG.append(r_l)
    return np.array(C), np.array(Mg), np.array(L), np.array(OB), np.array(RG)


def cmd_rollout(a):
    import torch
    import sweeps
    from poweralloc import EqualPowerAllocator
    torch.set_num_threads(1)

    x, args = point_info(a.sweep, a.point)
    os.makedirs(a.out, exist_ok=True)
    data = {"x": np.array(x), "args": np.array(json.dumps(args)),
            "seed_base": np.array(a.seed_base)}
    over = mobility_overrides(a)
    data["test_shift"] = np.array(json.dumps(over))
    if over:
        print("test-condition shift (simulation only):", over, flush=True)

    base = [("uniform", EqualPowerAllocator)]
    if not a.no_sca:
        from poweralloc.sca import SCAAllocator
        base.append(("sca", SCAAllocator))
    for key, cls in base:
        c, m, _, ob, rg = rollout_one(args, a.layouts, a.slots,
                                      make_alloc=lambda cfg, cls=cls: cls(cfg), seed_base=a.seed_base,
                                      overrides=over)
        data[f"{key}_cost"], data[f"{key}_margin"] = c, m
        data[f"{key}_obj"], data[f"{key}_reg"] = ob, rg
        print(f"{key:8s} crit %slot {100 * np.mean(c > TOL):6.2f}", flush=True)

    for m in LEARNED:
        for s in a.seeds:
            agent = load_agent(a.runs, a.sweep, x, s, m, args)
            c, mg, lv, ob, rg = rollout_one(args, a.layouts, a.slots, agent=agent,
                                            seed_base=a.seed_base, overrides=over)
            k = f"{m}_s{s}"
            data[f"{k}_cost"], data[f"{k}_margin"] = c, mg
            data[f"{k}_obj"], data[f"{k}_reg"] = ob, rg
            if m == "LSAC":
                data[f"{k}_lyap"] = lv
            print(f"{m:4s} seed {s} crit %slot {100 * np.mean(c > TOL):6.2f}", flush=True)

    np.savez_compressed(npz_path(a.out, a.sweep, x), **data)
    print("wrote", npz_path(a.out, a.sweep, x))


def _methods(d) -> dict:
    """{display name: list of (layouts, slots) cost arrays, one per seed}."""
    import sweeps
    out = {"Uniform PA": [d["uniform_cost"]]}
    if "sca_cost" in d:
        out["SCA"] = [d["sca_cost"]]
    for m in LEARNED:
        out[m] = [d[f"{m}_s{s}_cost"] for s in sweeps.SEEDS if f"{m}_s{s}_cost" in d]
    return out


def _keys(d) -> dict:
    """npz key prefixes per method, in the same order as _methods()."""
    import sweeps
    out = {"Uniform PA": ["uniform"]}
    if "sca_cost" in d:
        out["SCA"] = ["sca"]
    for m in LEARNED:
        out[m] = [f"{m}_s{s}" for s in sweeps.SEEDS if f"{m}_s{s}_cost" in d]
    return out


METRIC_COLS = [  # key, header, format
    ("viol_pct", "Viol. (\\%)", "{:.2f}"),
    ("mean_short", "Mean shortfall", "{:.2f}"),
    ("p99_short", "P99 shortfall", "{:.2f}"),
    ("max_short", "Max shortfall", "{:.2f}"),
    ("cvar95", "CVaR$_{0.95}$", "{:.3f}"),
    ("mean_run", "Mean run", "{:.2f}"),
    ("max_run", "Max run", "{:.1f}"),
]


def cmd_metrics(a):
    from scipy.stats import wilcoxon
    x, _ = point_info(a.sweep, a.point)
    d = np.load(npz_path(a.out, a.sweep, x))
    methods = _methods(d)

    keys = _keys(d)
    per_seed = {}
    for m, arrs in methods.items():
        per_seed[m] = []
        for c, k in zip(arrs, keys[m]):
            rec = {**severity(c), **durations(c)}
            if f"{k}_obj" in d:
                rec["objective"] = float(d[f"{k}_obj"].mean())
                rec["reg_viol_pct"] = 100.0 * float(d[f"{k}_reg"].mean())
            per_seed[m].append(rec)
    summary = {}
    for m, recs in per_seed.items():
        summary[m] = {k: mean_std([r[k] for r in recs]) for k in recs[0]}

    # Paired comparison per test layout (seed-averaged), LSAC vs SAC
    paired = {}
    for name, fn in (("viol_pct", lambda c: 100 * np.mean(c > TOL, axis=1)),
                     ("mean_cost", lambda c: c.mean(axis=1))):
        lsac = np.mean([fn(c) for c in methods["LSAC"]], axis=0)
        sac = np.mean([fn(c) for c in methods["SAC"]], axis=0)
        try:
            p = float(wilcoxon(lsac, sac).pvalue)
        except ValueError:          # all paired differences are zero
            p = float("nan")
        paired[name] = {"lsac_mean": float(lsac.mean()), "sac_mean": float(sac.mean()),
                        "lsac_better_layouts": int(np.sum(lsac < sac)),
                        "sac_better_layouts": int(np.sum(sac < lsac)),
                        "wilcoxon_p": p}

    # Text table
    hdr = f"{'method':11s}" + "".join(f"{k:>18s}" for k, _, _ in METRIC_COLS) + "  seeds w/ viol."
    lines = [f"{a.sweep} x={x}", hdr]
    for m, s in summary.items():
        row = f"{m:11s}"
        for k, _, _ in METRIC_COLS:
            mu, sd, n = s[k]
            row += f"{mu:11.3f}+/-{sd:5.3f}" if len(per_seed[m]) > 1 else f"{mu:18.3f}"
        nv = sum(r["n_viol"] > 0 for r in per_seed[m])
        lines.append(row + f"  {nv}/{len(per_seed[m])}")
    if "objective" in per_seed["LSAC"][0]:
        lines.append("Trade-off (same slots):   avg objective        regular viol. %slot")
        for m, s in summary.items():
            (o, osd, _), (g, gsd, _) = s["objective"], s["reg_viol_pct"]
            lines.append(f"  {m:11s}  {o:9.4f} +/- {osd:6.4f}   {g:7.3f} +/- {gsd:6.3f}")
    lines.append("LSAC vs SAC, paired over the evaluated layouts:")
    for k, v in paired.items():
        lines.append(f"  {k}: LSAC {v['lsac_mean']:.4f}  SAC {v['sac_mean']:.4f}  "
                     f"LSAC better on {v['lsac_better_layouts']} layouts, SAC on "
                     f"{v['sac_better_layouts']}, Wilcoxon p = {v['wilcoxon_p']:.3g}")
    print("\n".join(lines))
    with open(os.path.join(a.out, f"metrics_{a.sweep}_{x}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"summary": {m: {k: list(v) for k, v in s.items()} for m, s in summary.items()},
               "per_seed": per_seed, "paired": paired},
              open(os.path.join(a.out, f"metrics_{a.sweep}_{x}.json"), "w"), indent=1, default=float)

    # LaTeX table (IEEE style, booktabs)
    def cell(m, k, fmt):
        mu, sd, n = summary[m][k]
        if np.isnan(mu):
            return "--"
        return fmt.format(mu) + (f" $\\pm$ {fmt.format(sd)}" if len(per_seed[m]) > 1 else "")
    tex = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{Severity and duration of critical-QoS violations ({a.sweep} $= {x}$). "
           "Shortfalls in Mbps, run lengths in slots ($T_s=1$~ms); learned methods: "
           "mean $\\pm$ std over the seeds that violated at least once; "
           "--: no violation.}",
           f"\\label{{tab:safety_{a.sweep}_{x}}}", "\\resizebox{\\columnwidth}{!}{%",
           "\\begin{tabular}{l" + "c" * len(METRIC_COLS) + "}", "\\toprule",
           "Method & " + " & ".join(h for _, h, _ in METRIC_COLS) + " \\\\", "\\midrule"]
    for m in summary:
        tex.append(m + " & " + " & ".join(cell(m, k, f) for k, _, f in METRIC_COLS) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(os.path.join(a.out, f"table_safety_{a.sweep}_{x}.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # CCDF plots: shortfall given violation, and run length (pooled over seeds)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for m, arrs in methods.items():
        if m == "Uniform PA" and not a.with_uniform:
            continue
        pooled = np.concatenate([c.ravel() for c in arrs])
        short = np.sort(pooled[pooled > TOL])
        runs = np.sort(np.concatenate([run_lengths(row > TOL) for c in arrs for row in c]))
        for axis, v in ((ax[0], short), (ax[1], runs)):
            if v.size:
                axis.step(v, 1.0 - np.arange(v.size) / v.size, where="post",
                          color=COLORS[m], label=m)
    ax[0].set_xlabel("Critical shortfall $c_t$ given violation (Mbps)")
    ax[1].set_xlabel("Violation run length (slots)")
    for axis in ax:
        axis.set_yscale("log"); axis.set_ylabel("CCDF"); axis.grid(alpha=0.3)
    ax[1].set_xscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"ccdf_{a.sweep}_{x}.pdf"))
    print("wrote metrics, table and CCDF plot to", a.out)


def cmd_training(a):
    import sweeps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from poweralloc.lsac import LSACConfig

    x, _ = point_info(a.sweep, a.point)
    rand_eps = LSACConfig().random_episodes
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.4))
    finals = {}
    for m in LEARNED:
        per_slot, cum, edge = [], [], []
        for s in sweeps.SEEDS:
            path = os.path.join(a.runs, sweeps.tag(a.sweep, x, s, m), "result.json")
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found: the training-cost figure is built from the "
                                 "training history in result.json, which train_lsac.py writes "
                                 "at the end of training. Ask for these files to make this figure.")
            h = json.load(open(path))["history"]
            c = np.array([r["ep_cost"] for r in h])
            n = np.array([r["steps"] + 1 for r in h])          # ep_cost sums all slots
            per_slot.append(c / n)
            cum.append(np.cumsum(c) / 1e3)
            edge.append(100 * np.array([r["edge_frac"] for r in h]))
        finals[m] = [cu[-1] for cu in cum]
        w = a.smooth
        for axis, arr in ((ax[0], per_slot), (ax[1], cum), (ax[2], edge)):
            arr = np.array(arr)
            if axis is not ax[1] and w > 1:
                k = np.ones(w) / w
                arr = np.array([np.convolve(r, k, mode="valid") for r in arr])
            ep = np.arange(arr.shape[1]) + (w - 1 if axis is not ax[1] and w > 1 else 0)
            mu, sd = arr.mean(0), arr.std(0, ddof=1)
            axis.plot(ep + 1, mu, color=COLORS[m], label=m)
            axis.fill_between(ep + 1, mu - sd, mu + sd, color=COLORS[m], alpha=0.25, lw=0)
    ax[0].set_ylabel("Shortfall per slot (Mbps)")
    ax[1].set_ylabel("Cum. undelivered data (Mb)")
    ax[2].set_ylabel("Slots with $c_t\\geq\\epsilon$ (%)")
    for axis in ax:
        axis.set_xlabel("Training episode")
        axis.axvspan(0, rand_eps, color="k", alpha=0.08, lw=0)   # random-action episodes
        axis.grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"training_cost_{a.sweep}_{x}.pdf"))
    for m, v in finals.items():
        print(f"{m}: cumulative training shortfall {np.mean(v):.2f} +/- "
              f"{np.std(v, ddof=1):.2f} (Mbit, with T_s = 1 ms) over seeds")
    print("wrote", os.path.join(a.out, f"training_cost_{a.sweep}_{x}.pdf"))


def cmd_lyapauc(a):
    import sweeps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, _ = point_info(a.sweep, a.point)
    d = np.load(npz_path(a.out, a.sweep, x))
    seeds = [s for s in sweeps.SEEDS if f"LSAC_s{s}_lyap" in d]
    res = {n: {k: [] for k in ("L_all", "cost_all", "margin_all", "L_safe", "margin_safe",
                               "pos_rate_safe")} for n in a.horizons}
    for s in seeds:
        C, Mg, Lv = d[f"LSAC_s{s}_cost"], d[f"LSAC_s{s}_margin"], d[f"LSAC_s{s}_lyap"]
        for n in a.horizons:
            y = np.concatenate([future_label(row > TOL, n) for row in C])
            cut = C.shape[1] - n
            l, c, m = Lv[:, :cut].ravel(), C[:, :cut].ravel(), Mg[:, :cut].ravel()
            safe = c <= TOL
            r = res[n]
            r["L_all"].append(roc_auc(l, y))
            r["cost_all"].append(roc_auc(c, y))
            r["margin_all"].append(roc_auc(-m, y))
            r["L_safe"].append(roc_auc(l[safe], y[safe]))
            r["margin_safe"].append(roc_auc(-m[safe], y[safe]))
            r["pos_rate_safe"].append(100 * float(y[safe].mean()))

    lines = [f"Lyapunov-critic ROC-AUC, {a.sweep} x={x}, LSAC, mean +/- std over {len(seeds)} seeds",
             "  all slots: L(s,a) | current cost c_t | current rate margin;  "
             "safe slots (c_t=0, onset): L(s,a) | rate margin | % positives"]
    for n in a.horizons:
        f = {k: mean_std(v) for k, v in res[n].items()}
        lines.append(
            f"n={n:3d}  all: {f['L_all'][0]:.3f}+/-{f['L_all'][1]:.3f} | "
            f"{f['cost_all'][0]:.3f} | {f['margin_all'][0]:.3f}   "
            f"safe: {f['L_safe'][0]:.3f}+/-{f['L_safe'][1]:.3f} | "
            f"{f['margin_safe'][0]:.3f} | {f['pos_rate_safe'][0]:.2f}%")
    print("\n".join(lines))
    with open(os.path.join(a.out, f"lyapauc_{a.sweep}_{x}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # Risk curve: P(violation within n slots | L-decile), safe slots, seeds pooled
    n = a.horizons[len(a.horizons) // 2]
    Ls, Ys = [], []
    for s in seeds:
        C, Lv = d[f"LSAC_s{s}_cost"], d[f"LSAC_s{s}_lyap"]
        y = np.concatenate([future_label(row > TOL, n) for row in C])
        cut = C.shape[1] - n
        safe = C[:, :cut].ravel() <= TOL
        Ls.append(Lv[:, :cut].ravel()[safe]); Ys.append(y[safe])
    Ls, Ys = np.concatenate(Ls), np.concatenate(Ys)
    edges = np.unique(np.quantile(Ls, np.linspace(0, 1, a.bins + 1)))
    idx = np.clip(np.searchsorted(edges, Ls, side="right") - 1, 0, len(edges) - 2)
    centers = [np.median(Ls[idx == b]) for b in range(len(edges) - 1)]
    risk = [100 * Ys[idx == b].mean() for b in range(len(edges) - 1)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(centers, risk, "o-", color=COLORS["LSAC"])
    ax.set_xscale("log")
    ax.set_xlabel("Lyapunov critic $L(\\mathbf{s}_t,\\mathbf{a}_t)$ (decile median)")
    ax.set_ylabel(f"P(violation in next {n} slots) (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"lyap_risk_{a.sweep}_{x}.pdf"))
    print("wrote", os.path.join(a.out, f"lyap_risk_{a.sweep}_{x}.pdf"))


def main():
    import sweeps
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rollout", "metrics", "training", "lyapauc"):
        p = sub.add_parser(name)
        p.add_argument("sweep", choices=list(sweeps.SWEEPS))
        p.add_argument("--point", type=int, required=True)
        p.add_argument("--out", default="results/safety")
        if name in ("rollout", "training"):
            p.add_argument("--runs", default="runs")
        if name == "rollout":
            p.add_argument("--layouts", type=int, default=sweeps.TEST_LAYOUTS)
            p.add_argument("--slots", type=int, default=sweeps.TEST_SLOTS)
            p.add_argument("--no-sca", action="store_true", help="skip SCA (the slow baseline)")
            p.add_argument("--shift", choices=sorted(SHIFTS), default=None, nargs="+",
                           help="test-condition shift: " + "; ".join(
                               f"{k}: {v}" for k, v in sorted(SHIFTS.items())))
            for flag, typ in (("--v-min", float), ("--v-max", float), ("--mobility-memory", float),
                              ("--speed-std", float), ("--direction-std", float)):
                p.add_argument(flag, type=typ, default=None,
                               help="test-only system parameter (overrides --shift)")
            p.add_argument("--test-critical", type=int, default=None,
                           help="test with users 0..N-1 critical (flag-based models adapt; "
                                "index-based models still protect their training set)")
            p.add_argument("--test-critical-set", type=int, nargs="+", default=None,
                           help="test with exactly these users critical")
            p.add_argument("--seeds", type=int, nargs="+", default=list(sweeps.SEEDS),
                           help="training seeds to load (default: all five)")
            p.add_argument("--seed-base", type=int, default=sweeps.TEST_SEED,
                           help="first layout seed: 70000 = test layouts (default), "
                                "12345 = validation layouts, for choosing settings")
        if name == "metrics":
            p.add_argument("--with-uniform", action="store_true", help="include Uniform PA in CCDFs")
        if name == "training":
            p.add_argument("--smooth", type=int, default=10, help="moving-average window (episodes)")
        if name == "lyapauc":
            p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
            p.add_argument("--bins", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"rollout": cmd_rollout, "metrics": cmd_metrics, "training": cmd_training,
     "lyapauc": cmd_lyapauc}[a.cmd](a)


if __name__ == "__main__":
    main()
