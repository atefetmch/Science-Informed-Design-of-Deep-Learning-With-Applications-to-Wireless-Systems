"""System and agent configuration from a training run's recorded arguments."""

from __future__ import annotations

from ..config import SystemConfig
from .config import LSACConfig


def system_config(args: dict, **overrides) -> SystemConfig:
    """num_critical makes users 0..n-1 critical and the rest regular."""
    kw = {"seed": args.get("seed", 0)}
    if args.get("r_crit") is not None:
        kw["default_r_crit"] = args["r_crit"]
    if args.get("r_min") is not None:
        kw["default_r_min"] = args["r_min"]
    if args.get("num_users") is not None:
        kw["num_users"] = int(args["num_users"])
    if args.get("num_critical") is not None:
        kw["critical_users"] = tuple(range(int(args["num_critical"])))
    if args.get("p_max_dbm") is not None:
        kw["p_max_dbm"] = float(args["p_max_dbm"])
    if args.get("num_antennas") is not None:
        kw["num_antennas"] = int(args["num_antennas"])
    kw.update(overrides)
    return SystemConfig(**kw)


def lsac_config(args: dict) -> LSACConfig:
    lcfg = LSACConfig(lyapunov=args.get("method", "lsac") != "sac", seed=args.get("seed", 0))
    for key in ("num_episodes", "episode_len", "eval_every", "eval_layouts", "eval_slots",
                "crit_margin", "min_critical", "max_critical"):
        if args.get(key) is not None:
            setattr(lcfg, key, args[key])
    # command-line name -> LSACConfig field
    for key, field in (("mu_update", "mu_update"), ("mu_lr", "lr_mu_sgd"),
                       ("mu_cap", "max_lyap_multiplier"), ("alpha3", "alpha3"),
                       ("crit_weight", "crit_weight")):
        if args.get(key) is not None:
            setattr(lcfg, field, args[key])
    for key in ("critical_flag", "random_critical"):
        if args.get(key) is not None:
            setattr(lcfg, key, bool(int(args[key])))
    if lcfg.random_critical:
        lcfg.critical_flag = True
    return lcfg


def agent_from_run(args: dict, checkpoint: str):
    from .agent import LSACAgent
    agent = LSACAgent(system_config(args), lsac_config(args))
    agent.load(checkpoint)
    return agent
