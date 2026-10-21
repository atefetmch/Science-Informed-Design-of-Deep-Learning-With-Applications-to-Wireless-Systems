# Fair Power Allocation with Critical QoS Requirements

**Note:** Claude (Anthropic) was used to assist with code development; all code was reviewed, testing, and validated by the authors.


Simulation code for downlink multi-user MISO power allocation with two user groups:

- **critical users**, whose rate must meet $R^{\mathrm{crit}}$ in every slot;
- **regular users**, whose sliding-window average rate must meet $R^{\min}$.

The objective combines proportional fairness among the regular users with the sum throughput. The code compares four methods:

- a Lyapunov-based safe actor–critic (**LSAC**) with a uniform-ultimate-boundedness (UUB) condition on the critical-user constraint, after Han et al., *Automatica* 129 (2021) 109689;
- a soft actor–critic (**SAC**) baseline with a fixed critical-user penalty;
- uniform power allocation (**Uniform PA**);
- a per-slot successive convex approximation (**SCA**) benchmark.

## Repository layout

```
poweralloc/            system model and solvers
  config.py            system parameters (SystemConfig)
  channel.py           Gauss-Markov mobility, path loss, Gauss-Markov fading
  phy.py               MRT beamforming, SINR, rates
  problem.py           per-slot objective and constraints (c1)-(c4)
  env.py               slot-by-slot simulator
  baselines.py         solver interface, Uniform PA
  sca.py               SCA benchmark
  lsac/                LSAC and the SAC baseline
scripts/
  train_lsac.py        train one LSAC or SAC policy
  sweeps.py            simulation settings, training plan, evaluation tables
  safety_analysis.py   test rollouts, violation metrics, mobility shifts, tables
  timing.py            decision time per slot
tests/                 unit tests (pytest)
```

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
python -m pytest -q tests        # optional check, about 1 minute
```

## Reproducing the results

The paper reports two settings, both with $R^{\mathrm{crit}}=12$ Mbps, $R^{\min}=5$ Mbps and $P_{\max}=40$ dBm:

| Setting | Users | Antennas | Sweep and point | Run-folder tag |
|---|---|---|---|---|
| Nominal load | $K=6$ (2 critical) | $M=8$ | `rcrit --point 2` | `sw_rcrit_12_seed<s>` |
| Heavy load | $K=16$ (3 critical) | $M=18$ | `heavy --point 1` | `sw_heavy_16_seed<s>` |

SAC runs use the same tag with the prefix `sac_`. Five seeds are trained per method (`<s>` = 0–4).

### 1. Training

These are the LSAC and SAC commands for one seed. Repeat them for seeds 0–4 (500 episodes of 200 slots each; the best checkpoint on 10 validation layouts is kept):

```bash
# K = 6
python scripts/train_lsac.py --tag sw_rcrit_12_seed0 --seed 0 --num-users 6 --num-critical 2 \
    --r-crit 12 --p-max-dbm 40 --method lsac --mu-update sgd --mu-cap 1000 --out-dir runs
python scripts/train_lsac.py --tag sac_sw_rcrit_12_seed0 --seed 0 --num-users 6 --num-critical 2 \
    --r-crit 12 --p-max-dbm 40 --method sac --out-dir runs

# K = 16
python scripts/train_lsac.py --tag sw_heavy_16_seed0 --seed 0 --num-users 16 --num-critical 3 \
    --r-crit 12 --p-max-dbm 40 --num-antennas 18 --method lsac --mu-update sgd --mu-cap 1000 --out-dir runs
python scripts/train_lsac.py --tag sac_sw_heavy_16_seed0 --seed 0 --num-users 16 --num-critical 3 \
    --r-crit 12 --p-max-dbm 40 --num-antennas 18 --method sac --out-dir runs
```

The paper uses `--mu-update sgd --mu-cap 1000`: projected dual ascent on $\mu$ with step 1 and $\mu \in [10^{-6}, 10^{3}]$. Each run writes `best.pt`, `last.pt` and `result.json` to `runs/<tag>/`.

### 2. Evaluation (50 test layouts × 1000 slots)

Nominal mobility, all four methods:

```bash
python scripts/safety_analysis.py rollout rcrit --point 2 --runs runs --out results/nominal
python scripts/safety_analysis.py metrics rcrit --point 2 --out results/nominal
```

Unseen mobility at test time, without retraining. SCA is skipped with `--no-sca`; it re-solves each slot with the true channel.

```bash
python scripts/safety_analysis.py rollout rcrit --point 2 --runs runs --out results/erratic --shift erratic --no-sca
python scripts/safety_analysis.py metrics rcrit --point 2 --out results/erratic
```

| `--shift` | Test mobility (paper name) |
|---|---|
| `erratic` | stop-and-go: mobility memory 0.5, speed std 8 m/s, heading std 1 rad |
| `fast` | mean speeds 15–30 m/s |
| `highway` | mean speeds 30–60 m/s |

For $K=16$, replace `rcrit --point 2` with `heavy --point 1`.

### 3. Decision time per slot

```bash
python scripts/timing.py heavy --runs runs --layouts 2 --out results/timing/timing_heavy.json
python scripts/timing.py rcrit --runs runs --layouts 2 --out results/timing/timing_rcrit.json
```

Each call times every point of the sweep, on a single CPU thread, after discarding the first 50 decisions of each layout. Learned methods are timed only where a trained run exists.

## Notes

- **Optional features not used in the paper.** The code includes a training-only safety margin (`--crit-margin`) and a variable critical set (`--random-critical`). It also defines further sweeps (`rmin`, `ncrit`, `users`, `anycrit`).
- **Defaults.** `LSACConfig` defaults to the Adam update of $\mu$ with cap 100. Pass the flags above to reproduce the paper's LSAC.
