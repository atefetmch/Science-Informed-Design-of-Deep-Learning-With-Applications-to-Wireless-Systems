"""
Downlink multi-user MISO power allocation with critical (per-slot QoS) and
regular (sliding-window average QoS) users.

    config.py     SystemConfig (all system parameters)
    channel.py    user mobility + path loss + Gauss-Markov fading
    phy.py        MRT beamforming, SINR, rates
    problem.py    per-slot objective and constraints
    env.py        slot-by-slot simulator
    baselines.py  solver interface and equal-power allocator
    sca.py        SCA per-slot optimiser (classical benchmark)
    lsac/         Lyapunov-based safe actor-critic solver
"""

from .baselines import EqualPowerAllocator, PowerAllocator, run_episode
from .channel import ChannelModel
from .config import SystemConfig
from .env import PowerAllocationEnv
from .phy import effective_gains, mrt_beamformers, rates, sinr
from .problem import (OUTAGE_TOL, PowerAllocationProblem, SlotState,
                      in_outage)

__all__ = [
    "OUTAGE_TOL", "in_outage",
    "ChannelModel", "EqualPowerAllocator", "PowerAllocationEnv", "PowerAllocationProblem",
    "PowerAllocator", "SlotState", "SystemConfig", "effective_gains", "mrt_beamformers",
    "rates", "run_episode", "sinr",
]
