import numpy as np
from utils import run_WMMSE, compute_channel

def evaluate(cfg):
    np.random.seed(cfg.test_seed)
    wsr_list = []
    for _ in range(cfg.nr_of_batches_test):
        wsr_batch = 0.0
        for _ in range(cfg.nr_of_samples_per_batch):
            _, _, Hc, _ = compute_channel(cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
                                          cfg.path_loss_option, cfg.path_loss_range)

            _, _, _, wsr_one = run_WMMSE(
                cfg.epsilon, Hc, list(cfg.scheduled_users), cfg.total_power,
                cfg.noise_power, cfg.user_weights_regular(),
                cfg.nr_of_iterations_truncated,
                cfg.power_tolerance,
                log=False
            )
            wsr_batch += wsr_one

        wsr_list.append(wsr_batch / cfg.nr_of_samples_per_batch)

    return float(np.mean(wsr_list))
