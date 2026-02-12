from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    nr_of_users: int = 8          # K_model
    active_nr_of_users: int = 4          # K_active (<= nr_of_users)

    nr_of_BS_antennas: int = 8

    channel_noise_std: float = 0.0
    channel_noise_relative: bool = True
    channel_mean_shift: float = 0.0


    total_power: float = 10.0
    noise_power: float = 1

    path_loss_option: bool = False
    path_loss_range: list = field(default_factory=lambda: [-5, 5])

    epsilon: float = 1e-4
    power_tolerance: float = 1e-4
    nr_of_iterations_wmmse: int = 1 #wmmse iterations

    nr_of_iterations_truncated: int = 1

    nr_of_iterations_nn: int = 1
    pgd_steps: int = 4

    nr_of_batches_training: int = 5000
    nr_of_batches_test: int = 1000
    nr_of_samples_per_batch: int = 100

    learning_rate: float = 1e-3
    dnn_hidden: tuple = (512, 1024, 512)

    train_seed: int = 1
    test_seed: int = 10

    use_input_norm: bool = True
    phase_canonicalize: bool = True
    ds_enc_hidden: tuple = (256, 256)
    ds_emb_dim: int = 256
    ds_dec_hidden: tuple = (256, 256)


    train_supervised: bool = False
    train_unsupervised: bool = False
    warmup_steps: int = 500
    rzf_alpha: float | None = None

    @property
    def scheduled_users(self):
        return list(range(self.active_nr_of_users))

    def user_weights_batch(self):
        B = self.nr_of_samples_per_batch
        K = self.nr_of_users
        w = np.ones((B, K, 1), dtype=np.float64)
        if self.active_nr_of_users < self.nr_of_users:
            w[:, self.active_nr_of_users:, 0] = 0.0
        return w

    def user_weights_regular(self):
        w = np.ones(self.nr_of_users, dtype=np.float64)
        if self.active_nr_of_users < self.nr_of_users:
            w[self.active_nr_of_users:] = 0.0
        return w
