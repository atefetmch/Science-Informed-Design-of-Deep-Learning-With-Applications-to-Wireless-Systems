"""
CSI Data Augmentation Functions

Implements multiple augmentation techniques to make the dataset more challenging:
- SNR noise
- Phase noise
- Amplitude scaling
- Time warping
- Subcarrier dropout
- Frequency-selective fading
- Burst errors
- DC offset
- Frequency offset
- Temporal dropout
- Impulse noise
"""

import numpy as np
from typing import Dict, List
from scipy import interpolate
import random


class CSIAugmentor:
    """Augment CSI data to increase difficulty."""

    def __init__(self, aug_params: Dict = None):
        """
        Initialize augmentor with parameters.

        Args:
            aug_params: Dictionary with augmentation parameters.
                        If None, uses 'medium' difficulty defaults.
        """
        if aug_params is None:
            aug_params = get_augmentation_params('medium')

        self.params = aug_params

    def augment_dataset(self, csi_data: List[np.ndarray],
                        verbose: bool = True) -> List[np.ndarray]:
        """
        Augment entire dataset.

        Args:
            csi_data: List of CSI matrices (each: 150x104 complex)
            verbose: Print progress

        Returns:
            augmented_data: List of augmented CSI matrices
        """
        augmented_data = []

        if verbose:
            print(f"Augmenting {len(csi_data)} samples...")

        for i, sample in enumerate(csi_data):
            aug_sample = self.augment_sample(sample)
            augmented_data.append(aug_sample)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(csi_data)} samples...")

        if verbose:
            print(f"  Augmentation complete!")

        return augmented_data

    def augment_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single CSI sample.

        Args:
            sample: CSI matrix (time x features), expects complex values

        Returns:
            augmented: Augmented CSI matrix
        """
        sample = sample.copy()

        if not np.iscomplexobj(sample):
            sample = sample.astype(complex)

        sample = self._add_gaussian_noise(sample)
        sample = self._add_phase_noise(sample)
        sample = self._amplitude_scaling(sample)
        sample = self._time_warping(sample)
        sample = self._subcarrier_dropout(sample)
        sample = self._frequency_selective_fading(sample)

        # Severe augmentations
        sample = self._burst_errors(sample)
        sample = self._dc_offset(sample)
        sample = self._frequency_offset(sample)
        sample = self._temporal_dropout(sample)
        sample = self._impulse_noise(sample)

        return sample

    def _add_gaussian_noise(self, sample: np.ndarray) -> np.ndarray:
        """Add SNR-based Gaussian noise."""
        if np.random.rand() < self.params['apply_prob']:
            signal_power = np.mean(np.abs(sample) ** 2)
            snr_linear = 10 ** (self.params['snr_db'] / 10)
            noise_power = signal_power / snr_linear

            noise_real = np.random.randn(*sample.shape) * np.sqrt(noise_power / 2)
            noise_imag = np.random.randn(*sample.shape) * np.sqrt(noise_power / 2)
            noise = noise_real + 1j * noise_imag

            sample = sample + noise

        return sample

    def _add_phase_noise(self, sample: np.ndarray) -> np.ndarray:
        """Add phase noise."""
        if np.random.rand() < self.params['apply_prob']:
            phase_noise = self.params['phase_noise_std'] * np.random.randn(*sample.shape)
            magnitude = np.abs(sample)
            phase = np.angle(sample) + phase_noise
            sample = magnitude * np.exp(1j * phase)

        return sample

    def _amplitude_scaling(self, sample: np.ndarray) -> np.ndarray:
        """Random amplitude scaling."""
        if np.random.rand() < self.params['apply_prob']:
            scale_min, scale_max = self.params['amp_scale_range']
            scale = scale_min + (scale_max - scale_min) * np.random.rand()
            sample = sample * scale

        return sample

    def _time_warping(self, sample: np.ndarray) -> np.ndarray:
        """Apply time warping."""
        if np.random.rand() < self.params['apply_prob']:
            sample = self._warp_time(sample, self.params['time_warp_factor'])

        return sample

    def _warp_time(self, signal: np.ndarray, strength: float) -> np.ndarray:
        """
        Time warping implementation.

        Args:
            signal: (n_time, n_sub) or (n_sub, n_time)
            strength: Warping strength
        """
        # Detect if (time, features) or (features, time)
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
            transposed = True
        else:
            transposed = False

        n_sub, n_time = signal.shape
        warp_points = 5

        ctrl_warp = 1 + strength * (2 * np.random.rand(warp_points) - 1)

        x_ctrl = np.linspace(0, n_time - 1, warp_points)
        x_new = np.arange(n_time)
        warp_curve = interpolate.interp1d(x_ctrl, ctrl_warp, kind='cubic')(x_new)

        new_indices = np.cumsum(warp_curve)
        new_indices = ((new_indices - new_indices.min())
                       / (new_indices.max() - new_indices.min()) * (n_time - 1))

        warped = np.zeros_like(signal)
        for i in range(n_sub):
            real_interp = interpolate.interp1d(
                x_new, signal[i, :].real, kind='linear', fill_value='extrapolate')
            imag_interp = interpolate.interp1d(
                x_new, signal[i, :].imag, kind='linear', fill_value='extrapolate')
            warped[i, :] = real_interp(new_indices) + 1j * imag_interp(new_indices)

        if transposed:
            warped = warped.T

        return warped

    def _subcarrier_dropout(self, sample: np.ndarray) -> np.ndarray:
        """Random subcarrier dropout."""
        if np.random.rand() < self.params['apply_prob']:
            if sample.shape[0] > sample.shape[1]:
                n_sub = sample.shape[1]
                axis = 1
            else:
                n_sub = sample.shape[0]
                axis = 0

            n_drop = int(n_sub * self.params['subcarrier_dropout'])
            if n_drop > 0:
                drop_idx = np.random.choice(n_sub, n_drop, replace=False)
                if axis == 0:
                    sample[drop_idx, :] = 0
                else:
                    sample[:, drop_idx] = 0

        return sample

    def _frequency_selective_fading(self, sample: np.ndarray) -> np.ndarray:
        """Frequency-selective fading."""
        if np.random.rand() < self.params['apply_prob']:
            if sample.shape[0] > sample.shape[1]:
                n_sub = sample.shape[1]
                fading = 0.8 + 0.4 * np.random.rand(1, n_sub)
            else:
                n_sub = sample.shape[0]
                fading = 0.8 + 0.4 * np.random.rand(n_sub, 1)

            sample = sample * fading

        return sample

    def _burst_errors(self, sample: np.ndarray) -> np.ndarray:
        """Burst errors - consecutive subcarrier failures."""
        if ('burst_error_prob' in self.params
                and np.random.rand() < self.params['burst_error_prob']):

            burst_min, burst_max = self.params['burst_length_range']
            burst_length = np.random.randint(burst_min, burst_max + 1)

            if sample.shape[0] > sample.shape[1]:
                n_sub = sample.shape[1]
                burst_start = np.random.randint(0, n_sub - burst_length)
                sample[:, burst_start:burst_start + burst_length] = 0
            else:
                n_sub = sample.shape[0]
                burst_start = np.random.randint(0, n_sub - burst_length)
                sample[burst_start:burst_start + burst_length, :] = 0

        return sample

    def _dc_offset(self, sample: np.ndarray) -> np.ndarray:
        """DC offset - hardware calibration error."""
        if ('dc_offset' in self.params
                and np.random.rand() < self.params['apply_prob']):
            dc = self.params['dc_offset'] * (2 * np.random.rand() - 1)
            sample = sample + dc

        return sample

    def _frequency_offset(self, sample: np.ndarray) -> np.ndarray:
        """Frequency offset - carrier frequency error."""
        if ('frequency_offset_hz' in self.params
                and np.random.rand() < self.params['apply_prob']):

            if sample.shape[0] > sample.shape[1]:
                n_time = sample.shape[0]
                axis = 0
            else:
                n_time = sample.shape[1]
                axis = 1

            f_offset = self.params['frequency_offset_hz'] * (2 * np.random.rand() - 1)

            t = np.arange(n_time)
            phase_rotation = np.exp(1j * 2 * np.pi * f_offset * t / 1000)

            if axis == 0:
                sample = sample * phase_rotation[:, np.newaxis]
            else:
                sample = sample * phase_rotation[np.newaxis, :]

        return sample

    def _temporal_dropout(self, sample: np.ndarray) -> np.ndarray:
        """Temporal dropout - entire time segments missing."""
        if ('temporal_dropout' in self.params
                and np.random.rand() < self.params['temporal_dropout']):

            if sample.shape[0] > sample.shape[1]:
                n_time = sample.shape[0]
                dropout_length = np.random.randint(5, 16)
                dropout_start = np.random.randint(0, n_time - dropout_length)
                sample[dropout_start:dropout_start + dropout_length, :] = 0
            else:
                n_time = sample.shape[1]
                dropout_length = np.random.randint(5, 16)
                dropout_start = np.random.randint(0, n_time - dropout_length)
                sample[:, dropout_start:dropout_start + dropout_length] = 0

        return sample

    def _impulse_noise(self, sample: np.ndarray) -> np.ndarray:
        """Impulse noise - sudden spikes."""
        if ('impulse_noise_prob' in self.params
                and np.random.rand() < self.params['impulse_noise_prob']):

            n_impulses = np.random.randint(1, 6)
            impulse_amp = 5 * np.mean(np.abs(sample))

            for _ in range(n_impulses):
                i = np.random.randint(0, sample.shape[0])
                j = np.random.randint(0, sample.shape[1])
                sample[i, j] = impulse_amp * np.exp(1j * 2 * np.pi * np.random.rand())

        return sample


def get_augmentation_params(difficulty='medium'):
    """
    Get predefined augmentation parameters.

    Args:
        difficulty: 'easy', 'medium', 'hard', or 'severe'

    Returns:
        aug_params: Dictionary of augmentation parameters
    """
    if difficulty == 'easy':
        return {
            'snr_db': 20,
            'phase_noise_std': 0.1,
            'amp_scale_range': [0.85, 1.15],
            'time_warp_factor': 0.1,
            'subcarrier_dropout': 0.1,
            'apply_prob': 0.5,
        }

    elif difficulty == 'medium':
        return {
            'snr_db': 15,
            'phase_noise_std': 0.2,
            'amp_scale_range': [0.7, 1.3],
            'time_warp_factor': 0.15,
            'subcarrier_dropout': 0.15,
            'apply_prob': 0.7,
            'burst_error_prob': 0.1,
            'burst_length_range': [3, 7],
            'dc_offset': 0.15,
            'frequency_offset_hz': 50,
        }

    elif difficulty == 'hard':
        return {
            'snr_db': 10,
            'phase_noise_std': 0.3,
            'amp_scale_range': [0.6, 1.4],
            'time_warp_factor': 0.2,
            'subcarrier_dropout': 0.2,
            'apply_prob': 0.8,
            'burst_error_prob': 0.2,
            'burst_length_range': [5, 10],
            'dc_offset': 0.2,
            'frequency_offset_hz': 100,
            'temporal_dropout': 0.15,
            'impulse_noise_prob': 0.15,
        }

    elif difficulty == 'severe':
        return {
            'snr_db': 5,
            'phase_noise_std': 0.6,
            'amp_scale_range': [0.3, 2.0],
            'time_warp_factor': 0.4,
            'subcarrier_dropout': 0.35,
            'apply_prob': 0.95,
            'burst_error_prob': 0.6,
            'burst_length_range': [10, 20],
            'dc_offset': 0.5,
            'frequency_offset_hz': 300,
            'temporal_dropout': 0.3,
            'impulse_noise_prob': 0.3,
        }

    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
