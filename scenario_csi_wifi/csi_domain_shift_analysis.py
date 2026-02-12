"""
CSI Domain Shift Analysis - L-inf Norm Additive Noise
======================================================

Creates domain shift for CSI WiFi activity recognition using
additive noise with L-infinity norm constraint.

Three shift methods:
1. Fixed: Worst-case (all perturbations at +/-alpha)
2. Uniform: Random perturbations in [-alpha, +alpha]
3. Mixture: 70% small + 30% large with temporal smoothing
"""

import numpy as np
import pickle
from pathlib import Path
from scipy import ndimage


# ============================================================================
# CSI Data Loading
# ============================================================================

def normalize_csi_per_sample(csi_data):
    """
    Apply per-sample normalization (mean=0, std=1).
    This MUST match the normalization used during training.

    Parameters
    ----------
    csi_data : np.ndarray, shape (n_samples, 150, 104)
        Raw CSI data (magnitude)

    Returns
    -------
    csi_normalized : np.ndarray, shape (n_samples, 150, 104)
    """
    n_samples = csi_data.shape[0]
    csi_normalized = np.zeros_like(csi_data, dtype=np.float32)

    for i in range(n_samples):
        sample = csi_data[i]
        mean = np.mean(sample)
        std = np.std(sample)
        if std == 0:
            std = 1.0
        csi_normalized[i] = (sample - mean) / std

    return csi_normalized


def load_csi_test_data(data_path):
    """
    Load CSI test data from pickle file with SAME preprocessing as training.

    Applies per-sample normalization (mean=0, std=1) to match training.

    Parameters
    ----------
    data_path : str
        Path to pickle file containing test data.

    Returns
    -------
    csi_data : np.ndarray, shape (n_samples, 150, 104)
        CSI sequences (normalized per sample, mean~0, std~1)
    labels : np.ndarray, shape (n_samples,)
        Activity labels (0-3)
    """
    print(f"\nLoading CSI test data from: {data_path}")

    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    if 'test' in data_dict:
        csi_data_raw = data_dict['test']['csi_data']
        labels = np.array(data_dict['test']['labels'], dtype=np.int64)
    else:
        csi_data_raw = data_dict['csi_data']
        labels = np.array(data_dict['labels'], dtype=np.int64)

    if isinstance(csi_data_raw, list):
        print(f"  Converting list to array...")
        csi_data_raw = [np.array(x) for x in csi_data_raw]
        if np.iscomplexobj(csi_data_raw[0]):
            print(f"  Converting complex CSI to magnitude...")
            csi_data_raw = [np.abs(x) for x in csi_data_raw]
        csi_data_raw = np.array(csi_data_raw, dtype=np.float32)
    else:
        if np.iscomplexobj(csi_data_raw):
            print(f"  Converting complex CSI to magnitude...")
            csi_data_raw = np.abs(csi_data_raw)
        csi_data_raw = csi_data_raw.astype(np.float32)

    print(f"  Applying per-sample normalization (mean=0, std=1)...")
    csi_data = normalize_csi_per_sample(csi_data_raw)

    if csi_data.ndim == 2:
        raise ValueError(f"Expected 3D CSI data, got 2D: {csi_data.shape}")

    overall_mean = csi_data.mean()
    overall_std = csi_data.std()

    print(f"  Loaded and normalized CSI data:")
    print(f"    Shape: {csi_data.shape}")
    print(f"    Labels: {labels.shape}")
    print(f"    Classes: {np.unique(labels)} (EMPTY=0, SIT=1, STAND=2, WALK=3)")
    print(f"    NORMALIZED: overall mean={overall_mean:.4f}, overall std={overall_std:.4f}")
    print(f"    Range: [{csi_data.min():.2f}, {csi_data.max():.2f}]")

    for i, activity in enumerate(['EMPTY', 'SIT', 'STAND', 'WALK']):
        count = np.sum(labels == i)
        print(f"    {activity}: {count} samples ({100 * count / len(labels):.1f}%)")

    return csi_data, labels


def get_activity_name(label_idx):
    """Get activity name from label index."""
    activities = ['EMPTY', 'SIT', 'STAND', 'WALK']
    return activities[label_idx]


# ============================================================================
# L-inf Norm Constraint: ||d||_inf <= alpha
# ============================================================================

def compute_alpha_for_linf_norm(data_std, target_multiplier):
    """
    Compute alpha for L-infinity norm constraint: ||d||_inf <= alpha.

    For normalized data (mean=0, std~1):
      * 0.1-0.5 sigma: Mild shift
      * 1.0-2.0 sigma: Medium shift
      * 2.0-5.0 sigma: Severe shift

    Parameters
    ----------
    data_std : float
        Standard deviation of CSI data.
    target_multiplier : float
        Multiplier for std.

    Returns
    -------
    alpha : float
        Maximum perturbation bound.
    """
    alpha = target_multiplier * data_std
    print(f"\n  Alpha computation:")
    print(f"    Data std: {data_std:.4f}")
    if data_std > 10.0:
        print(f"    WARNING: Data std > 10! Is your data normalized?")
        print(f"    For normalized data, std should be approx 1.0")
    print(f"    Multiplier: {target_multiplier} sigma")
    print(f"    Alpha: {alpha:.4f}")
    print(f"    Constraint: ||d||_inf <= {alpha:.4f}")

    return alpha


# ============================================================================
# COVARIATE SHIFT: Additive Noise to CSI Inputs (3 Methods)
# ============================================================================

def _renormalize_per_sample(csi_shifted, n_samples):
    """Re-normalize each sample after shift to match training preprocessing."""
    print(f"  Re-normalizing each sample after shift...")
    for i in range(n_samples):
        sample = csi_shifted[i]
        mean = np.mean(sample)
        std = np.std(sample)
        if std == 0:
            std = 1.0
        csi_shifted[i] = (sample - mean) / std
    return csi_shifted


def apply_covariate_shift_uniform_linf(csi_data, alpha):
    """
    METHOD 1: Uniform perturbations with L-infinity constraint.

    Each CSI element perturbed by d[i,j,k] ~ Uniform[-alpha, +alpha].
    Guarantees: ||d||_inf <= alpha.
    Re-normalizes each sample after shift.

    Parameters
    ----------
    csi_data : np.ndarray, shape (n_samples, 150, 104)
    alpha : float

    Returns
    -------
    csi_shifted, d, linf_norm, mean_abs
    """
    n_samples, seq_len, n_features = csi_data.shape

    d = np.random.uniform(-alpha, alpha, size=(n_samples, seq_len, n_features))
    d = d.astype(np.float32)

    csi_shifted = csi_data + d

    # Compute norms before re-normalization
    linf_norm = np.abs(d).max()
    mean_abs = np.abs(d).mean()

    csi_shifted = _renormalize_per_sample(csi_shifted, n_samples)

    print(f"\n  Uniform shift applied:")
    print(f"    Constraint: ||d||_inf <= {alpha:.4f}")
    print(f"    Achieved ||d||_inf: {linf_norm:.4f}")
    print(f"    Mean |d|: {mean_abs:.4f} (expected ~ {alpha / 2:.4f})")
    print(f"    Each sample re-normalized (mean~0, std~1)")

    return csi_shifted, d, linf_norm, mean_abs


def apply_covariate_shift_fixed_linf(csi_data, alpha):
    """
    METHOD 2: Fixed magnitude perturbations (WORST CASE).

    Each element: d[i,j,k] = +/-alpha (random sign).
    This is the worst case under L-infinity constraint.
    Re-normalizes each sample after shift.

    Parameters
    ----------
    csi_data : np.ndarray, shape (n_samples, 150, 104)
    alpha : float

    Returns
    -------
    csi_shifted, d, linf_norm, mean_abs
    """
    n_samples, seq_len, n_features = csi_data.shape

    signs = np.random.choice([-1, 1], size=(n_samples, seq_len, n_features))
    d = (signs * alpha).astype(np.float32)

    csi_shifted = csi_data + d

    linf_norm = np.abs(d).max()
    mean_abs = np.abs(d).mean()

    csi_shifted = _renormalize_per_sample(csi_shifted, n_samples)

    print(f"\n  Fixed (worst-case) shift applied:")
    print(f"    Constraint: ||d||_inf <= {alpha:.4f}")
    print(f"    Achieved ||d||_inf: {linf_norm:.4f}")
    print(f"    Mean |d|: {mean_abs:.4f} (should equal alpha)")
    print(f"    Each sample re-normalized (mean~0, std~1)")

    return csi_shifted, d, linf_norm, mean_abs


def apply_covariate_shift_mixture_linf(csi_data, alpha, small_ratio=0.7,
                                       temporal_smooth=True, sigma=3):
    """
    METHOD 3: Mixture perturbations - 70% small + 30% large.

    With optional temporal smoothing across sequence.
    This creates realistic domain shift where most features see small
    perturbations (0.1*alpha) and some see large perturbations (up to alpha).
    Re-normalizes each sample after shift.

    Parameters
    ----------
    csi_data : np.ndarray, shape (n_samples, 150, 104)
    alpha : float
    small_ratio : float, default=0.7
    temporal_smooth : bool, default=True
    sigma : float, default=3

    Returns
    -------
    csi_shifted, d, linf_norm, mean_abs
    """
    n_samples, seq_len, n_features = csi_data.shape

    # Vectorized mixture generation (replaces triple-nested loop)
    mask = np.random.random(size=(n_samples, seq_len, n_features)) < small_ratio

    # Small perturbations (10% of alpha)
    small_d = np.random.uniform(-0.1 * alpha, 0.1 * alpha,
                                size=(n_samples, seq_len, n_features))
    # Large perturbations (full alpha)
    large_d = np.random.uniform(-alpha, alpha,
                                size=(n_samples, seq_len, n_features))

    d = np.where(mask, small_d, large_d).astype(np.float32)

    # Temporal smoothing
    if temporal_smooth:
        print(f"  Applying temporal smoothing (sigma={sigma})...")
        for i in range(n_samples):
            for k in range(n_features):
                d[i, :, k] = ndimage.gaussian_filter1d(d[i, :, k], sigma=sigma)

    # Ensure L-inf constraint after smoothing
    current_linf = np.abs(d).max()
    if current_linf > alpha:
        scale_factor = alpha * 0.95 / current_linf
        d = d * scale_factor
        print(f"  Rescaled by {scale_factor:.4f} to satisfy constraint")

    csi_shifted = csi_data + d

    # Compute norms before re-normalization
    linf_norm = np.abs(d).max()
    mean_abs = np.abs(d).mean()

    csi_shifted = _renormalize_per_sample(csi_shifted, n_samples)

    print(f"\n  Mixture shift applied:")
    print(f"    Small ratio: {small_ratio * 100:.0f}% at +/-{0.1 * alpha:.4f}")
    print(f"    Large ratio: {(1 - small_ratio) * 100:.0f}% at +/-{alpha:.4f}")
    print(f"    Temporal smoothing: {temporal_smooth} "
          f"(sigma={sigma if temporal_smooth else 'N/A'})")
    print(f"    Constraint: ||d||_inf <= {alpha:.4f}")
    print(f"    Achieved ||d||_inf: {linf_norm:.4f}")
    print(f"    Mean |d|: {mean_abs:.4f}")
    print(f"    Each sample re-normalized (mean~0, std~1)")

    return csi_shifted, d, linf_norm, mean_abs


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_classification_metrics(y_true, y_pred, activity_names=None):
    """
    Compute classification metrics for CSI activity recognition.

    Returns
    -------
    metrics : dict with accuracy, per_class_accuracy, confusion_matrix
    """
    if activity_names is None:
        activity_names = ['EMPTY', 'SIT', 'STAND', 'WALK']

    accuracy = np.mean(y_true == y_pred)

    per_class_acc = {}
    for i, activity in enumerate(activity_names):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            per_class_acc[activity] = np.mean(y_pred[class_mask] == i)
        else:
            per_class_acc[activity] = 0.0

    n_classes = len(activity_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label, pred_label] += 1

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion,
    }


def print_metrics(metrics, title="Classification Metrics"):
    """Pretty print classification metrics."""
    print(f"\n{title}:")
    print(f"  Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"\n  Per-Class Accuracy:")
    for activity, acc in metrics['per_class_accuracy'].items():
        print(f"    {activity:8s}: {acc * 100:.2f}%")


def print_shift_statistics(d, alpha, method='unknown'):
    """Print statistics about the applied shift."""
    print(f"\n  Shift Statistics ({method}):")
    print(f"    Constraint: ||d||_inf <= {alpha:.4f}")
    print(f"    Actual ||d||_inf: {np.abs(d).max():.4f}")
    print(f"    Mean |d|: {np.abs(d).mean():.4f}")
    print(f"    Std(d): {d.std():.4f}")
    print(f"    Range: [{d.min():.4f}, {d.max():.4f}]")

    if np.abs(d).max() <= alpha * 1.01:
        print(f"    Constraint satisfied")
    else:
        print(f"    Constraint VIOLATED!")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CSI Domain Shift Analysis - L-inf Norm Methods")
    print("=" * 80)

    print("""
Three Shift Methods:

1. FIXED (Worst Case):
   - Each element: d = +/-alpha (random sign)
   - Creates SEVERE domain shift

2. UNIFORM (Moderate):
   - Each element: d ~ U[-alpha, +alpha]
   - Creates MODERATE domain shift

3. MIXTURE (Realistic):
   - 70% small (+/-0.1*alpha) + 30% large (+/-alpha)
   - Temporal smoothing for coherence
   - Creates REALISTIC domain shift

Usage:
    from csi_domain_shift_analysis import (
        load_csi_test_data,
        compute_alpha_for_linf_norm,
        apply_covariate_shift_fixed_linf,
        apply_covariate_shift_uniform_linf,
        apply_covariate_shift_mixture_linf,
        compute_classification_metrics,
    )

    # Load and normalize
    csi_data, labels = load_csi_test_data('lstm_data_prepared_pytorch.pkl')

    # Compute alpha
    data_std = csi_data.std()
    alpha = compute_alpha_for_linf_norm(data_std, target_multiplier=10.0)

    # Apply shift
    csi_shifted, d, linf, mean_abs = apply_covariate_shift_mixture_linf(csi_data, alpha)

    # Evaluate
    predictions = model_predict(csi_shifted)
    metrics = compute_classification_metrics(labels, predictions)
    """)
