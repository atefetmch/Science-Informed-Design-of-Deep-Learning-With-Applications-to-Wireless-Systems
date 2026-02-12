"""
Create Target Domain Dataset with L-inf Norm Domain Shift


Three shift methods:
1. Fixed: Worst-case (all perturbations at +/-alpha)
2. Uniform: Random perturbations in [-alpha, +alpha]
3. Mixture: 70% small + 30% large with temporal smoothing
"""

import numpy as np
import pickle
from pathlib import Path

from csi_domain_shift_analysis import (
    normalize_csi_per_sample,
    compute_alpha_for_linf_norm,
    apply_covariate_shift_fixed_linf,
    apply_covariate_shift_uniform_linf,
    apply_covariate_shift_mixture_linf,
    print_shift_statistics,
)


# ============================================================================
# Alpha Presets for Different Severity Levels
# ============================================================================

SHIFT_PRESETS = {
    'mild': {
        'description': 'Mild domain shift (0.5 sigma)',
        'target_multiplier': 0.5,
    },
    'medium': {
        'description': 'Medium domain shift (1.0 sigma)',
        'target_multiplier': 1.0,
    },
    'hard': {
        'description': 'Hard domain shift (2.0 sigma)',
        'target_multiplier': 2.0,
    },
    'severe': {
        'description': 'Severe domain shift (3.0 sigma)',
        'target_multiplier': 3.0,
    },
}


def create_target_domain_dataset(
    source_file='lstm_data_prepared_pytorch.pkl',
    output_file='csi_data_target_domain.pkl',
    shift_method='all',
    severity='hard',
    custom_multiplier=None,
):
    """
    Create target domain dataset using L-inf norm additive perturbations.

    Args:
        source_file: Original CSI data file (pickle)
        output_file: Output file for target domain data
        shift_method: 'fixed', 'uniform', 'mixture', or 'all'
        severity: Preset severity level ('mild', 'medium', 'hard', 'severe')
        custom_multiplier: Override severity preset with custom sigma multiplier

    Returns:
        target_dict: Target domain dataset dictionary
    """
    print("=" * 70)
    print("CREATE TARGET DOMAIN DATASET - L-inf NORM DOMAIN SHIFT")
    print("=" * 70 + "\n")

    # Step 1: Load original data
    print(f"Step 1: Loading original CSI data from {source_file}...")

    try:
        with open(source_file, 'rb') as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {source_file} not found!")
        return None

    # Extract CSI data and labels (handle different formats)
    if 'test' in data_dict:
        csi_data_raw = data_dict['test']['csi_data']
        labels = np.array(data_dict['test']['labels'], dtype=np.int64)
    else:
        csi_data_raw = data_dict['csi_data']
        labels = np.array(data_dict['labels'], dtype=np.int64)

    if isinstance(csi_data_raw, list):
        csi_data_raw = [np.array(x) for x in csi_data_raw]
        if np.iscomplexobj(csi_data_raw[0]):
            print("  Converting complex CSI to magnitude...")
            csi_data_raw = [np.abs(x) for x in csi_data_raw]
        csi_data_raw = np.array(csi_data_raw, dtype=np.float32)
    else:
        if np.iscomplexobj(csi_data_raw):
            print("  Converting complex CSI to magnitude...")
            csi_data_raw = np.abs(csi_data_raw)
        csi_data_raw = csi_data_raw.astype(np.float32)

    print("  Applying per-sample normalization (mean=0, std=1)...")
    csi_data = normalize_csi_per_sample(csi_data_raw)

    n_samples = csi_data.shape[0]
    activities = data_dict.get('activities', ['EMPTY', 'SIT', 'STAND', 'WALK'])

    print(f"  Loaded {n_samples} samples, shape: {csi_data.shape}")
    print(f"  Activities: {activities}")
    print(f"  Normalized: mean={csi_data.mean():.4f}, std={csi_data.std():.4f}\n")

    # Step 2: Compute alpha
    print(f"Step 2: Computing L-inf perturbation bound (alpha)...")

    if custom_multiplier is not None:
        target_multiplier = custom_multiplier
        severity_desc = f'Custom ({custom_multiplier} sigma)'
    else:
        preset = SHIFT_PRESETS[severity]
        target_multiplier = preset['target_multiplier']
        severity_desc = preset['description']

    data_std = csi_data.std()
    alpha = compute_alpha_for_linf_norm(data_std, target_multiplier)

    print(f"\n  Severity: {severity_desc}")
    print(f"  ||d||_inf <= {alpha:.4f}")

    # Step 3: Apply domain shift(s)
    print(f"\nStep 3: Applying L-inf domain shift...")

    if shift_method == 'all':
        methods_to_apply = ['fixed', 'uniform', 'mixture']
    else:
        methods_to_apply = [shift_method]

    shift_results = {}

    for method in methods_to_apply:
        print(f"\n{'-' * 50}")
        print(f"  Applying {method.upper()} shift (alpha={alpha:.4f})...")
        print(f"{'-' * 50}")

        if method == 'fixed':
            csi_shifted, d, linf_norm, mean_abs = apply_covariate_shift_fixed_linf(
                csi_data, alpha)
        elif method == 'uniform':
            csi_shifted, d, linf_norm, mean_abs = apply_covariate_shift_uniform_linf(
                csi_data, alpha)
        elif method == 'mixture':
            csi_shifted, d, linf_norm, mean_abs = apply_covariate_shift_mixture_linf(
                csi_data, alpha, small_ratio=0.7, temporal_smooth=True, sigma=3)
        else:
            raise ValueError(f"Unknown shift method: {method}")

        print_shift_statistics(d, alpha, method=method)

        shift_results[method] = {
            'csi_data': csi_shifted,
            'perturbation': d,
            'linf_norm': linf_norm,
            'mean_abs': mean_abs,
        }

    # Step 4: Create target domain dict
    print(f"\n{'=' * 50}")
    print(f"Step 4: Creating target domain dataset...")

    target_dict = {
        'labels': labels,
        'activities': activities,
        'num_samples': n_samples,
        'domain': 'target',
        'shift_type': 'L-inf additive noise',
        'alpha': float(alpha),
        'target_multiplier': float(target_multiplier),
        'severity': severity_desc,
        'data_std': float(data_std),
    }

    if shift_method == 'all':
        for method, result in shift_results.items():
            target_dict[f'csi_data_{method}'] = result['csi_data']
            target_dict[f'perturbation_{method}'] = result['perturbation']
            target_dict[f'linf_norm_{method}'] = result['linf_norm']
            target_dict[f'mean_abs_{method}'] = result['mean_abs']
        # Default csi_data points to mixture (most realistic)
        target_dict['csi_data'] = shift_results['mixture']['csi_data']
        target_dict['shift_methods'] = list(shift_results.keys())
    else:
        method = methods_to_apply[0]
        result = shift_results[method]
        target_dict['csi_data'] = result['csi_data']
        target_dict['perturbation'] = result['perturbation']
        target_dict['linf_norm'] = result['linf_norm']
        target_dict['mean_abs'] = result['mean_abs']
        target_dict['shift_method'] = method

    # Also keep original (source domain) for comparison
    target_dict['csi_data_source'] = csi_data

    # Step 5: Save
    print(f"\nStep 5: Saving target domain dataset to {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(target_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = Path(output_file).stat().st_size / (1024 ** 2)
    print(f"  Saved: {output_file} ({file_size:.1f} MB)\n")

    # Step 6: Summary
    print("Step 6: Source vs Target Domain Comparison...")
    print("-" * 60)

    sample_idx = 0
    orig_sample = csi_data[sample_idx]

    print(f"\n{'Method':<12} {'||d||_inf':<12} {'Mean|d|':<12} "
          f"{'Shifted mean':<14} {'Shifted std':<12}")
    print("-" * 60)
    print(f"{'Source':<12} {'--':<12} {'--':<12} "
          f"{orig_sample.mean():<14.4f} {orig_sample.std():<12.4f}")

    for method, result in shift_results.items():
        shifted = result['csi_data'][sample_idx]
        print(f"{method:<12} {result['linf_norm']:<12.4f} {result['mean_abs']:<12.4f} "
              f"{shifted.mean():<14.4f} {shifted.std():<12.4f}")

    print("\n" + "=" * 70)
    print("TARGET DOMAIN DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"\n  File: {output_file}")
    print(f"  Samples: {n_samples}")
    print(f"  Severity: {severity_desc}")
    print(f"  Alpha (||d||_inf): {alpha:.4f}")
    print(f"  Shift type: L-inf additive noise")
    if shift_method == 'all':
        print(f"  Methods: fixed, uniform, mixture")
        print(f"  Default csi_data: mixture (most realistic)")
    else:
        print(f"  Method: {shift_method}")

    print(f"\nDomain Shift Scenario:")
    print(f"  SOURCE: Normalized CSI data (training distribution)")
    print(f"  TARGET: CSI + L-inf bounded noise (deployment shift)")
    print(f"  Constraint: ||d||_inf <= {alpha:.4f} ({target_multiplier} sigma)")

    print(f"\nKeys in saved dict:")
    for key in sorted(target_dict.keys()):
        val = target_dict[key]
        if isinstance(val, np.ndarray):
            print(f"  '{key}': ndarray {val.shape} {val.dtype}")
        else:
            print(f"  '{key}': {val}")

    print(f"\nNext step: python test_on_target_domain.py")
    print("=" * 70 + "\n")

    return target_dict


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create target domain dataset with L-inf norm domain shift')
    parser.add_argument('--source', type=str, default='lstm_data_prepared_pytorch.pkl',
                        help='Source data file (prepared split with test key)')
    parser.add_argument('--output', type=str, default='csi_data_target_domain.pkl',
                        help='Output file for target domain')
    parser.add_argument('--method', type=str, default='mixture',
                        choices=['fixed', 'uniform', 'mixture', 'all'],
                        help='Shift method (default: all three)')
    parser.add_argument('--severity', type=str, default='hard',
                        choices=['mild', 'medium', 'hard', 'severe'],
                        help='Shift severity preset')
    parser.add_argument('--multiplier', type=float, default=None,
                        help='Custom sigma multiplier (overrides --severity)')

    args = parser.parse_args()

    return create_target_domain_dataset(
        source_file=args.source,
        output_file=args.output,
        shift_method=args.method,
        severity=args.severity,
        custom_multiplier=args.multiplier,
    )


if __name__ == '__main__':
    try:
        main()
        print("Target domain dataset ready!")
        print("\nUsage examples:")
        print("  python create_target_domain.py --severity severe --method all")
        print("  python create_target_domain.py --severity medium --method mixture")
        print("  python create_target_domain.py --multiplier 7.5 --method uniform")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()