"""
Create Augmented CSI Dataset

Makes dataset more challenging by applying various augmentations
to simulate realistic wireless channel conditions.
"""

import numpy as np
import pickle
from pathlib import Path
from csi_augmentation import CSIAugmentor, get_augmentation_params


def calculate_class_separability(data_dict, activities):
    """
    Calculate pairwise distances between activity classes
    to verify augmentation is making the dataset harder.
    """
    print("\nChecking class separability...")

    samples = {}
    for i, activity in enumerate(activities):
        class_mask = (data_dict['labels'] == i)
        class_indices = np.where(class_mask)[0]
        if len(class_indices) > 0:
            sample = data_dict['csi_data'][class_indices[0]]
            samples[activity] = np.abs(sample).flatten()

    distances = {}
    activity_pairs = [
        ('EMPTY', 'SIT'), ('EMPTY', 'STAND'), ('EMPTY', 'WALK'),
        ('SIT', 'STAND'), ('SIT', 'WALK'), ('STAND', 'WALK')
    ]

    print("\nPairwise Euclidean Distances:")
    for act1, act2 in activity_pairs:
        if act1 in samples and act2 in samples:
            dist = np.linalg.norm(samples[act1] - samples[act2])
            distances[f"{act1} vs {act2}"] = dist
            print(f"  {act1} vs {act2:5s}: {dist:,.2f}")

    avg_dist = np.mean(list(distances.values()))
    print(f"\n  Average distance: {avg_dist:,.2f}")

    if avg_dist < 40000:
        print("  Distances reduced - dataset is more challenging")
    elif avg_dist < 50000:
        print("  Moderate reduction - consider increasing augmentation")
    else:
        print("  Distances still high - increase augmentation strength")

    return distances, avg_dist


def create_augmented_dataset(difficulty='hard',
                             input_file='csi_data_python.pkl',
                             output_file='csi_data_augmented_pytorch.pkl'):
    """
    Create augmented dataset from original data.

    Args:
        difficulty: 'easy', 'medium', 'hard', or 'severe'
        input_file: Input pickle file with original data
        output_file: Output pickle file for augmented data
    """
    print("=" * 70)
    print("CSI DATA AUGMENTATION PIPELINE")
    print("=" * 70 + "\n")

    # Step 1: Load original data
    print(f"Step 1: Loading original data from {input_file}...")

    try:
        with open(input_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"  Loaded {data_dict['num_samples']} samples")
        print(f"  Activities: {data_dict['activities']}\n")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        print("Please run load_matlab_csi_data.py first!")
        return

    # Step 2: Set augmentation parameters
    print(f"Step 2: Setting augmentation parameters (difficulty: {difficulty.upper()})...")
    aug_params = get_augmentation_params(difficulty)

    print("\nAugmentation Settings:")
    for key, value in aug_params.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Step 3: Check original separability
    print("\nStep 3: Checking original class separability...")
    original_distances, original_avg_dist = calculate_class_separability(
        data_dict, data_dict['activities']
    )

    if original_avg_dist > 60000:
        print(f"\n  Classes are VERY separable (avg dist: {original_avg_dist:.0f})")
        print(f"  Target: Reduce to < 40000 with '{difficulty}' augmentation")

    # Step 4: Apply augmentations
    print(f"\nStep 4: Applying augmentations to all {data_dict['num_samples']} samples...")
    print(f"This may take 2-5 minutes...\n")

    augmentor = CSIAugmentor(aug_params)
    augmented_csi_data = augmentor.augment_dataset(data_dict['csi_data'], verbose=True)

    # Step 5: Create augmented data dict
    print("\nStep 5: Creating augmented dataset...")

    augmented_dict = {
        'csi_data': augmented_csi_data,
        'labels': data_dict['labels'],
        'filenames': data_dict['filenames'],
        'activities': data_dict['activities'],
        'num_samples': data_dict['num_samples'],
        'augmentation_params': aug_params,
        'augmentation_difficulty': difficulty,
    }

    # Step 6: Check augmented separability
    print("\nStep 6: Checking augmented class separability...")
    augmented_distances, augmented_avg_dist = calculate_class_separability(
        augmented_dict, augmented_dict['activities']
    )

    reduction = (original_avg_dist - augmented_avg_dist) / original_avg_dist * 100
    print(f"\nSeparability reduction: {reduction:.1f}%")
    print(f"  Original average: {original_avg_dist:,.0f}")
    print(f"  Augmented average: {augmented_avg_dist:,.0f}")

    if reduction > 20:
        print(f"  Significant reduction! Dataset is now more challenging")
    elif reduction > 10:
        print(f"  Moderate reduction - may need stronger augmentation")
    else:
        print(f"  Small reduction - consider 'severe' difficulty")

    # Step 7: Save augmented dataset
    print(f"\nStep 7: Saving augmented dataset to {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(augmented_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = Path(output_file).stat().st_size / (1024 ** 2)
    print(f"  Saved: {output_file} ({file_size:.1f} MB)\n")

    # Summary
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70 + "\n")

    print(f"  Applied {difficulty.upper()} augmentation to {data_dict['num_samples']} samples")
    print(f"  Reduced class separability by {reduction:.1f}%")
    print(f"  Saved to: {output_file}")

    print("\nNext steps:")
    print("  1. Prepare data:  python prepare_lstm_data_pytorch.py")
    print("  2. Train LSTM:    python train_csi_lstm_pytorch.py")

    expected = {'easy': '90-95%', 'medium': '85-90%', 'hard': '75-85%', 'severe': '65-75%'}
    print(f"  Expected accuracy: {expected.get(difficulty, 'N/A')}")

    print("\n" + "=" * 70 + "\n")

    return augmented_dict


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Create augmented CSI dataset')
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard', 'severe'],
                        help='Augmentation difficulty level')
    parser.add_argument('--input', type=str, default='csi_data_python.pkl',
                        help='Input pickle file')
    parser.add_argument('--output', type=str, default='csi_data_augmented_pytorch.pkl',
                        help='Output pickle file')

    args = parser.parse_args()

    return create_augmented_dataset(
        difficulty=args.difficulty,
        input_file=args.input,
        output_file=args.output,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
