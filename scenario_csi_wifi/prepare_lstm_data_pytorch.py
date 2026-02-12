"""
Prepare LSTM Data for PyTorch

Creates train/val/test split and PyTorch Dataset classes.
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import random

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class CSI_LSTM_Dataset(Dataset):
    """
    PyTorch Dataset for CSI LSTM classifier.

    Handles:
    - Magnitude extraction from complex CSI
    - Per-sample normalization (optional, for pre-normalized data)
    - Returns data as (time, features) = (150, 104)
    - After batching: (batch, 150, 104) for LSTM with batch_first=True
    """

    def __init__(self, csi_data: List[np.ndarray], labels: np.ndarray,
                 use_magnitude: bool = True, normalize: bool = True):
        """
        Args:
            csi_data: List of CSI matrices (150, 104)
            labels: Array of labels (0-3)
            use_magnitude: If True, take magnitude of complex data
            normalize: If True, apply per-sample normalization (mean=0, std=1).
                       Set to False when data is already normalized (e.g.,
                       pre-processed target domain data).
        """
        self.csi_data = csi_data
        self.labels = labels
        self.use_magnitude = use_magnitude
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        csi = self.csi_data[idx]

        if self.use_magnitude and np.iscomplexobj(csi):
            csi = np.abs(csi)

        csi = np.asarray(csi, dtype=np.float32)

        if self.normalize:
            mean = np.mean(csi)
            std = np.std(csi)
            if std == 0:
                std = 1.0
            csi = (csi - mean) / std

        # Shape: (150, 104) = (seq_len, input_size)
        csi_tensor = torch.FloatTensor(csi)
        label_tensor = torch.LongTensor([self.labels[idx]])[0]

        return csi_tensor, label_tensor


def stratified_split(data_dict: Dict, train_ratio: float = 0.70,
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict:
    """
    Create stratified train/val/test split.

    Args:
        data_dict: Dictionary with 'csi_data', 'labels', 'activities'
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        split_dict: Dictionary with train/val/test data and indices
    """
    print("\n" + "=" * 70)
    print("CREATING STRATIFIED SPLIT")
    print("=" * 70 + "\n")

    csi_data = data_dict['csi_data']
    labels = data_dict['labels']
    activities = data_dict['activities']
    num_samples = len(labels)

    print(f"Total samples: {num_samples}")
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, "
          f"Test={test_ratio:.0%}\n")

    train_indices = []
    val_indices = []
    test_indices = []

    for class_id, activity in enumerate(activities):
        class_mask = (labels == class_id)
        class_indices = np.where(class_mask)[0]
        num_class_samples = len(class_indices)

        print(f"{activity}: {num_class_samples} samples")

        np.random.shuffle(class_indices)

        num_train = int(train_ratio * num_class_samples)
        num_val = int(val_ratio * num_class_samples)

        train_indices.extend(class_indices[:num_train])
        val_indices.extend(class_indices[num_train:num_train + num_val])
        test_indices.extend(class_indices[num_train + num_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    print(f"\n  Split created:")
    print(f"  Train: {len(train_indices)} samples "
          f"({100 * len(train_indices) / num_samples:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples "
          f"({100 * len(val_indices) / num_samples:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples "
          f"({100 * len(test_indices) / num_samples:.1f}%)")

    # Verify no overlap
    assert len(set(train_indices) & set(test_indices)) == 0, "Train/test overlap!"
    assert len(set(train_indices) & set(val_indices)) == 0, "Train/val overlap!"
    assert len(set(val_indices) & set(test_indices)) == 0, "Val/test overlap!"
    print("  No overlap between splits")

    train_data = [csi_data[i] for i in train_indices]
    train_labels = labels[train_indices]

    val_data = [csi_data[i] for i in val_indices]
    val_labels = labels[val_indices]

    test_data = [csi_data[i] for i in test_indices]
    test_labels = labels[test_indices]

    split_dict = {
        'train': {
            'csi_data': train_data,
            'labels': train_labels,
            'indices': train_indices,
        },
        'val': {
            'csi_data': val_data,
            'labels': val_labels,
            'indices': val_indices,
        },
        'test': {
            'csi_data': test_data,
            'labels': test_labels,
            'indices': test_indices,
        },
        'activities': activities,
    }

    return split_dict


def create_dataloaders(split_dict: Dict, batch_size: int = 16,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.

    Args:
        split_dict: Dictionary from stratified_split()
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 for Windows)

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 70)
    print("CREATING PYTORCH DATALOADERS")
    print("=" * 70 + "\n")

    train_dataset = CSI_LSTM_Dataset(
        split_dict['train']['csi_data'],
        split_dict['train']['labels'],
    )

    val_dataset = CSI_LSTM_Dataset(
        split_dict['val']['csi_data'],
        split_dict['val']['labels'],
    )

    test_dataset = CSI_LSTM_Dataset(
        split_dict['test']['csi_data'],
        split_dict['test']['labels'],
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    print(f"\nDataLoader configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def verify_data_format(loader: DataLoader, name: str = "Train"):
    """Verify the data format from a DataLoader."""
    print(f"\n{name} Data Format Check:")
    print("-" * 50)

    for batch_data, batch_labels in loader:
        print(f"  Batch CSI shape: {batch_data.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        print(f"  CSI dtype: {batch_data.dtype}")
        print(f"  Labels dtype: {batch_labels.dtype}")
        print(f"  Labels range: {batch_labels.min()}-{batch_labels.max()}")

        expected_shape = (batch_data.shape[0], 150, 104)
        if batch_data.shape == expected_shape:
            print(f"  Shape correct: (batch, time, features) = {batch_data.shape}")
        else:
            print(f"  Shape unexpected: got {batch_data.shape}, "
                  f"expected (batch, 150, 104)")
        break


def save_prepared_data(split_dict: Dict,
                       output_file: str = 'lstm_data_prepared_pytorch.pkl'):
    """Save the prepared data for future use."""
    print(f"\nSaving prepared data to {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(split_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saved: {output_file}")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare CSI data for PyTorch LSTM training')
    parser.add_argument('--input', type=str, default='csi_data_augmented_pytorch.pkl',
                        help='Input pickle file (augmented CSI data)')
    parser.add_argument('--output', type=str, default='lstm_data_prepared_pytorch.pkl',
                        help='Output pickle file (split data)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for DataLoaders')

    args = parser.parse_args()

    print("=" * 70)
    print("CSI DATA PREPARATION FOR PYTORCH LSTM")
    print("=" * 70 + "\n")

    # Step 1: Load data
    print(f"Step 1: Loading CSI data from {args.input}...")

    from load_matlab_csi_data import load_from_pickle

    try:
        data_dict = load_from_pickle(args.input)
        print(f"  Loaded {data_dict['num_samples']} samples\n")
    except FileNotFoundError:
        print(f"ERROR: {args.input} not found!")
        print("Please run the appropriate previous step first.")
        return

    # Step 2: Stratified split
    print("Step 2: Creating train/val/test split...")
    split_dict = stratified_split(
        data_dict, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # Step 3: DataLoaders
    print("\nStep 3: Creating PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_dict, batch_size=args.batch_size, num_workers=0)

    # Step 4: Verify
    print("\nStep 4: Verifying data format...")
    verify_data_format(train_loader, "Train")
    verify_data_format(val_loader, "Val")
    verify_data_format(test_loader, "Test")

    # Step 5: Save
    print("\nStep 5: Saving prepared data...")
    save_prepared_data(split_dict, args.output)

    # Summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70 + "\n")

    print(f"  Total samples: {data_dict['num_samples']}")
    print(f"  Train: {len(split_dict['train']['labels'])} samples")
    print(f"  Val:   {len(split_dict['val']['labels'])} samples")
    print(f"  Test:  {len(split_dict['test']['labels'])} samples")
    print(f"  Activities: {split_dict['activities']}")
    print(f"  Data format: (150, 104) = (time, features) - ready for LSTM")
    print(f"  Normalization: Per-sample (mean=0, std=1)")
    print(f"\nSaved: {args.output}")
    print(f"\nNext step: python train_csi_lstm_pytorch.py")

    return split_dict, train_loader, val_loader, test_loader


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
