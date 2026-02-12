"""
Test Trained Model on Target Domain
Domain Shift Robustness Evaluation

Tests your trained LSTM (from source domain) on target domain
to quantify performance degradation due to domain shift.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from lstm_csi_classifier_pytorch import CSI_LSTM_Classifier
from prepare_lstm_data_pytorch import CSI_LSTM_Dataset
from torch.utils.data import DataLoader


def load_trained_model(model_path='lstm_results_pytorch/best_model.pth', device='cuda'):
    """
    Load the trained model.

    Args:
        model_path: Path to saved model weights
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model ready for inference
    """
    print(f"Loading trained model from {model_path}...")

    model = CSI_LSTM_Classifier(
        input_size=104, hidden_size1=128,
        hidden_size2=64, num_classes=4, dropout=0.3,
    )

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  CUDA not available, using CPU")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully (device: {device})")
    return model


def create_target_domain_loader(target_file='csi_data_target_domain.pkl',
                                batch_size=16):
    """
    Create DataLoader for target domain data.

    IMPORTANT: Target domain data is already normalized and stored as float.
    We set use_magnitude=False and normalize=False to avoid double-processing.

    Args:
        target_file: Target domain data file
        batch_size: Batch size

    Returns:
        target_loader, target_dict
    """
    print(f"\nLoading target domain data from {target_file}...")

    try:
        with open(target_file, 'rb') as f:
            target_dict = pickle.load(f)

        print(f"  Loaded target domain data")
        print(f"  Samples: {target_dict['num_samples']}")
        print(f"  Domain: {target_dict.get('domain', 'unknown')}")
        print(f"  Severity: {target_dict.get('severity', 'unknown')}")
        print(f"  Shift type: {target_dict.get('shift_type', 'unknown')}")

    except FileNotFoundError:
        print(f"ERROR: {target_file} not found!")
        print("Please run create_target_domain.py first!")
        return None, None

    # CRITICAL: Data is already normalized float32 from create_target_domain.py.
    #   use_magnitude=False  -> do NOT call np.abs() (data has negative values)
    #   normalize=False      -> do NOT re-normalize (would wash out domain shift)
    target_dataset = CSI_LSTM_Dataset(
        target_dict['csi_data'],
        target_dict['labels'],
        use_magnitude=False,
        normalize=False,
    )

    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    print(f"  Batches: {len(target_loader)}")
    return target_loader, target_dict


def evaluate_on_target_domain(model, target_loader, device='cuda'):
    """
    Evaluate model on target domain.

    Returns:
        predictions, accuracy, true_labels
    """
    print("\n" + "=" * 70)
    print("EVALUATING ON TARGET DOMAIN")
    print("=" * 70 + "\n")

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_csi, batch_labels in tqdm(target_loader, desc='Testing on Target'):
            batch_csi = batch_csi.to(device)

            outputs = model(batch_csi)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = 100.0 * np.sum(all_predictions == all_labels) / len(all_labels)

    return all_predictions, accuracy, all_labels


def compare_source_vs_target(source_results_file='lstm_results_pytorch/training_results.pkl',
                             target_accuracy=None, target_predictions=None,
                             target_labels=None, output_dir='domain_shift_results'):
    """
    Compare source domain vs target domain performance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("SOURCE vs TARGET DOMAIN COMPARISON")
    print("=" * 70 + "\n")

    activity_names = ['EMPTY', 'SIT', 'STAND', 'WALK']

    # Load source results
    source_accuracy = None
    source_predictions = None
    source_labels = None

    try:
        with open(source_results_file, 'rb') as f:
            source_results = pickle.load(f)
        source_accuracy = source_results['test_accuracy']
        source_predictions = source_results['predictions']
        source_labels = source_results['true_labels']
    except FileNotFoundError:
        print(f"Warning: {source_results_file} not found")
        print("Using only target domain results")

    # Print comparison
    if source_accuracy is not None:
        print("Overall Accuracy:")
        print(f"  Source Domain:  {source_accuracy:.2f}%")
        print(f"  Target Domain:  {target_accuracy:.2f}%")
        print(f"  Performance Drop: {source_accuracy - target_accuracy:.2f}%")
        print()
    else:
        print(f"Target Domain Accuracy: {target_accuracy:.2f}%\n")

    # Per-class comparison
    print("Per-Class Performance:")
    print(f"{'Activity':<10} {'Source':<12} {'Target':<12} {'Drop':<10}")
    print("-" * 50)

    class_drops = []

    for i, activity in enumerate(activity_names):
        target_mask = (target_labels == i)
        target_correct = np.sum((target_predictions == i) & target_mask)
        target_total = np.sum(target_mask)
        target_acc = 100.0 * target_correct / target_total if target_total > 0 else 0

        if source_accuracy is not None:
            source_mask = (source_labels == i)
            source_correct = np.sum((source_predictions == i) & source_mask)
            source_total = np.sum(source_mask)
            source_acc = 100.0 * source_correct / source_total if source_total > 0 else 0

            drop = source_acc - target_acc
            class_drops.append(drop)

            print(f"{activity:<10} {source_acc:>8.2f}%   {target_acc:>8.2f}%   {drop:>6.2f}%")
        else:
            print(f"{activity:<10} {'N/A':<12} {target_acc:>8.2f}%   {'N/A':<10}")

    print()

    # Domain shift analysis
    if source_accuracy is not None and class_drops:
        avg_drop = np.mean(class_drops)
        max_drop = np.max(class_drops)
        min_drop = np.min(class_drops)

        print("Domain Shift Analysis:")
        print(f"  Average drop: {avg_drop:.2f}%")
        print(f"  Max drop: {max_drop:.2f}% ({activity_names[np.argmax(class_drops)]})")
        print(f"  Min drop: {min_drop:.2f}% ({activity_names[np.argmin(class_drops)]})")
        print()

        if avg_drop > 20:
            print(f"  SIGNIFICANT domain shift detected ({avg_drop:.1f}% drop)")
        elif avg_drop > 10:
            print(f"  Moderate domain shift ({avg_drop:.1f}% drop)")
        else:
            print(f"  Small domain shift ({avg_drop:.1f}% drop)")
            print("  Consider using stronger target augmentation")

    # Plot comparison
    plot_domain_comparison(source_predictions, source_labels,
                          target_predictions, target_labels,
                          source_accuracy, target_accuracy,
                          activity_names, output_dir)

    # Save report
    save_domain_shift_report(source_accuracy, target_accuracy,
                             source_predictions, source_labels,
                             target_predictions, target_labels,
                             activity_names, output_dir)


def plot_domain_comparison(source_pred, source_true, target_pred, target_true,
                           source_acc, target_acc, activity_names, output_dir):
    """Plot side-by-side confusion matrices."""
    output_dir = Path(output_dir)

    if source_acc is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        cm_target = confusion_matrix(target_true, target_pred)
        sns.heatmap(cm_target, annot=True, fmt='d', cmap='Reds',
                    xticklabels=activity_names, yticklabels=activity_names,
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Target Domain\nAccuracy: {target_acc:.2f}%')

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cm_source = confusion_matrix(source_true, source_pred)
        sns.heatmap(cm_source, annot=True, fmt='d', cmap='Blues',
                    xticklabels=activity_names, yticklabels=activity_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title(f'Source Domain\nAccuracy: {source_acc:.2f}%')

        cm_target = confusion_matrix(target_true, target_pred)
        sns.heatmap(cm_target, annot=True, fmt='d', cmap='Reds',
                    xticklabels=activity_names, yticklabels=activity_names,
                    ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title(f'Target Domain\nAccuracy: {target_acc:.2f}%')

    plt.tight_layout()
    plt.savefig(output_dir / 'domain_shift_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/domain_shift_comparison.png")
    plt.close()


def save_domain_shift_report(source_acc, target_acc, source_pred, source_true,
                             target_pred, target_true, activity_names, output_dir):
    """Save detailed domain shift report."""
    output_dir = Path(output_dir)

    with open(output_dir / 'domain_shift_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DOMAIN SHIFT ROBUSTNESS EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        import time
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Experiment Setup:\n")
        f.write("-" * 50 + "\n")
        f.write("Source Domain (Training):\n")
        f.write("  - Purpose: Model training\n")
        if source_acc is not None:
            f.write(f"  - Test Accuracy: {source_acc:.2f}%\n\n")
        else:
            f.write("  - Test Accuracy: N/A\n\n")

        f.write("Target Domain (Deployment Simulation):\n")
        f.write("  - Shift type: L-inf additive noise\n")
        f.write(f"  - Test Accuracy: {target_acc:.2f}%\n\n")

        f.write("=" * 70 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 70 + "\n\n")

        if source_acc is not None:
            drop = source_acc - target_acc
            f.write(f"Overall Performance:\n")
            f.write(f"  Source: {source_acc:.2f}%\n")
            f.write(f"  Target: {target_acc:.2f}%\n")
            f.write(f"  Drop:   {drop:.2f}%\n\n")

        f.write("Per-Class Results:\n")
        f.write(f"{'Activity':<10} {'Source':<12} {'Target':<12} {'Drop':<10}\n")
        f.write("-" * 50 + "\n")

        for i, activity in enumerate(activity_names):
            target_mask = (target_true == i)
            target_correct = np.sum((target_pred == i) & target_mask)
            target_total = np.sum(target_mask)
            target_class_acc = (100.0 * target_correct / target_total
                                if target_total > 0 else 0)

            if source_acc is not None and source_pred is not None:
                source_mask = (source_true == i)
                source_correct = np.sum((source_pred == i) & source_mask)
                source_total = np.sum(source_mask)
                source_class_acc = (100.0 * source_correct / source_total
                                    if source_total > 0 else 0)
                class_drop = source_class_acc - target_class_acc

                f.write(f"{activity:<10} {source_class_acc:>8.2f}%   "
                        f"{target_class_acc:>8.2f}%   {class_drop:>6.2f}%\n")
            else:
                f.write(f"{activity:<10} {'N/A':<12} {target_class_acc:>8.2f}%\n")

        f.write("\n" + "=" * 70 + "\n\n")

        if source_acc is not None:
            drop = source_acc - target_acc

            if drop > 20:
                f.write(f"SIGNIFICANT domain shift: {drop:.1f}% performance drop\n\n")
                f.write("Research Implications:\n")
                f.write("  1. Baseline LSTM is NOT robust to domain shift\n")
                f.write(f"  2. Standard LSTM achieves {source_acc:.1f}% on source domain but\n")
                f.write(f"     degrades to {target_acc:.1f}% under domain shift "
                        f"({drop:.1f}% drop)\n")
            elif drop > 10:
                f.write(f"Moderate domain shift: {drop:.1f}% performance drop\n\n")
                f.write("Good for demonstrating value of robust approach.\n")
            else:
                f.write(f"Small domain shift: {drop:.1f}% performance drop\n\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"  Saved: {output_dir}/domain_shift_report.txt")


def main():
    """Main execution."""
    print("=" * 70)
    print("TEST TRAINED MODEL ON TARGET DOMAIN")
    print("Domain Shift Robustness Evaluation")
    print("=" * 70 + "\n")

    config = {
        'model_path': 'lstm_results_pytorch/best_model.pth',
        'target_file': 'csi_data_target_domain.pkl',
        'source_results': 'lstm_results_pytorch/training_results.pkl',
        'batch_size': 16,
        'output_dir': 'domain_shift_results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load trained model
    model = load_trained_model(config['model_path'], config['device'])

    # Step 2: Load target domain data
    target_loader, target_dict = create_target_domain_loader(
        config['target_file'], config['batch_size'])

    if target_loader is None:
        return

    # Step 3: Evaluate on target domain
    target_predictions, target_accuracy, target_labels = evaluate_on_target_domain(
        model, target_loader, config['device'])

    print(f"\n{'=' * 70}")
    print(f"TARGET DOMAIN ACCURACY: {target_accuracy:.2f}%")
    print(f"{'=' * 70}\n")

    # Step 4: Compare source vs target
    compare_source_vs_target(
        config['source_results'],
        target_accuracy,
        target_predictions,
        target_labels,
        config['output_dir'],
    )

    # Summary
    print("\n" + "=" * 70)
    print("DOMAIN SHIFT EVALUATION COMPLETE")
    print("=" * 70 + "\n")

    print("Files created:")
    print(f"  {output_dir}/domain_shift_comparison.png")
    print(f"  {output_dir}/domain_shift_report.txt")
    print("\n" + "=" * 70 + "\n")

    return target_accuracy, target_predictions, target_labels


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found!")
        print(f"Details: {e}\n")
        print("Make sure you have:")
        print("  1. Trained model: lstm_results_pytorch/best_model.pth")
        print("  2. Target data: csi_data_target_domain.pkl")
        print("\nRun these first:")
        print("  python train_csi_lstm_pytorch.py")
        print("  python create_target_domain.py")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
