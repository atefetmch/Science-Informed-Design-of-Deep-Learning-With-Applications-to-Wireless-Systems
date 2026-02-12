"""
Complete Training Script for CSI LSTM Classifier
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import time

from lstm_csi_classifier_pytorch import (
    CSI_LSTM_Classifier,
    train_model,
    test_model,
    save_model_summary,
)
from prepare_lstm_data_pytorch import create_dataloaders


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("CSI LSTM CLASSIFIER - PYTORCH TRAINING")
    print("=" * 70 + "\n")

    # Configuration
    config = {
        'input_size': 104,
        'hidden_size1': 128,
        'hidden_size2': 64,
        'num_classes': 4,
        'dropout': 0.3,
        'num_epochs': 20,
        'batch_size': 16,
        'learning_rate': 0.001,
        'output_dir': 'lstm_results_pytorch',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Step 1: Load prepared data
    print("Step 1: Loading prepared data...")

    try:
        with open('lstm_data_prepared_pytorch.pkl', 'rb') as f:
            split_dict = pickle.load(f)
        print(f"  Loaded split data")
        print(f"  Train: {len(split_dict['train']['labels'])} samples")
        print(f"  Val:   {len(split_dict['val']['labels'])} samples")
        print(f"  Test:  {len(split_dict['test']['labels'])} samples\n")
    except FileNotFoundError:
        print("ERROR: lstm_data_prepared_pytorch.pkl not found!")
        print("Please run prepare_lstm_data_pytorch.py first!")
        return

    # Step 2: Create DataLoaders
    print("Step 2: Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_dict, batch_size=config['batch_size'], num_workers=0)

    # Step 3: Create model
    print("\nStep 3: Creating model...")
    model = CSI_LSTM_Classifier(
        input_size=config['input_size'],
        hidden_size1=config['hidden_size1'],
        hidden_size2=config['hidden_size2'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
    )

    print(f"  Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {config['device']}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    save_model_summary(model, output_dir)

    # Step 4: Train
    print("\nStep 4: Training model...")
    start_time = time.time()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        output_dir=config['output_dir'],
    )

    training_time = time.time() - start_time

    # Step 5: Test
    print("\nStep 5: Testing model...")
    predictions, accuracy, true_labels = test_model(
        model=model,
        test_loader=test_loader,
        device=config['device'],
        output_dir=config['output_dir'],
    )

    # Results summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Training Time: {training_time / 60:.1f} minutes\n")

    activity_names = ['EMPTY', 'SIT', 'STAND', 'WALK']
    print("Per-Class Performance:")
    print(f"{'Activity':<12} {'Accuracy':<10} {'Correct/Total':<15}")
    print("-" * 40)

    for i, activity in enumerate(activity_names):
        class_mask = (true_labels == i)
        class_correct = np.sum((predictions == i) & class_mask)
        class_total = np.sum(class_mask)
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
        print(f"{activity:<12} {class_acc:>8.2f}% {class_correct:>7d}/{class_total:<7d}")

    print("=" * 70 + "\n")

    # Step 6: Save results
    print("Step 6: Saving results...")

    results = {
        'config': config,
        'history': history,
        'test_accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'training_time': training_time,
    }

    with open(output_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open(output_dir / 'training_summary.txt', 'w') as f:
        f.write("LSTM Training Summary - PyTorch\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Dataset:\n")
        f.write(f"  Train: {len(split_dict['train']['labels'])} samples\n")
        f.write(f"  Val:   {len(split_dict['val']['labels'])} samples\n")
        f.write(f"  Test:  {len(split_dict['test']['labels'])} samples\n\n")

        f.write("Results:\n")
        f.write(f"  Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"  Training Time: {training_time / 60:.1f} min\n\n")

        f.write("Per-Class Performance:\n")
        for i, activity in enumerate(activity_names):
            class_mask = (true_labels == i)
            class_correct = np.sum((predictions == i) & class_mask)
            class_total = np.sum(class_mask)
            class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
            f.write(f"  {activity:<8s}: {class_acc:>6.2f}% ({class_correct}/{class_total})\n")

    print(f"  Results saved to: {output_dir}/")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70 + "\n")

    print("Files saved:")
    print(f"  {output_dir}/best_model.pth")
    print(f"  {output_dir}/final_model.pth")
    print(f"  {output_dir}/training_progress.png")
    print(f"  {output_dir}/confusion_matrix.png")
    print(f"  {output_dir}/training_results.pkl")
    print(f"  {output_dir}/training_summary.txt")

    print(f"\nNext step: python create_target_domain.py")

    return model, results


if __name__ == '__main__':
    try:
        model, results = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
