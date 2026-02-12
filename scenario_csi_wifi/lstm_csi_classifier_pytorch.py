"""
PyTorch LSTM Classifier for CSI Data

Architecture: 104 -> LSTM(128) -> LSTM(64) -> FC(4)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm
import time


class CSI_LSTM_Classifier(nn.Module):
    """
    LSTM Classifier for CSI Activity Recognition.

    Architecture:
        Input (104 features) ->
        LSTM1 (128 hidden units, sequence output) ->
        Dropout (0.3) ->
        LSTM2 (64 hidden units, last output) ->
        Dropout (0.3) ->
        FC (4 classes) ->
        Softmax
    """

    def __init__(self, input_size=104, hidden_size1=128, hidden_size2=64,
                 num_classes=4, dropout=0.3):
        super(CSI_LSTM_Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_classes = num_classes

        self.lstm1 = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size1,
            num_layers=1, batch_first=True, dropout=0,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1, hidden_size=hidden_size2,
            num_layers=1, batch_first=True, dropout=0,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
               In our case: (batch_size, 150, 104)

        Returns:
            output: Class logits (batch_size, num_classes)
        """
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        last_output = lstm2_out[:, -1, :]
        last_output = self.dropout2(last_output)

        output = self.fc(last_output)
        return output

    def forward_with_states(self, x):
        """
        Forward pass that returns internal LSTM states.
        For state extraction and DMDc modeling.

        Returns:
            output, (h1, c1), lstm1_outputs, (h2, c2)
        """
        lstm1_out, (h1, c1) = self.lstm1(x)
        lstm1_out_dropped = self.dropout1(lstm1_out)

        lstm2_out, (h2, c2) = self.lstm2(lstm1_out_dropped)
        last_output = lstm2_out[:, -1, :]
        last_output = self.dropout2(last_output)

        output = self.fc(last_output)
        return output, (h1, c1), lstm1_out, (h2, c2)


def train_model(model, train_loader, val_loader, num_epochs=20,
                learning_rate=0.001, device='cuda', output_dir='lstm_results'):
    """
    Train the LSTM model.

    Args:
        model: CSI_LSTM_Classifier instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: 'cuda' or 'cpu'
        output_dir: Directory to save plots and models

    Returns:
        model: Trained model
        history: Training history dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Plots will be saved to: {output_dir}/")
    print(f"\nArchitecture: {model.input_size} -> {model.hidden_size1} -> "
          f"{model.hidden_size2} -> {model.num_classes}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }

    print("\n=== Training LSTM ===\n")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_csi, batch_labels in train_pbar:
            batch_csi = batch_csi.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_csi)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_csi.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_csi, batch_labels in val_loader:
                batch_csi = batch_csi.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_csi)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item() * batch_csi.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, output_dir / 'best_model.pth')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  Loaded best model (Val Acc: {best_val_acc:.2f}%)")

    torch.save(model.state_dict(), output_dir / 'final_model.pth')

    plot_training_history(history, output_dir)

    return model, history


def test_model(model, test_loader, device='cuda', output_dir='lstm_results'):
    """
    Test the trained model.

    Returns:
        predictions, accuracy, all_labels
    """
    print("\n=== Testing ===\n")

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_csi, batch_labels in tqdm(test_loader, desc='Testing'):
            batch_csi = batch_csi.to(device)

            outputs = model(batch_csi)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = 100.0 * np.sum(all_predictions == all_labels) / len(all_labels)

    print(f"\nTest Accuracy: {accuracy:.2f}%\n")

    plot_confusion_matrix(all_labels, all_predictions, accuracy, output_dir)

    activity_names = ['EMPTY', 'SIT', 'STAND', 'WALK']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=activity_names, digits=4))

    return all_predictions, accuracy, all_labels


def plot_training_history(history, output_dir):
    """Plot and save training history."""
    output_dir = Path(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: training_progress.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, accuracy, output_dir):
    """Plot and save confusion matrix."""
    output_dir = Path(output_dir)
    activity_names = ['EMPTY', 'SIT', 'STAND', 'WALK']
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'LSTM CSI Classifier - Test Accuracy: {accuracy:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: confusion_matrix.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names,
                cbar_kws={'label': 'Percentage'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Normalized Confusion Matrix - Accuracy: {accuracy:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: confusion_matrix_normalized.png")
    plt.close()


def save_model_summary(model, output_dir):
    """Save model architecture summary."""
    output_dir = Path(output_dir)

    with open(output_dir / 'model_architecture.txt', 'w') as f:
        f.write("CSI LSTM Classifier Architecture\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(model) + "\n\n")
        f.write("=" * 50 + "\n")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total:,}\n")
        f.write(f"Trainable parameters: {trainable:,}\n")

    print(f"  Saved: model_architecture.txt")
