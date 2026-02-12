# CSI WiFi Activity Recognition with Domain Shift Analysis

LSTM-based human activity recognition using Channel State Information (CSI) from WiFi signals, with L-infinity norm domain shift analysis for robustness evaluation.

## Activities

| Label | Activity |
|-------|----------|
| 0 | EMPTY (no person) |
| 1 | SIT |
| 2 | STAND |
| 3 | WALK |

```bash
# Install dependencies
pip install -r requirements.txt

### Pipeline Scripts (run in order)
| File | Description |
|------|-------------|
MATLAB FILES
| `step1_load_csi_data.m` | Load CSI data from PCAP files |
| `step2_prepare_lstm_data.m` | Stratified split and LSTM formatting |
| `step3_train_lstm.m` | Train and evaluate baseline LSTM |
| `step4_create_target_domain.m` | Create target domain test set |
| `step5_test_on_target_domain.m` | Evaluate under domain shift |
| `step6_train_dualpath.m` | Train dual-path (time+freq) LSTM |
| `step7_test_dualpath_on_target.m` | Evaluate dual-path under shift |
PYTHON FILES
|'load_matlab_csi_data.py'| Step 1: Load MATLAB data → csi_data_python.pkl|
|'create_augmented_dataset.py --difficulty hard'| Step 2: Augment data → csi_data_augmented_pytorch.pkl|
|'prepare_lstm_data_pytorch.py'|Step 3: Prepare train/val/test splits → lstm_data_prepared_pytorch.pkl|
|'train_csi_lstm_pytorch.py'| Step 4: Train LSTM → lstm_results_pytorch/best_model.pth|
|'create_target_domain.py --severity hard --method all'| Step 5: Create target domain with L-inf shift → csi_data_target_domain.pkl|
|'test_on_target_domain.py'| Step 6: Evaluate robustness → domain_shift_results/|


## Domain Shift Methods in create_target_domain.py

All three methods use L-infinity norm constraint: `||d||_inf <= alpha`

| Method | Description | Use Case |
|--------|-------------|----------|
| **Fixed** | `d = ±alpha` (random sign) | Worst-case analysis |
| **Uniform** | `d ~ U[-alpha, +alpha]` | Moderate shift |
| **Mixture** | 70% small (±0.1α) + 30% large (±α) + temporal smoothing | Realistic shift |


## Model Architecture

```
Input (104 CSI features, 150 timesteps)
  → LSTM Layer 1 (128 hidden units, sequence output)
  → Dropout (0.3)
  → LSTM Layer 2 (64 hidden units, last output)
  → Dropout (0.3)
  → Fully Connected (4 classes)
```

## File Descriptions

| File | Purpose |
|------|---------|
| `load_matlab_csi_data.py` | Load CSI data from MATLAB .mat format |
| `csi_augmentation.py` | CSI augmentation functions (SNR noise, phase noise, etc.) |
| `create_augmented_dataset.py` | Apply augmentations to create training data |
| `prepare_lstm_data_pytorch.py` | Stratified split and PyTorch DataLoader creation |
| `lstm_csi_classifier_pytorch.py` | LSTM model definition and training/testing functions |
| `train_csi_lstm_pytorch.py` | Complete training script |
| `csi_domain_shift_analysis.py` | L-inf norm domain shift functions |
| `create_target_domain.py` | Generate target domain with L-inf perturbations |
| `test_on_target_domain.py` | Evaluate trained model under domain shift |

## Data Format

- **CSI shape**: `(n_samples, 150, 104)` — 150 time steps, 104 subcarrier features
- **Normalization**: Per-sample (mean=0, std=1), applied during training and before shift
- **Labels**: Integer 0–3 corresponding to EMPTY, SIT, STAND, WALK

## Requirements

- Python 3.8+
- PyTorch 1.10+
- MATLAB data file: `csi_raw_data_large.mat` (from MATLAB preprocessing)
