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
