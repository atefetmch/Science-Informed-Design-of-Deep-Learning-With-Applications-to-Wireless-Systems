%% STEP 6: Train Dual-Path LSTM (Time + Frequency)
% Combines time-domain and frequency-domain features for improved
% activity recognition.
%
% Input:  lstm_data_prepared.mat (from step3)
% Output: lstm_results_dualpath/
clear; clc; close all;

fprintf('========================================\n');
fprintf('  STEP 7: Dual-Path LSTM Training        \n');
fprintf('  (Time Domain + Frequency Domain)       \n');
fprintf('========================================\n\n');

%% Configuration
input_file = 'lstm_data_prepared.mat';
results_dir = 'lstm_results_dualpath';

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Load prepared dataset
fprintf('Step 1: Loading prepared dataset from %s...\n', input_file);
if ~exist(input_file, 'file')
    error('File not found: %s\nRun step2_create_dataset.m first!', input_file);
end

load(input_file);
fprintf('  Train: %d, Val: %d, Test: %d\n\n', ...
    length(Labels_Train), length(Labels_Val), length(Labels_Test));

%% Prepare FFT features
fprintf('Step 2: Computing FFT features...\n');

% CRITICAL: Each split uses its OWN data for FFT computation.
% Do NOT compute all FFT from training data (that would be data leakage).
[Data_Train_FFT, ~] = prepare_fft_features_robust(Data_Train_LSTM);
[Data_Val_FFT, ~]   = prepare_fft_features_robust(Data_Val_LSTM);
[Data_Test_FFT, ~]  = prepare_fft_features_robust(Data_Test_LSTM);

fprintf('  FFT features ready\n\n');

%% Train Dual-Path LSTM
fprintf('Step 3: Training Dual-Path LSTM...\n');
fprintf('This combines time + frequency representations.\n');
fprintf('Training time: 5-15 minutes...\n\n');

tic;
[Prediction, net, accuracy] = LSTM_CSI_Classifier_DualPath(...
    Data_Train_LSTM, Data_Train_FFT, Labels_Train, ...
    Data_Val_LSTM, Data_Val_FFT, Labels_Val, ...
    Data_Test_LSTM, Data_Test_FFT, Labels_Test, ...
    results_dir);
training_time = toc;

%% Results
fprintf('\n========================================\n');
fprintf('  RESULTS - SOURCE DOMAIN (DUAL-PATH)   \n');
fprintf('========================================\n');
fprintf('Test Accuracy: %.2f%%\n', accuracy);
fprintf('Training Time: %.1f minutes\n\n', training_time/60);

activity_names = {'EMPTY', 'SIT', 'STAND', 'WALK'};
fprintf('Per-Class Performance:\n');
fprintf('%-12s %10s %15s\n', 'Activity', 'Accuracy', 'Correct/Total');
fprintf('----------------------------------------\n');

for i = 1:4
    class_idx = find(Labels_Test == i);
    class_pred = Prediction(class_idx);
    num_correct = sum(class_pred == categorical(i));
    class_acc = (num_correct / length(class_idx)) * 100;
    fprintf('%-12s %9.2f%% %12d/%d\n', ...
        activity_names{i}, class_acc, num_correct, length(class_idx));
end

fprintf('========================================\n\n');

%% Save model
save(fullfile(results_dir, 'trained_model_dualpath.mat'), ...
    'net', 'accuracy', 'Prediction', 'Labels_Test', 'training_time');
fprintf('  Model saved to: %s/\n\n', results_dir);

fprintf('Next step: run step7_test_dualpath_on_target.m\n');
