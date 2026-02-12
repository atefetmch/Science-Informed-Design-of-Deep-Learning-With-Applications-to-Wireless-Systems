%% STEP 7: Test Dual-Path LSTM on Target Domain
% Evaluates the dual-path LSTM against the target domain and compares
% with the baseline LSTM.
% Output: lstm_results_dualpath_target/
clear; clc; close all;

fprintf('========================================\n');
fprintf('  STEP 8: Dual-Path - Target Domain Test \n');
fprintf('========================================\n\n');

%% Configuration
dualpath_model_file = 'lstm_results_dualpath/trained_model_dualpath.mat';
target_data_file = 'lstm_data_target_domain.mat';
baseline_results_file = 'lstm_results_target_domain/target_domain_results.mat';
results_dir = 'lstm_results_dualpath_target';

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Load trained dual-path model
fprintf('Loading dual-path model from %s...\n', dualpath_model_file);
if ~exist(dualpath_model_file, 'file')
    error('Model not found: %s\nRun step7_train_dualpath.m first!', dualpath_model_file);
end

dualpath_data = load(dualpath_model_file, 'net', 'accuracy', 'Labels_Test');
net = dualpath_data.net;
dualpath_source = dualpath_data.accuracy;
fprintf('  Model loaded (source accuracy: %.2f%%)\n\n', dualpath_source);

%% Load target domain data
fprintf('Loading target domain data from %s...\n', target_data_file);
if ~exist(target_data_file, 'file')
    error('Target data not found: %s\nRun step5_create_target_domain.m first!', target_data_file);
end

load(target_data_file);
fprintf('  Loaded %d target samples\n\n', length(Labels_Test_Target));

%% Prepare FFT features for target domain
fprintf('Computing FFT features for target domain...\n');
[Data_Test_Target_FFT, ~] = prepare_fft_features_robust(Data_Test_LSTM_Target);
fprintf('  FFT features ready\n\n');

%% Preprocess
fprintf('Preprocessing...\n');

% Convert to magnitude
if ~isreal(Data_Test_LSTM_Target{1})
    Data_Test_LSTM_Target = cellfun(@abs, Data_Test_LSTM_Target, 'UniformOutput', false);
    Data_Test_Target_FFT = cellfun(@abs, Data_Test_Target_FFT, 'UniformOutput', false);
end

% Normalize (uses shared normalize_per_sample.m)
Data_Test_LSTM_Target = normalize_per_sample(Data_Test_LSTM_Target);
Data_Test_Target_FFT = normalize_per_sample(Data_Test_Target_FFT);

% Concatenate features (uses shared concatenate_features.m)
fprintf('  Concatenating time + frequency features...\n');
Data_Test_Combined = concatenate_features(Data_Test_LSTM_Target, Data_Test_Target_FFT);

fprintf('  Time features:      [%d x %d]\n', size(Data_Test_LSTM_Target{1}));
fprintf('  Frequency features: [%d x %d]\n', size(Data_Test_Target_FFT{1}));
fprintf('  Combined features:  [%d x %d]\n', size(Data_Test_Combined{1}));
fprintf('  Preprocessing complete\n\n');

%% Predict on target domain
fprintf('=== Testing on Target Domain ===\n');
Prediction_DualPath_Target = classify(net, Data_Test_Combined, ...
    'MiniBatchSize', 16, 'SequenceLength', 'longest');

Labels_Test_Target_cat = categorical(Labels_Test_Target);
dualpath_target = sum(Prediction_DualPath_Target == Labels_Test_Target_cat) / ...
                  length(Labels_Test_Target_cat) * 100;

%% Load baseline results for comparison
fprintf('\nLoading baseline results for comparison...\n');

baseline_source = NaN;
baseline_target = NaN;
Prediction_Baseline = [];

if exist(baseline_results_file, 'file')
    bl = load(baseline_results_file, 'accuracy_target', 'accuracy_source', ...
              'Prediction_Target');
    baseline_source = bl.accuracy_source;
    baseline_target = bl.accuracy_target;
    Prediction_Baseline = bl.Prediction_Target;
    fprintf('  Baseline results loaded\n');
else
    fprintf('  Warning: %s not found - showing dual-path only\n', baseline_results_file);
end

%% Results comparison
activity_names = {'EMPTY', 'SIT', 'STAND', 'WALK'};

fprintf('\n========================================\n');
fprintf('  COMPARISON: Baseline vs Dual-Path     \n');
fprintf('========================================\n\n');

if ~isnan(baseline_source)
    fprintf('SOURCE DOMAIN (Augmented Test Set):\n');
    fprintf('  Baseline LSTM:    %.2f%%\n', baseline_source);
    fprintf('  Dual-Path LSTM:   %.2f%%\n', dualpath_source);
    fprintf('  Difference:       %+.2f%%\n\n', dualpath_source - baseline_source);

    fprintf('TARGET DOMAIN:\n');
    fprintf('  Baseline LSTM:    %.2f%%\n', baseline_target);
    fprintf('  Dual-Path LSTM:   %.2f%%\n', dualpath_target);
    fprintf('  Improvement:      %+.2f%%\n\n', dualpath_target - baseline_target);

    fprintf('DOMAIN SHIFT ROBUSTNESS:\n');
    fprintf('  Baseline drop:    %.2f%%\n', baseline_source - baseline_target);
    fprintf('  Dual-Path drop:   %.2f%%\n', dualpath_source - dualpath_target);
    fprintf('  Robustness gain:  %.2f%%\n', ...
        (baseline_source - baseline_target) - (dualpath_source - dualpath_target));
else
    fprintf('TARGET DOMAIN ONLY:\n');
    fprintf('  Dual-Path LSTM:   %.2f%%\n', dualpath_target);
end

fprintf('========================================\n\n');

%% Per-class analysis
fprintf('Per-Class Performance (Target Domain):\n');

if ~isnan(baseline_target) && ~isempty(Prediction_Baseline)
    fprintf('%-12s %12s %12s %12s\n', 'Activity', 'Baseline', 'Dual-Path', 'Improvement');
    fprintf('----------------------------------------------------------\n');

    for i = 1:4
        class_idx = find(Labels_Test_Target == i);

        % Baseline
        pred_base = Prediction_Baseline(class_idx);
        acc_base = sum(pred_base == categorical(i)) / length(class_idx) * 100;

        % Dual-path
        pred_dual = Prediction_DualPath_Target(class_idx);
        acc_dual = sum(pred_dual == categorical(i)) / length(class_idx) * 100;

        improvement = acc_dual - acc_base;

        fprintf('%-12s %11.2f%% %11.2f%% %+11.2f%%\n', ...
            activity_names{i}, acc_base, acc_dual, improvement);
    end
else
    fprintf('%-12s %12s %15s\n', 'Activity', 'Accuracy', 'Correct/Total');
    fprintf('----------------------------------------\n');

    for i = 1:4
        class_idx = find(Labels_Test_Target == i);
        pred = Prediction_DualPath_Target(class_idx);
        num_correct = sum(pred == categorical(i));
        class_acc = (num_correct / length(class_idx)) * 100;
        fprintf('%-12s %11.2f%% %12d/%d\n', ...
            activity_names{i}, class_acc, num_correct, length(class_idx));
    end
end

fprintf('========================================\n\n');

%% Confusion matrix
fig_cm = figure('Position', [100, 100, 800, 600]);
cm = confusionchart(Labels_Test_Target_cat, Prediction_DualPath_Target);
if ~isnan(baseline_target)
    cm.Title = sprintf('Dual-Path Target: %.2f%% (Baseline: %.2f%%, %+.2f%%)', ...
        dualpath_target, baseline_target, dualpath_target - baseline_target);
else
    cm.Title = sprintf('Dual-Path Target Domain: %.2f%%', dualpath_target);
end
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

saveas(fig_cm, fullfile(results_dir, 'confusion_matrix_dualpath_target.png'));
saveas(fig_cm, fullfile(results_dir, 'confusion_matrix_dualpath_target.fig'));
fprintf('  Saved: confusion_matrix_dualpath_target.png\n');

%% Save results
save(fullfile(results_dir, 'dualpath_target_results.mat'), ...
    'Prediction_DualPath_Target', 'Labels_Test_Target', ...
    'dualpath_target', 'dualpath_source', ...
    'baseline_target', 'baseline_source');

fprintf('  Results saved to: %s/\n\n', results_dir);
fprintf('=== Pipeline Complete ===\n');
