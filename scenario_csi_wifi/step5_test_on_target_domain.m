%% STEP 5: Test Trained Model on Target Domain
% Evaluates the baseline LSTM (trained on source domain) against
% the target domain to quantify performance degradation.
%
% Inputs: lstm_results/trained_model.mat (from step4)
%         lstm_data_target_domain.mat (from step4)
% Output: lstm_results_target_domain/
clear; clc; close all;

fprintf('========================================\n');
fprintf('  STEP 6: Test on Target Domain          \n');
fprintf('========================================\n\n');

%% Configuration
model_file = 'lstm_results/trained_model.mat';
target_file = 'lstm_data_target_domain.mat';
results_dir = 'lstm_results_target_domain';

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Load trained model
fprintf('Loading trained model from %s...\n', model_file);
if ~exist(model_file, 'file')
    error('Model not found: %s\nRun step4_train_lstm.m first!', model_file);
end

model_data = load(model_file, 'net', 'accuracy', 'aug_params', ...
                  'Prediction', 'Labels_Test');
net = model_data.net;
accuracy_source = model_data.accuracy;
Prediction_Source = model_data.Prediction;
Labels_Test_Source = model_data.Labels_Test;

fprintf('  Model loaded (source accuracy: %.2f%%)\n\n', accuracy_source);

%% Load target domain data
fprintf('Loading target domain data from %s...\n', target_file);
if ~exist(target_file, 'file')
    error('Target data not found: %s\nRun step5_create_target_domain.m first!', target_file);
end

load(target_file);
fprintf('  Loaded %d target domain samples\n\n', length(Labels_Test_Target));

%% Preprocess target domain data (same as training)
fprintf('Preprocessing target domain data...\n');

% Convert to magnitude only
if ~isreal(Data_Test_LSTM_Target{1})
    Data_Test_LSTM_Target = cellfun(@abs, Data_Test_LSTM_Target, 'UniformOutput', false);
end

% Per-sample normalization (CRITICAL: same as training!)
Data_Test_LSTM_Target = normalize_per_sample(Data_Test_LSTM_Target);
fprintf('  Preprocessing complete\n\n');

%% Predict on target domain
fprintf('=== Testing on Target Domain ===\n');
Prediction_Target = classify(net, Data_Test_LSTM_Target, ...
    'MiniBatchSize', 16, 'SequenceLength', 'longest');

Labels_Test_Target_cat = categorical(Labels_Test_Target);
accuracy_target = sum(Prediction_Target == Labels_Test_Target_cat) / ...
                  length(Labels_Test_Target_cat) * 100;

%% Results comparison
activity_names = {'EMPTY', 'SIT', 'STAND', 'WALK'};

fprintf('\n========================================\n');
fprintf('      DOMAIN SHIFT EVALUATION           \n');
fprintf('========================================\n');
fprintf('Source Domain (Train/Test): %.2f%%\n', accuracy_source);
fprintf('Target Domain (Test only):  %.2f%%\n', accuracy_target);
fprintf('Performance Drop:           %.2f%%\n', accuracy_source - accuracy_target);
fprintf('========================================\n\n');

%% Per-class analysis
fprintf('Per-Class Performance:\n');
fprintf('%-12s %10s %10s %12s %10s\n', 'Activity', 'Source', 'Target', 'Correct/Tot', 'Drop');
fprintf('--------------------------------------------------------------\n');

% Source per-class (from source test set predictions)
for i = 1:4
    % Source domain per-class accuracy
    src_idx = find(Labels_Test_Source == i);
    src_pred = Prediction_Source(src_idx);
    src_correct = sum(src_pred == categorical(i));
    src_acc = (src_correct / length(src_idx)) * 100;

    % Target domain per-class accuracy
    tgt_idx = find(Labels_Test_Target == i);
    tgt_pred = Prediction_Target(tgt_idx);
    tgt_correct = sum(tgt_pred == categorical(i));
    tgt_acc = (tgt_correct / length(tgt_idx)) * 100;

    drop = src_acc - tgt_acc;

    fprintf('%-12s %9.2f%% %9.2f%% %10d/%-4d %9.2f%%\n', ...
        activity_names{i}, src_acc, tgt_acc, tgt_correct, length(tgt_idx), drop);
end

fprintf('========================================\n\n');

%% Confusion matrix for target domain
fig_cm_target = figure('Position', [100, 100, 800, 600]);
cm_target = confusionchart(Labels_Test_Target_cat, Prediction_Target);
cm_target.Title = sprintf('Target Domain - Accuracy: %.2f%% (Drop: %.2f%%)', ...
    accuracy_target, accuracy_source - accuracy_target);
cm_target.RowSummary = 'row-normalized';
cm_target.ColumnSummary = 'column-normalized';

saveas(fig_cm_target, fullfile(results_dir, 'confusion_matrix_target_domain.png'));
saveas(fig_cm_target, fullfile(results_dir, 'confusion_matrix_target_domain.fig'));
fprintf('  Saved: confusion_matrix_target_domain.png\n');

%% Comparison plot
fig_comparison = figure('Position', [100, 100, 1000, 500]);

subplot(1, 2, 1);
bar([accuracy_source, accuracy_target]);
set(gca, 'XTickLabel', {'Source Domain', 'Target Domain'});
ylabel('Accuracy (%)');
title('Overall Accuracy Comparison');
ylim([0, 100]);
grid on;
text(1, accuracy_source+2, sprintf('%.1f%%', accuracy_source), 'HorizontalAlignment', 'center');
text(2, accuracy_target+2, sprintf('%.1f%%', accuracy_target), 'HorizontalAlignment', 'center');

subplot(1, 2, 2);
class_acc_source = zeros(1, 4);
class_acc_target_arr = zeros(1, 4);

for i = 1:4
    src_idx = find(Labels_Test_Source == i);
    class_acc_source(i) = sum(Prediction_Source(src_idx) == categorical(i)) / length(src_idx) * 100;

    tgt_idx = find(Labels_Test_Target == i);
    class_acc_target_arr(i) = sum(Prediction_Target(tgt_idx) == categorical(i)) / length(tgt_idx) * 100;
end

bar([class_acc_source; class_acc_target_arr]');
set(gca, 'XTickLabel', activity_names);
ylabel('Accuracy (%)');
title('Per-Class Accuracy: Source vs Target');
legend('Source Domain', 'Target Domain', 'Location', 'best');
ylim([0, 100]);
grid on;

saveas(fig_comparison, fullfile(results_dir, 'source_vs_target_comparison.png'));
saveas(fig_comparison, fullfile(results_dir, 'source_vs_target_comparison.fig'));
fprintf('  Saved: source_vs_target_comparison.png\n');

%% Save results
save(fullfile(results_dir, 'target_domain_results.mat'), ...
    'Prediction_Target', 'accuracy_target', 'accuracy_source', ...
    'Labels_Test_Target', 'target_params');

%% Report
fid = fopen(fullfile(results_dir, 'domain_shift_report.txt'), 'w');
fprintf(fid, 'Domain Shift Evaluation Report\n');
fprintf(fid, '==============================\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));
fprintf(fid, 'SOURCE DOMAIN (Training):\n');
fprintf(fid, '  Accuracy: %.2f%%\n\n', accuracy_source);
fprintf(fid, 'TARGET DOMAIN (Testing):\n');
fprintf(fid, '  Accuracy: %.2f%%\n\n', accuracy_target);
fprintf(fid, 'Performance Drop: %.2f%%\n\n', accuracy_source - accuracy_target);

fprintf(fid, 'Per-Class Results:\n');
for i = 1:4
    tgt_idx = find(Labels_Test_Target == i);
    tgt_pred = Prediction_Target(tgt_idx);
    tgt_correct = sum(tgt_pred == categorical(i));
    tgt_acc = (tgt_correct / length(tgt_idx)) * 100;
    fprintf(fid, '  %-8s: %.2f%% (%d/%d)\n', ...
        activity_names{i}, tgt_acc, tgt_correct, length(tgt_idx));
end
fclose(fid);
fprintf('  Saved: domain_shift_report.txt\n');

%% Interpretation
fprintf('\n=== Interpretation ===\n');
drop = accuracy_source - accuracy_target;
if drop > 10
    fprintf('  SEVERE domain shift (%.1f%% drop)\n', drop);
    fprintf('  This motivates H-inf robust control for domain adaptation!\n');
elseif drop > 5
    fprintf('  MODERATE domain shift (%.1f%% drop)\n', drop);
    fprintf('  Good evidence for H-inf controller benefit.\n');
else
    fprintf('  MILD domain shift (%.1f%% drop)\n', drop);
    fprintf('  Consider increasing target domain difficulty.\n');
end

fprintf('\nResults saved to: %s/\n', results_dir);
fprintf('\nNext step: run step6_train_dualpath.m\n');
