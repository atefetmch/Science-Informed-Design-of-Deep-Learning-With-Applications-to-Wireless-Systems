%% STEP 4: Train LSTM and Evaluate
% Trains the baseline LSTM classifier on augmented data.
%
% Input:  lstm_data_prepared.mat (from step2)
% Output: lstm_results/ (model, plots, reports)
clear; clc;

fprintf('========================================\n');
fprintf('  STEP 4: Train LSTM Classifier          \n');
fprintf('========================================\n\n');

%% Configuration
input_file ='lstm_data_prepared.mat';% 
results_dir = 'lstm_results';

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Load prepared data
fprintf('Loading data from %s...\n', input_file);
if ~exist(input_file, 'file')
    error('File not found: %s\nRun step3_create_augmented_dataset.m first!', input_file);
end

load(input_file);
fprintf('  Train: %d, Val: %d, Test: %d\n', ...
    length(Labels_Train), length(Labels_Val), length(Labels_Test));

fprintf('\nData type check:\n');
fprintf('  First sample is complex: %s\n', mat2str(~isreal(Data_Train_LSTM{1})));
fprintf('  Sample data shape: %dx%d\n\n', size(Data_Train_LSTM{1}));

%% Train LSTM
[Prediction, net, accuracy] = LSTM_CSI_Classifier(...
    Data_Train_LSTM, Labels_Train, ...
    Data_Val_LSTM, Labels_Val, ...
    Data_Test_LSTM, Labels_Test, ...
    results_dir);

%% Detailed results
fprintf('\n========================================\n');
fprintf('  DETAILED RESULTS                      \n');
fprintf('========================================\n');
fprintf('Overall Test Accuracy: %.2f%%\n\n', accuracy);

activity_names = {'EMPTY', 'SIT', 'STAND', 'WALK'};

fprintf('Per-Class Performance:\n');
fprintf('%-10s %10s %15s\n', 'Activity', 'Accuracy', 'Correct/Total');
fprintf('----------------------------------------\n');

results_table = table();
for i = 1:4
    class_idx = find(Labels_Test == i);
    class_pred = Prediction(class_idx);
    num_correct = sum(class_pred == categorical(i));
    num_total = length(class_idx);
    class_acc = (num_correct / num_total) * 100;

    fprintf('%-10s %9.2f%% %10d/%d\n', ...
        activity_names{i}, class_acc, num_correct, num_total);

    results_table = [results_table; table({activity_names{i}}, class_acc, ...
        num_correct, num_total, ...
        'VariableNames', {'Activity', 'Accuracy', 'Correct', 'Total'})];
end

%% Save results
fprintf('\n=== Saving Results ===\n');

% Per-class CSV
writetable(results_table, fullfile(results_dir, 'per_class_results.csv'));
fprintf('  Saved: per_class_results.csv\n');

% Predictions CSV
predictions_table = table(Labels_Test, double(Prediction), ...
    'VariableNames', {'TrueLabel', 'PredictedLabel'});
writetable(predictions_table, fullfile(results_dir, 'predictions.csv'));
fprintf('  Saved: predictions.csv\n');

% Model and results (includes aug_params for downstream comparison)
save(fullfile(results_dir, 'trained_model.mat'), ...
    'net', 'Prediction', 'Labels_Test', 'accuracy', ...
    'activity_names');
%save(fullfile(results_dir, 'trained_model.mat'), ...
   % 'net', 'Prediction', 'Labels_Test', 'accuracy', ...
   % 'activity_names', 'aug_params');
fprintf('  Saved: trained_model.mat\n');

%% Summary report
fid = fopen(fullfile(results_dir, 'training_summary.txt'), 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'LSTM CSI Activity Recognition - Results\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Overall Test Accuracy: %.2f%%\n\n', accuracy);
fprintf(fid, 'Dataset Split:\n');
fprintf(fid, '  Training:   %d samples\n', length(Labels_Train));
fprintf(fid, '  Validation: %d samples\n', length(Labels_Val));
fprintf(fid, '  Testing:    %d samples\n\n', length(Labels_Test));
fprintf(fid, 'Per-Class Results:\n');
fprintf(fid, '%-10s %10s %15s\n', 'Activity', 'Accuracy', 'Correct/Total');
fprintf(fid, '----------------------------------------\n');
for i = 1:height(results_table)
    fprintf(fid, '%-10s %9.2f%% %10d/%d\n', ...
        results_table.Activity{i}, results_table.Accuracy(i), ...
        results_table.Correct(i), results_table.Total(i));
end
fclose(fid);
fprintf('  Saved: training_summary.txt\n');

%% Per-class accuracy bar chart
fig_bar = figure('Position', [100, 100, 800, 600]);
bar(results_table.Accuracy);
set(gca, 'XTickLabel', activity_names);
ylabel('Accuracy (%)');
xlabel('Activity Class');
title(sprintf('Per-Class Accuracy (Overall: %.2f%%)', accuracy));
ylim([0, 100]);
grid on;
for i = 1:length(results_table.Accuracy)
    text(i, results_table.Accuracy(i) + 2, ...
        sprintf('%.1f%%', results_table.Accuracy(i)), ...
        'HorizontalAlignment', 'center');
end
saveas(fig_bar, fullfile(results_dir, 'per_class_accuracy.png'));
saveas(fig_bar, fullfile(results_dir, 'per_class_accuracy.fig'));
fprintf('  Saved: per_class_accuracy.png/.fig\n');

%% Summary
fprintf('\n========================================\n');
fprintf('All results saved to: %s/\n', results_dir);
fprintf('========================================\n');
fprintf('Files created:\n');
fprintf('  - training_progress.png/fig\n');
fprintf('  - confusion_matrix.png/fig\n');
fprintf('  - per_class_accuracy.png/fig\n');
fprintf('  - per_class_results.csv\n');
fprintf('  - predictions.csv\n');
fprintf('  - trained_model.mat\n');
fprintf('  - training_summary.txt\n');
fprintf('========================================\n');
fprintf('\nNext step: run step4_create_target_domain.m\n');
