%% STEP 2: Split data into Train/Validation/Test for LSTM
% Loads CSI data, creates stratified split, transposes for LSTM format.
%
% Input:  csi_raw_data.mat (from step1)
% Output: lstm_data_prepared.mat
clear; clc;

fprintf('========================================\n');
fprintf('  STEP 2: Prepare LSTM Data             \n');
fprintf('========================================\n\n');

%% Configuration
input_file = 'csi_raw_data.mat';       % Change to match your step1 output
output_file = 'lstm_data_prepared.mat';

train_ratio = 0.70;
val_ratio = 0.15;
test_ratio = 0.15;

rng(42);  % Reproducibility

%% Load CSI data from Step 1
fprintf('Loading data from %s...\n', input_file);
if ~exist(input_file, 'file')
    error('File not found: %s\nRun step1_load_csi_data.m first!', input_file);
end

load(input_file);
fprintf('  Loaded %d samples\n\n', length(all_csi_data));

%% Stratified split (equal proportion per class)
fprintf('Creating stratified split (%.0f/%.0f/%.0f)...\n', ...
    train_ratio*100, val_ratio*100, test_ratio*100);

num_classes = length(activities);
train_indices = [];
val_indices = [];
test_indices = [];

for class_id = 1:num_classes
    class_samples = find(all_labels == class_id);
    num_samples = length(class_samples);

    fprintf('  %s: %d samples\n', activities{class_id}, num_samples);

    % Shuffle
    class_samples = class_samples(randperm(num_samples));

    % Split points
    num_train = round(train_ratio * num_samples);
    num_val = round(val_ratio * num_samples);

    train_indices = [train_indices; class_samples(1:num_train)];
    val_indices = [val_indices; class_samples(num_train+1:num_train+num_val)];
    test_indices = [test_indices; class_samples(num_train+num_val+1:end)];
end

%% Extract datasets
Data_Train = all_csi_data(train_indices);
Labels_Train = all_labels(train_indices);

Data_Val = all_csi_data(val_indices);
Labels_Val = all_labels(val_indices);

Data_Test = all_csi_data(test_indices);
Labels_Test = all_labels(test_indices);

%% Verify no overlap
assert(isempty(intersect(train_indices, test_indices)), 'Train/test overlap!');
assert(isempty(intersect(train_indices, val_indices)), 'Train/val overlap!');
assert(isempty(intersect(val_indices, test_indices)), 'Val/test overlap!');
fprintf('\n  No overlap between splits\n');

%% Display split info
fprintf('\n=== Dataset Split ===\n');
fprintf('Training:   %d samples (%.1f%%)\n', length(Labels_Train), ...
    100*length(Labels_Train)/length(all_labels));
fprintf('Validation: %d samples (%.1f%%)\n', length(Labels_Val), ...
    100*length(Labels_Val)/length(all_labels));
fprintf('Testing:    %d samples (%.1f%%)\n', length(Labels_Test), ...
    100*length(Labels_Test)/length(all_labels));

fprintf('\n=== Class Distribution ===\n');
fprintf('           ');
for i = 1:num_classes, fprintf('%-8s', activities{i}); end
fprintf('\nTrain:     ');
for i = 1:num_classes, fprintf('%-8d', sum(Labels_Train == i)); end
fprintf('\nVal:       ');
for i = 1:num_classes, fprintf('%-8d', sum(Labels_Val == i)); end
fprintf('\nTest:      ');
for i = 1:num_classes, fprintf('%-8d', sum(Labels_Test == i)); end
fprintf('\n');

%% Transpose for LSTM
% LSTM expects: [numFeatures x numTimeSteps] (subcarriers x time)
% Raw data is:  [numTimeSteps x numFeatures] (time x subcarriers)
fprintf('\n=== Transposing for LSTM ===\n');
fprintf('Original format: %dx%d (time x subcarriers)\n', size(Data_Train{1}));

Data_Train_LSTM = cellfun(@transpose, Data_Train, 'UniformOutput', false);
Data_Val_LSTM = cellfun(@transpose, Data_Val, 'UniformOutput', false);
Data_Test_LSTM = cellfun(@transpose, Data_Test, 'UniformOutput', false);

fprintf('LSTM format:     %dx%d (subcarriers x time)\n', size(Data_Train_LSTM{1}));

%% Save prepared data
fprintf('\nSaving to %s...\n', output_file);
save(output_file, ...
    'Data_Train_LSTM', 'Labels_Train', ...
    'Data_Val_LSTM', 'Labels_Val', ...
    'Data_Test_LSTM', 'Labels_Test', ...
    'activities', '-v7.3');

fprintf('  Saved: %s\n', output_file);
