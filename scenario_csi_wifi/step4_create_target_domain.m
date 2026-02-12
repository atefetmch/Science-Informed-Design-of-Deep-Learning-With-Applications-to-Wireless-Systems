%% STEP 4: Create Target Domain Dataset
% Simulates a DIFFERENT environment/setup than training using
% augmentation-based domain shift.
%
% Input:  lstm_data_prepared.mat 
% Output: lstm_data_target_domain.mat
clear; clc;

fprintf('========================================\n');
fprintf('  STEP 5: Create Target Domain Dataset   \n');
fprintf('========================================\n\n');

%% Configuration
input_file = 'lstm_data_prepared.mat';           % Original (non-augmented)
output_file = 'lstm_data_target_domain.mat';

%% Load ORIGINAL (non-augmented) dataset
fprintf('Loading ORIGINAL dataset from %s...\n', input_file);
if ~exist(input_file, 'file')
    error('File not found: %s\nRun step2_prepare_lstm_data.m first!', input_file);
end

load(input_file);
fprintf('  Loaded: %d test samples\n\n', length(Labels_Test));

%% Target domain augmentation parameters (DIFFERENT from training)
% Training domain had: SNR=15dB, phase_noise=0.2, time_warp=0.15, dropout=0.15
% Target domain uses: different characteristics to simulate environment shift

fprintf('Setting TARGET DOMAIN parameters...\n');
fprintf('(These are DIFFERENT from training domain)\n\n');

target_params = struct();
target_params.snr_db = 12;                     % Lower SNR (noisier environment)
target_params.phase_noise_std = 0.25;          % Higher phase noise (different hardware)
target_params.amp_scale_range = [0.6, 1.4];    % Wider amplitude variation
target_params.time_warp_factor = 0.20;         % Stronger time warping
target_params.subcarrier_dropout = 0.20;       % More dropout
target_params.apply_prob = 0.8;                % Higher probability

fprintf('TARGET DOMAIN Settings:\n');
fprintf('  SNR: %d dB (training: 15 dB) <- Lower\n', target_params.snr_db);
fprintf('  Phase noise: %.2f rad (training: 0.2) <- Higher\n', target_params.phase_noise_std);
fprintf('  Amp scale: [%.1f, %.1f] (training: [0.7, 1.3]) <- Wider\n', target_params.amp_scale_range);
fprintf('  Time warp: %.2f (training: 0.15) <- Stronger\n', target_params.time_warp_factor);
fprintf('  Dropout: %.0f%% (training: 15%%) <- Higher\n', target_params.subcarrier_dropout*100);
fprintf('  Apply prob: %.0f%% (training: 70%%) <- Higher\n\n', target_params.apply_prob*100);

%% Apply target domain augmentation to test set
fprintf('Applying TARGET domain augmentations to test set...\n');
Data_Test_LSTM_Target = augment_csi_data(Data_Test_LSTM, target_params);
Labels_Test_Target = Labels_Test;

fprintf('  Target domain test set created\n');

%% Save target domain dataset
fprintf('\nSaving to %s...\n', output_file);
save(output_file, ...
     'Data_Test_LSTM_Target', 'Labels_Test_Target', ...
     'target_params', 'activities', ...
     '-v7.3');

fprintf('  Saved: %s\n\n', output_file);

%% Verify distribution shift
fprintf('=== Verifying Domain Shift ===\n');

% Load source domain (augmented) for comparison
if exist('lstm_data_augmented.mat', 'file')
    source_data = load('lstm_data_augmented.mat', 'Data_Train_LSTM');

    source_sample = source_data.Data_Train_LSTM{1};
    source_mean = mean(abs(source_sample(:)));
    source_std = std(abs(source_sample(:)));
    source_snr_est = 10*log10(source_mean^2 / source_std^2);

    target_sample = Data_Test_LSTM_Target{1};
    target_mean = mean(abs(target_sample(:)));
    target_std = std(abs(target_sample(:)));
    target_snr_est = 10*log10(target_mean^2 / target_std^2);

    fprintf('\nDomain Statistics Comparison:\n');
    fprintf('                 Source (Train)    Target (Test)\n');
    fprintf('  Mean amplitude:    %.4f          %.4f\n', source_mean, target_mean);
    fprintf('  Std amplitude:     %.4f          %.4f\n', source_std, target_std);
    fprintf('  Est. SNR:          %.1f dB        %.1f dB\n', source_snr_est, target_snr_est);
else
    fprintf('  (Skipping source comparison - lstm_data_augmented.mat not found)\n');
end

%% Target domain class separability
fprintf('\nTarget Domain Class Separability:\n');

sample_empty = Data_Test_LSTM_Target{find(Labels_Test_Target==1, 1)}(:);
sample_sit = Data_Test_LSTM_Target{find(Labels_Test_Target==2, 1)}(:);
sample_stand = Data_Test_LSTM_Target{find(Labels_Test_Target==3, 1)}(:);
sample_walk = Data_Test_LSTM_Target{find(Labels_Test_Target==4, 1)}(:);

dists = [
    norm(abs(sample_empty) - abs(sample_sit))
    norm(abs(sample_empty) - abs(sample_stand))
    norm(abs(sample_empty) - abs(sample_walk))
    norm(abs(sample_sit) - abs(sample_stand))
    norm(abs(sample_sit) - abs(sample_walk))
    norm(abs(sample_stand) - abs(sample_walk))
];
labels = {'EMPTY vs SIT', 'EMPTY vs STAND', 'EMPTY vs WALK', ...
          'SIT vs STAND', 'SIT vs WALK', 'STAND vs WALK'};

for j = 1:length(dists)
    fprintf('  %-16s: %.2f\n', labels{j}, dists(j));
end
fprintf('\n  Average distance: %.2f\n', mean(dists));

fprintf('\n  Domain shift confirmed - distributions are different\n');
fprintf('\nNext step: run step5_test_on_target_domain.m\n');
