%% STEP 1: Load all CSI data from PCAP files
% Creates: csi_raw_data.mat
%
% Requires: CSIReader() function on the MATLAB path
clear; clc;

fprintf('========================================\n');
fprintf('  STEP 1: Load CSI Data from PCAP Files\n');
fprintf('========================================\n\n');

%% Configuration - CHANGE THESE
main_folder = 'YOUR DATA PATH';  % <-- Set your data path here
activities = {'EMPTY', 'SIT', 'STAND', 'WALK'};
numpackets = 300;

%% Count total files first (dynamic allocation)
total_files = 0;
for act_idx = 1:length(activities)
    activity_folder = fullfile(main_folder, activities{act_idx});
    pcap_files = dir(fullfile(activity_folder, '*.pcap'));
    total_files = total_files + length(pcap_files);
end
fprintf('Found %d total PCAP files across %d activities\n\n', ...
    total_files, length(activities));

%% Pre-allocate with actual count
all_csi_data = cell(total_files, 1);
all_labels = zeros(total_files, 1);
all_filenames = cell(total_files, 1);

sample_idx = 1;

%% Loop through all activities and files
for act_idx = 1:length(activities)
    activity = activities{act_idx};
    activity_folder = fullfile(main_folder, activity);

    pcap_files = dir(fullfile(activity_folder, '*.pcap'));

    fprintf('=== Processing %s: %d files ===\n', activity, length(pcap_files));

    for file_idx = 1:length(pcap_files)
        filepath = fullfile(activity_folder, pcap_files(file_idx).name);

        % Read CSI - Keep as raw matrix (no PCA, no feature extraction)
        [csi_buff, NFFT] = CSIReader(filepath, numpackets);

        all_csi_data{sample_idx} = csi_buff;
        all_labels(sample_idx) = act_idx;
        all_filenames{sample_idx} = pcap_files(file_idx).name;

        fprintf('  Sample %d: %s -> size %dx%d\n', sample_idx, ...
            pcap_files(file_idx).name, size(csi_buff));

        sample_idx = sample_idx + 1;
    end
end

%% Trim if fewer files than expected
actual_count = sample_idx - 1;
all_csi_data = all_csi_data(1:actual_count);
all_labels = all_labels(1:actual_count);
all_filenames = all_filenames(1:actual_count);

%% Save
fprintf('\n=== Saving Data ===\n');
fprintf('Total samples: %d\n', actual_count);
save('csi_raw_data.mat', 'all_csi_data', 'all_labels', 'all_filenames', ...
     'activities', '-v7.3');
fprintf('  Saved: csi_raw_data.mat\n');

%% Verify
fprintf('\n=== Verification ===\n');
fprintf('First sample size: %dx%d\n', size(all_csi_data{1}));
fprintf('Class distribution:\n');
for i = 1:length(activities)
    fprintf('  %s: %d samples\n', activities{i}, sum(all_labels == i));
end

fprintf('\nNext step: run step2_prepare_lstm_data.m\n');
