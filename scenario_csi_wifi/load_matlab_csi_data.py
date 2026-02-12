"""
Load CSI Data from MATLAB .mat File


Loads the csi_raw_data.mat file created by step1_load_csi_data.m,
converts MATLAB cell arrays to Python lists, and saves as pickle
for fast future loading.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict


def _is_v73_mat(mat_file: str) -> bool:
    """Check if a .mat file is v7.3 (HDF5) format by trying h5py."""
    try:
        import h5py
        with h5py.File(mat_file, 'r') as f:
            pass  # If it opens, it's HDF5
        return True
    except Exception:
        return False


def _load_v73(mat_file: str) -> Dict:
    """
    Load v7.3 .mat file using h5py.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to read MATLAB v7.3 files.\n"
            "Install it with: pip install h5py"
        )

    print("  Detected MATLAB v7.3 (HDF5) format, using h5py...")

    with h5py.File(mat_file, 'r') as f:
        # --- Labels ---
        all_labels = np.array(f['all_labels']).flatten()

        # --- CSI data (cell array of matrices) ---
        print("  Reading CSI data cell array...")
        csi_refs = f['all_csi_data']

        csi_data = []
        # Cell arrays in HDF5: each element is an object reference
        # Shape is typically (N, 1) for a column cell array
        if csi_refs.ndim == 2:
            n_samples = csi_refs.shape[1]  # HDF5 stores transposed
            for i in range(n_samples):
                ref = csi_refs[0, i]
                matrix = np.array(f[ref])
                # HDF5 stores complex as struct with 'real' and 'imag',
                # or as a compound type, or already as complex
                if matrix.dtype.names and 'real' in matrix.dtype.names:
                    matrix = matrix['real'] + 1j * matrix['imag']
                # HDF5 stores MATLAB matrices transposed
                matrix = matrix.T
                csi_data.append(matrix)
        elif csi_refs.ndim == 1:
            n_samples = csi_refs.shape[0]
            for i in range(n_samples):
                ref = csi_refs[i]
                matrix = np.array(f[ref])
                if matrix.dtype.names and 'real' in matrix.dtype.names:
                    matrix = matrix['real'] + 1j * matrix['imag']
                matrix = matrix.T
                csi_data.append(matrix)
        else:
            raise ValueError(f"Unexpected csi_data shape: {csi_refs.shape}")

        # --- Filenames (cell array of strings) ---
        filenames = []
        if 'all_filenames' in f:
            fn_refs = f['all_filenames']
            try:
                refs_flat = np.array(fn_refs).flatten()
                for ref in refs_flat:
                    obj = f[ref]
                    raw = np.array(obj).flatten()
                    if raw.dtype.kind in ('U', 'S'):
                        filename = str(raw[0]) if len(raw) == 1 else ''.join(str(c) for c in raw)
                    elif raw.dtype.kind == 'O':
                        filename = str(raw[0])
                    else:
                        filename = ''.join(chr(int(c)) for c in raw)
                    filenames.append(filename.strip())
            except Exception as e:
                print(f"  Warning: Could not read filenames ({e})")
                filenames = [f'sample_{i}' for i in range(n_samples)]
        else:
            filenames = [f'sample_{i}' for i in range(n_samples)]

        # --- Activities (cell array of strings) ---
        activities = []
        if 'activities' in f:
            act_refs = f['activities']
            try:
                refs_flat = np.array(act_refs).flatten()
                for ref in refs_flat:
                    obj = f[ref]
                    raw = np.array(obj).flatten()
                    # Try decoding: could be uint16 char codes, bytes, or string
                    if raw.dtype.kind in ('U', 'S'):  # already string
                        activity = str(raw[0]) if len(raw) == 1 else ''.join(str(c) for c in raw)
                    elif raw.dtype.kind == 'O':  # object array
                        activity = str(raw[0])
                    else:  # numeric (uint16 char codes)
                        activity = ''.join(chr(int(c)) for c in raw)
                    activities.append(activity.strip())
            except Exception as e:
                print(f"  Warning: Could not read activities ({e})")
                activities = ['EMPTY', 'SIT', 'STAND', 'WALK']
        else:
            activities = ['EMPTY', 'SIT', 'STAND', 'WALK']

        print(f"  Activities found: {activities}")

    return all_labels, csi_data, filenames, activities


def _load_legacy(mat_file: str) -> Dict:
    """Load v5/v7 .mat file using scipy.io."""
    import scipy.io as sio

    print("  Detected legacy MATLAB format, using scipy.io...")

    mat_data = sio.loadmat(mat_file)

    all_labels = mat_data['all_labels'].flatten()

    all_csi_data_matlab = mat_data['all_csi_data']
    csi_data = []
    for i in range(len(all_csi_data_matlab)):
        csi_data.append(all_csi_data_matlab[i, 0])

    filenames = []
    if 'all_filenames' in mat_data:
        all_filenames_matlab = mat_data['all_filenames']
        for i in range(len(all_filenames_matlab)):
            filenames.append(all_filenames_matlab[i, 0][0])
    else:
        filenames = [f'sample_{i}' for i in range(len(all_labels))]

    activities = []
    if 'activities' in mat_data:
        activities_matlab = mat_data['activities']
        for i in range(len(activities_matlab[0])):
            activities.append(activities_matlab[0, i][0])
    else:
        activities = ['EMPTY', 'SIT', 'STAND', 'WALK']

    return all_labels, csi_data, filenames, activities


def load_csi_from_matlab(mat_file: str = 'csi_raw_data.mat') -> Dict:
    """
    Load CSI data from MATLAB .mat file (auto-detects format).

    Args:
        mat_file: Path to the .mat file created by MATLAB.

    Returns:
        data_dict: Dictionary containing:
            - 'csi_data': List of numpy arrays (each: 150x104)
            - 'labels': numpy array of labels (0-3)
            - 'filenames': List of filenames
            - 'activities': List of activity names
            - 'num_samples': Total number of samples
    """
    print(f"Loading CSI data from {mat_file}...")

    if _is_v73_mat(mat_file):
        all_labels, csi_data, filenames, activities = _load_v73(mat_file)
    else:
        all_labels, csi_data, filenames, activities = _load_legacy(mat_file)

    # MATLAB labels are 1-indexed, convert to 0-indexed
    labels = (all_labels - 1).astype(np.int64)

    print(f"  Loaded {len(labels)} samples")
    print("  Conversion complete")

    data_dict = {
        'csi_data': csi_data,
        'labels': labels,
        'filenames': filenames,
        'activities': activities,
        'num_samples': len(labels),
    }

    return data_dict


def verify_data(data_dict: Dict):
    """Verify the loaded CSI data."""
    print("\n" + "=" * 70)
    print("DATA VERIFICATION")
    print("=" * 70)

    num_samples = data_dict['num_samples']
    print(f"\nTotal samples: {num_samples}")
    print(f"Activities: {data_dict['activities']}")

    first_sample = data_dict['csi_data'][0]
    print(f"\nFirst sample:")
    print(f"  Shape: {first_sample.shape}")
    print(f"  Data type: {first_sample.dtype}")
    print(f"  Is complex: {np.iscomplexobj(first_sample)}")
    print(f"  Min value: {np.min(np.abs(first_sample)):.4f}")
    print(f"  Max value: {np.max(np.abs(first_sample)):.4f}")
    print(f"  Mean value: {np.mean(np.abs(first_sample)):.4f}")

    # Class distribution
    print("\nClass Distribution:")
    print("-" * 50)
    labels = data_dict['labels']
    activities = data_dict['activities']

    for i, activity in enumerate(activities):
        count = np.sum(labels == i)
        percentage = 100 * count / num_samples
        print(f"  {activity:8s} (label {i}): {count:4d} samples ({percentage:5.1f}%)")

    # Dimension check
    print("\nDimension Check:")
    print("-" * 50)
    shapes = set()
    for sample in data_dict['csi_data']:
        shapes.add(sample.shape)

    if len(shapes) == 1:
        shape = shapes.pop()
        print(f"  All {num_samples} samples have shape {shape}")
    else:
        print(f"  WARNING: Found {len(shapes)} different shapes:")
        for shape in sorted(shapes):
            count = sum(1 for s in data_dict['csi_data'] if s.shape == shape)
            print(f"    {shape}: {count} samples")

    # Memory estimate
    sample_size_bytes = first_sample.nbytes
    total_memory_mb = (num_samples * sample_size_bytes) / (1024 ** 2)
    print(f"\nMemory Usage:")
    print(f"  Per sample: ~{sample_size_bytes / 1024:.2f} KB")
    print(f"  Total: ~{total_memory_mb:.2f} MB")

    print("=" * 70 + "\n")


def save_to_pickle(data_dict: Dict, output_file: str = 'csi_data_python.pkl'):
    """Save loaded data to pickle for fast future loading."""
    print(f"Saving to pickle: {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = Path(output_file).stat().st_size / (1024 ** 2)
    print(f"  Saved: {output_file} ({file_size_mb:.2f} MB)")
    print(f"\nNext time, load instantly with:")
    print(f"  data = load_from_pickle('{output_file}')")


def load_from_pickle(pickle_file: str = 'csi_data_python.pkl') -> Dict:
    """Quick load from previously saved pickle file."""
    print(f"Loading from pickle: {pickle_file}...")

    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"  Loaded {data_dict['num_samples']} samples")
    return data_dict


def get_sample_by_index(data_dict: Dict, index: int):
    """Get a specific sample by index."""
    csi_matrix = data_dict['csi_data'][index]
    label = data_dict['labels'][index]
    activity = data_dict['activities'][label]
    filename = data_dict['filenames'][index]

    print(f"\nSample {index}:")
    print(f"  Filename: {filename}")
    print(f"  Activity: {activity} (label={label})")
    print(f"  Shape: {csi_matrix.shape}")
    print(f"  Type: {csi_matrix.dtype}")

    return csi_matrix, label, activity, filename


def get_samples_by_activity(data_dict: Dict, activity: str) -> Dict:
    """Get all samples for a specific activity."""
    activity_idx = data_dict['activities'].index(activity)
    mask = data_dict['labels'] == activity_idx

    activity_data = {
        'csi_data': [data_dict['csi_data'][i] for i in range(len(mask)) if mask[i]],
        'labels': data_dict['labels'][mask],
        'filenames': [data_dict['filenames'][i] for i in range(len(mask)) if mask[i]],
        'activity': activity,
        'num_samples': np.sum(mask),
    }

    print(f"Extracted {activity_data['num_samples']} samples for '{activity}'")
    return activity_data


def main():
    """Main execution."""
    print("=" * 70)
    print("CSI DATA LOADER - From MATLAB .mat File")
    print("=" * 70 + "\n")

    mat_file = 'csi_raw_data.mat'
    data_dict = load_csi_from_matlab(mat_file)
    verify_data(data_dict)
    save_to_pickle(data_dict, 'csi_data_python.pkl')

    print("\nExample: Accessing sample data")
    print("-" * 70)
    get_sample_by_index(data_dict, 0)

    print("\n" + "-" * 70)
    # Use second activity from loaded list (not hardcoded)
    if len(data_dict['activities']) > 1:
        get_samples_by_activity(data_dict, data_dict['activities'][1])
    else:
        get_samples_by_activity(data_dict, data_dict['activities'][0])

    print("\n" + "=" * 70)
    print("COMPLETE - Data is ready for PyTorch LSTM training!")
    print("=" * 70 + "\n")

    return data_dict


if __name__ == '__main__':
    data = main()