# reader.py
from pathlib import Path
import mne
import os
from src.tms_eeg.config.settings import ProjectConfig

def load_raw(config: ProjectConfig):
    """Load raw EEG data from BDF file.

    Args:
        config (ProjectConfig): Project configuration.

    Returns:
        mne.io.Raw: Raw EEG data.
    """
    base_dir = Path(__file__).parents[3]
    subject_dir = base_dir / "data" / "raw" / f"{config.subject_id}_data"

    file_path = next(subject_dir.glob("*.bdf"))
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    
    print(raw.info)
    print(raw.annotations)
    print(raw.ch_names)

    return raw


# def load_raw_downsampled(
#     config: ProjectConfig, 
#     target_sfreq: float = 500.0,
#     n_jobs: int = -1):
#     """Load and downsample raw EEG data to reduce memory usage.

#     Args:
#         config (ProjectConfig): Project configuration.
#         target_sfreq (float): Target sampling frequency in Hz. Defaults to 500.0.
#         n_jobs (int): Number of CPU cores to use. Defaults to -1 (uses all available cores minus 2).

#     Returns:
#         mne.io.Raw: Downsampl
        
#         ed raw EEG data.
#     """
#     # Calcula os núcleos dinamicamente, permitindo rodar em qualquer PC
#     if n_jobs == -1:
#         n_jobs = max(1, os.cpu_count() - 2)

#     base_dir = Path(__file__).parents[3]
#     subject_dir = base_dir / "data" / "raw" / f"{config.subject_id}_data"
#     file_path = next(subject_dir.glob("*.bdf"))

#     # Load WITHOUT putting data in RAM
#     raw = mne.io.read_raw_bdf(file_path, preload=False)

#     # Resample using multiple CPU cores dynamically
#     raw.resample(target_sfreq, n_jobs=n_jobs)

#     # NOW load into RAM (already downsampled)
#     raw.load_data()

#     return raw
