# reader.py
from pathlib import Path
import mne
from src.tms_eeg.config.settings import ProjectConfig

def load_data(config: ProjectConfig, data_type: str = "raw"):
    """Load EEG data from file.

    Args:
        config (ProjectConfig): Project configuration.
        data_type (str): "raw" for Raw (.bdf) or "epochs" for Epochs (-epo.fif).

    Returns:
        mne.io.Raw | mne.Epochs: EEG data.
    """
    base_dir = Path(__file__).parents[3]

    if data_type == "raw":
        subject_dir = base_dir / "data" / "raw" / f"{config.subject_id}_data"
        file_path = next(subject_dir.glob("*.bdf"))
        data = mne.io.read_raw_bdf(file_path, preload=True)

    elif data_type == "epochs":
        subject_dir = base_dir / "data" / "processed" / f"{config.subject_id}" / "processed"
        file_path = next(subject_dir.glob("*_epochs_processed.fif"))
        data = mne.read_epochs(file_path, preload=True)

    else:
        raise ValueError(f"data_type inválido: '{data_type}'. Use 'raw' or 'epochs'.")

    print(data.info)
    print(data.ch_names)
    if hasattr(data, "annotations"):
        print(data.annotations)

    return data