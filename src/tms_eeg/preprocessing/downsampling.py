import mne
from src.tms_eeg.config.settings import ProjectConfig

class Downsampler:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def downsample(self, epochs) -> 'mne.Epochs':
        """Downsample epoched EEG data."""
        resampled = epochs.copy().resample(self.config.epochs.downsample_freq)
        
        return resampled

    def downsample_emg_channels(self, epochs) -> 'mne.Epochs':
        """Extract EMG channels and downsample them at 3000 Hz."""
        # Extract only EMG channels
        emg_epochs = epochs.copy().pick_types(emg=True)
        # Downsample at EMG frequency
        resampled = emg_epochs.resample(self.config.epochs.emg_downsample_freq)
        return resampled
