from src.tms_eeg.config.settings import ProjectConfig

class Downsampler:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def downsample(self, epochs) -> 'mne.Epochs':
        """Downsample epoched EEG data."""
        resampled = epochs.copy().resample(self.config.epochs.downsample_freq)
        
        return resampled