from src.tms_eeg.config.settings import ProjectConfig

class EEGFilter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def bp_filter(self, raw) -> 'mne.io.BaseRaw':
        """Apply bandpass filter to raw EEG data."""
        bp = self.config.filters.bandpass
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1])
        
        return filtered   # type: ignore
        
    def notch_filter(self, raw):
        """Apply notch filter to raw EEG data."""
        notch = self.config.filters.notch
        filtered = raw.copy().notch_filter(freqs=notch)
        
        return filtered
        
    def bp_filter_epoch(self, raw):
        """Apply bandpass filter to epoched EEG data."""
        bp = self.config.filters.bandpass_epochs
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1])
        
        return filtered