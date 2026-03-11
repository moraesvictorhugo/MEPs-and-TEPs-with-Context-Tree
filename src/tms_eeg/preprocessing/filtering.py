from src.tms_eeg.config.settings import ProjectConfig

class Filter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def eeg_bp_filter(self, raw) -> 'mne.io.BaseRaw':
        """Apply bandpass filter to raw EEG data."""
        bp = self.config.filters.eeg_bandpass
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1], picks='eeg')
        
        return filtered
        
    def notch_filter(self, raw):
        """Apply notch filter to raw EEG data."""
        notch = self.config.filters.notch
        filtered = raw.copy().notch_filter(freqs=notch)
        
        return filtered
        
    def eeg_bp_filter_epoch(self, raw):
        """Apply bandpass filter to epoched EEG data."""
        bp = self.config.filters.eeg_bandpass_epochs
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1], picks='eeg')
        
        return filtered
    
    def emg_bp_filter(self, raw):
        """Apply bandpass filter to raw EMG data."""
        bp = self.config.filters.emg_bandpass
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1], picks='emg')
        
        return filtered