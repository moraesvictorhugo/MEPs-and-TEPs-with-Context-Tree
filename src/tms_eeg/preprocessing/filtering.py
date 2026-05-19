import numpy as np
import mne
from scipy.signal import iirnotch, filtfilt

from src.tms_eeg.config.settings import ProjectConfig

class Filter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def eeg_bp_filter(self, inst):
        """Apply bandpass filter to raw EEG data."""
        bp = self.config.filters.eeg_bandpass
        filtered = inst.copy().filter(l_freq=bp[0], h_freq=bp[1], picks='eeg')
        
        return filtered
        
    def emg_bp_filter(self, raw):
        """Apply bandpass filter to raw EMG data."""
        bp = self.config.filters.emg_bandpass
        filtered = raw.copy().filter(l_freq=bp[0], h_freq=bp[1], picks='emg')
        
        return filtered
    
    def notch_filter_epochs(self, data, band=None):
        """
        Apply notch filter using scipy.

        Parameters
        ----------
        data : Raw or Epochs
        band : tuple(float, float), optional
            (low, high) frequency band to notch, e.g. (58, 62).
            If None, uses config.filters.notch as center freq(s) with Q=30.
        """
        data = data.copy()
        sfreq = data.info['sfreq']
        arr = data.get_data()

        if band is not None:
            low, high = band
            f0 = (low + high) / 2
            Q = f0 / (high - low)
            b, a = iirnotch(w0=f0, Q=Q, fs=sfreq)
            arr = filtfilt(b, a, arr, axis=-1)
        else:
            notch_freqs = self.config.filters.notch
            if isinstance(notch_freqs, (int, float)):
                notch_freqs = [notch_freqs]
            for f0 in notch_freqs:
                b, a = iirnotch(w0=f0, Q=30, fs=sfreq)
                arr = filtfilt(b, a, arr, axis=-1)

        data._data = arr
        return data