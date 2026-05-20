from scipy.signal import iirnotch, filtfilt

from tms_eeg.config.settings import ProjectConfig


class Filter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def bp_filter(self, inst, ch_type: str):
        """Apply FIR bandpass filter to EEG or EMG data."""
        bp = getattr(self.config.filters, f'{ch_type}_bandpass')

        return inst.copy().filter(
            l_freq=bp[0],
            h_freq=bp[1],
            picks=ch_type,
            method='fir',
            phase='zero',
            fir_design='firwin',
            verbose=True,
        )
    
    def notch_filter(self, data, band=None):
        """Apply notch filter using scipy.iirnotch.
        
        Parameters
        ----------
        data : Raw or Epochs
        band : tuple(float, float), optional
            (low, high) band to notch. If None, uses config.filters.notch
            with Q=30.
        """
        data = data.copy().load_data()
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

        data._data[:] = arr
        return data