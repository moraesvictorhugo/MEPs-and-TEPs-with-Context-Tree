"""EMG visualization class."""

import mne
from typing import Optional


class EMGPlotter:
    """Plots EMG-related visualizations from epochs."""

    def __init__(self, xlim: tuple = (-0.01, 0.08)):
        self.xlim = xlim

    def plot_all(self, epochs: mne.Epochs):
        """Plot EMG evoked for each condition."""
        for condition in epochs.event_id.keys():
            evoked_emg = epochs[condition].average(picks='emg')
            self.plot_emg(evoked_emg, condition)

    def plot_emg(self, evoked_emg: mne.Evoked, condition: str):
        """Plot single EMG evoked."""
        evoked_emg.plot(
            xlim=self.xlim,
            titles=f'EMG - {condition}',
        )
