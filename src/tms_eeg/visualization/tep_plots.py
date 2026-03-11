"""TEP (TMS-Evoked Potential) visualization class."""

import mne
from typing import List, Optional

class TEPPlotter:
    """Plots TEP-related visualizations from epochs or evoked objects."""

    def __init__(
        self,
        xlim: tuple = (-0.02, 0.2),
        topo_times: Optional[List[float]] = None,
        joint_times: Optional[List[float]] = None,
        roi_picks: Optional[List[str]] = None,
    ):
        self.xlim = xlim
        self.topo_times = topo_times or [
            0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.1
        ]
        self.joint_times = joint_times or [
            0.015, 0.03, 0.045, 0.06, 0.1, 0.18
        ]
        self.roi_picks = roi_picks or [
            'C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'
        ]

    def plot_all(self, epochs: mne.Epochs):
        """Run all TEP plots for each condition in the epochs."""
        for condition in epochs.event_id.keys():
            evoked = epochs[condition].average(picks='eeg')
            evoked.comment = f'TEPs for {condition}'
            self._plot_condition(evoked, condition)

    def _plot_condition(self, evoked: mne.Evoked, condition: str):
        """Plot all TEP visualizations for a single condition."""
        title = f'TEPs - {condition}'

        self.plot_butterfly(evoked)
        self.plot_roi(evoked, condition)
        self.plot_joint(evoked, title)
        self.plot_image(evoked)
        self.plot_topomap_animation(evoked)
        self.plot_gfp(evoked)
        self.plot_topomap(evoked)
        self.plot_topo(evoked, condition)

    def plot_butterfly(self, evoked: mne.Evoked):
        """Butterfly plot of all EEG channels."""
        evoked.plot(xlim=self.xlim)

    def plot_roi(self, evoked: mne.Evoked, condition: str):
        """Plot selected ROI channels."""
        evoked.plot(
            picks=self.roi_picks,
            titles=f'TEPs for {condition}',
            xlim=self.xlim,
        )

    def plot_joint(self, evoked: mne.Evoked, title: str):
        """Joint plot (butterfly + topomaps at specific times)."""
        evoked.plot_joint(
            times=self.joint_times,
            title=title,
            ts_args=dict(xlim=self.xlim),
        )

    def plot_image(self, evoked: mne.Evoked):
        """Heatmap image of all EEG channels."""
        evoked.plot_image(picks='eeg', xlim=self.xlim, show_names='all')

    def plot_topomap_animation(self, evoked: mne.Evoked):
        """Animated topomap across time points."""
        evoked.copy().crop(tmin=self.xlim[0], tmax=self.xlim[1]).animate_topomap(
            times=self.joint_times, frame_rate=1
        )

    def plot_gfp(self, evoked: mne.Evoked):
        """Butterfly plot with Global Field Power."""
        evoked.plot(gfp=True, xlim=self.xlim)

    def plot_topomap(self, evoked: mne.Evoked):
        """Static topomaps at multiple time points."""
        evoked.plot_topomap(times=self.topo_times, colorbar=True)

    def plot_topo(
        self,
        evoked: mne.Evoked,
        condition: str,
        ylim: dict = None,
        crop: tuple = (-0.01, 0.2),
    ):
        """Topo plot (one waveform per channel location)."""
        ylim = ylim or dict(eeg=[-10, 10])
        evoked_crop = evoked.copy().crop(tmin=crop[0], tmax=crop[1])
        evoked_crop.plot_topo(
            ylim=ylim,
            vline=(0.0,),
            title=f'TEPs por canal - {condition}',
            color='blue',
            background_color='white',
        )

