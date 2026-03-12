"""TEP (TMS-Evoked Potential) visualization class."""

import mne
from typing import List, Optional, Union
from pathlib import Path

class TEPPlotter:
    """Plots TEP-related visualizations from epochs or evoked objects."""

    def __init__(
        self,
        config=None,
        xlim: tuple = None,
        topo_times: Optional[List[float]] = None,
        joint_times: Optional[List[float]] = None,
        roi_picks: Optional[List[str]] = None,
        writer=None,
    ):
        self.config = config
        self.xlim = xlim or (config.plots.tep_xlim if config else (-0.02, 0.2))
        self.topo_times = topo_times or (config.plots.tep_topo_times if config else [
            0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.1
        ])
        self.joint_times = joint_times or (config.plots.tep_joint_times if config else [
            0.015, 0.03, 0.045, 0.06, 0.1, 0.18
        ])
        self.roi_picks = roi_picks or (config.plots.tep_roi_channels if config else [
            'C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'
        ])
        self.writer = writer

    def _save_figure(self, fig, name: str, condition: str):
        """Save figure if save_figs is enabled in config."""
        if self.config and self.config.io.save_figs and fig is not None:
            if self.writer:
                writer = self.writer
            else:
                from src.tms_eeg.io.writer import EEGWriter
                writer = EEGWriter(self.config)
            
            filename = f"tep_{name}_{condition}"
            writer.save_figure(fig, filename)

    def plot_all(self, epochs: mne.Epochs):
        """Run all TEP plots for each condition in the epochs."""
        for condition in epochs.event_id.keys():
            evoked = epochs[condition].average(picks='eeg')
            if hasattr(evoked, 'comment'):
                evoked.comment = f'TEPs for {condition}'
            self._plot_condition(evoked, condition)

    def _plot_condition(self, evoked: mne.Evoked, condition: str):
        """Plot all TEP visualizations for a single condition."""
        title = f'TEPs - {condition}'

        self.plot_butterfly(evoked, condition)
        self.plot_roi(evoked, condition)
        self.plot_joint(evoked, title, condition)
        self.plot_image(evoked, condition)
        self.plot_topomap_animation(evoked, condition)
        self.plot_gfp(evoked, condition)
        self.plot_topomap(evoked, condition)
        self.plot_topo(evoked, condition)

    def plot_butterfly(self, evoked: mne.Evoked, condition: str):
        """Butterfly plot of all EEG channels."""
        fig = evoked.plot(xlim=self.xlim)
        self._save_figure(fig, "butterfly", condition)

    def plot_roi(self, evoked: mne.Evoked, condition: str):
        """Plot selected ROI channels."""
        fig = evoked.plot(
            picks=self.roi_picks,
            titles=f'TEPs for {condition}',
            xlim=self.xlim,
        )
        self._save_figure(fig, "roi", condition)

    def plot_joint(self, evoked: mne.Evoked, title: str, condition: str):
        """Joint plot (butterfly + topomaps at specific times)."""
        fig = evoked.plot_joint(
            times=self.joint_times,
            title=title,
            ts_args=dict(xlim=self.xlim),
        )
        self._save_figure(fig, "joint", condition)

    def plot_image(self, evoked: mne.Evoked, condition: str):
        """Heatmap image of all EEG channels."""
        fig = evoked.plot_image(picks='eeg', xlim=self.xlim, show_names='all')
        self._save_figure(fig, "image", condition)

    def plot_topomap_animation(self, evoked: mne.Evoked, condition: str):
        """Animated topomap across time points."""
        # animate_topomap returns a tuple, not a figure
        result = evoked.copy().crop(tmin=self.xlim[0], tmax=self.xlim[1]).animate_topomap(
            times=self.joint_times, frame_rate=1
        )
        # Don't try to save animation as a figure since it's interactive
        # Just display it for now
        pass

    def plot_gfp(self, evoked: mne.Evoked, condition: str):
        """Butterfly plot with Global Field Power."""
        fig = evoked.plot(gfp=True, xlim=self.xlim)
        self._save_figure(fig, "gfp", condition)

    def plot_topomap(self, evoked: mne.Evoked, condition: str):
        """Static topomaps at multiple time points."""
        fig = evoked.plot_topomap(times=self.topo_times, colorbar=True)
        self._save_figure(fig, "topomap", condition)

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
        fig = evoked_crop.plot_topo(
            ylim=ylim,
            vline=(0.0,),
            title=f'TEPs por canal - {condition}',
            color='blue',
            background_color='white',
        )
        self._save_figure(fig, "topo", condition)

