import mne
from src.tms_eeg.config.settings import ProjectConfig

class TEPPlotter:
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def plot_teps_by_8bit_trigger(self, epochs_by_trigger: dict) -> None:
        """Plot TEPs grouped by 8-bit trigger values."""
        for trigger_value, epochs in epochs_by_trigger.items():
            if epochs is None or len(epochs) == 0:
                continue
            evoked = epochs.average()
            self.plot_butterfly(evoked, title=f'TEPs - 8-bit Trigger {trigger_value}')
    
    def plot_comparison_by_8bit_triggers(self, epochs_by_trigger: dict) -> None:
        """Plot comparison of TEPs across different 8-bit triggers."""
        evokeds = {f'Trigger {k}': v.average() for k, v in epochs_by_trigger.items() 
                  if v is not None and len(v) > 0}
        mne.viz.plot_compare_evokeds(evokeds)
        
    def plot_butterfly(self, evoked: mne.Evoked, title: str = None) -> None:
        """Plot butterfly plot for evoked data."""
        evoked.plot(spatial_colors='group', gfp=True, titles=title)
    
    def plot_component_latencies(self, components: dict) -> None:
        """Plot component latencies."""
        # Implementation for component analysis
        pass
    
    def plot_tfr(self, tfr: mne.time_frequency.AverageTFR) -> None:
        """Plot time-frequency representation."""
        tfr.plot()
