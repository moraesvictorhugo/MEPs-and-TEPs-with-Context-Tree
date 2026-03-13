import mne
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Union, List

class Writer:
    """Class for saving EEG and EMG data (raw, epochs, evoked) to the processed directory."""
    
    def __init__(self, config):
        """
        Initialize Writer with project configuration.
        
        Parameters
        ----------
        config : ProjectConfig
            Configuration object containing subject_id and other settings
        """
        self.config = config
        self.subject_id = config.subject_id
        
    def _create_processed_dir(self) -> Path:
        """
        Create the processed data directory if it doesn't exist.
        
        Returns
        -------
        Path
            Path to the processed directory for this subject
        """
        processed_dir = Path("data/processed") / self.subject_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir
    
    def _get_filename(self, base_name: str, suffix: str, extension: str = ".fif") -> str:
        """
        Generate a standardized filename.
        
        Parameters
        ----------
        base_name : str
            Base name for the file (e.g., 'epochs', 'evoked')
        suffix : str
            Additional identifier (e.g., condition name, 'processed')
        extension : str
            File extension (default: '.fif')
            
        Returns
        -------
        str
            Complete filename
        """
        return f"{self.subject_id}_{base_name}_{suffix}{extension}"
    
    def save_epochs(self, epochs: mne.Epochs, subfolder: str = "processed") -> None:
        """
        Save processed epochs to the processed directory.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Processed epochs object
        subfolder : str
            Subfolder name within processed directory (default: 'processed')
        """
        # Check if export is enabled in configuration
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        filename = self._get_filename("epochs", "processed")
        full_path = processed_dir / filename
        
        print(f"Saving epochs to: {full_path}")
        epochs.save(full_path, overwrite=True)
        print(f"Epochs saved successfully!")
    
    def save_evoked(self, evoked: mne.Evoked, subfolder: str = "processed", 
                   condition: str = "average") -> None:
        """
        Save evoked response data to the processed directory.
        
        Parameters
        ----------
        evoked : mne.Evoked
            Evoked response object
        subfolder : str
            Subfolder name within processed directory (default: 'processed')
        condition : str, optional
            Condition name for filename (e.g., 'stimulus_0', 'stimulus_1')
        """
        # Check if export is enabled in configuration
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        suffix = condition if condition else "average"
            
        filename = self._get_filename("evoked", suffix)
        full_path = processed_dir / filename
        
        print(f"Saving evoked data to: {full_path}")
        evoked.save(full_path, overwrite=True)
        print(f"Evoked data saved successfully!")
    
    def save_evoked_conditions(self, epochs: mne.Epochs, subfolder: str = "processed") -> None:
        """
        Save evoked responses for all conditions in the epochs object.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object containing different conditions
        subfolder : str
            Subfolder name within processed directory (default: 'processed')
        """
        # Check if export is enabled in configuration
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        print(f"Saving evoked responses for {len(epochs.event_id)} conditions...")
        
        evoked_list: List[mne.Evoked] = []
        for condition in epochs.event_id.keys():
            evoked = epochs[condition].average()
            if hasattr(evoked, 'comment'):
                evoked.comment = f'TEPs for {condition}'
            # Ensure we have the right type
            evoked_list.append(evoked)  # type: ignore
            self.save_evoked(evoked, subfolder, condition)
        
        # Also save grand average across all conditions
        grand_average = mne.grand_average(evoked_list)
        if hasattr(grand_average, 'comment'):
            grand_average.comment = 'Grand average TEPs'
        self.save_evoked(grand_average, subfolder, "grand_average")
        
        print(f"All evoked responses saved successfully!")
        
    def save_raw(self, raw: mne.io.Raw, subfolder: str = "processed", 
             suffix: str = "processed") -> None:
        """
        Save raw EEG data to the processed directory.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data object
        subfolder : str
            Subfolder name within processed directory (default: 'processed')
        suffix : str
            Additional identifier for filename (default: 'processed')
        """
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        filename = self._get_filename("raw", suffix)
        full_path = processed_dir / filename
        
        print(f"Saving raw data to: {full_path}")
        raw.save(full_path, overwrite=True)
        print(f"Raw data saved successfully!")
        
    def _create_figures_dir(self, subfolder: str = "figures") -> Path:
        """
        Create the figures directory for this subject.
        
        Parameters
        ----------
        subfolder : str
            Subfolder name (default: 'figures')
            
        Returns
        -------
        Path
            Path to the figures directory
        """
        figures_dir = self._create_processed_dir() / subfolder
        figures_dir.mkdir(exist_ok=True)
        return figures_dir

    def save_figure(self, fig: Figure, name: str, subfolder: str = None,
                    fmt: str = None, dpi: int = None) -> None:
        """
        Save a single matplotlib figure.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        name : str
            Descriptive name for the file (e.g., 'butterfly_8bits_1')
        subfolder : str
            Subfolder within processed directory (default: from config)
        fmt : str
            Image format (default: from config)
        dpi : int
            Resolution (default: from config)
        """
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
        
        # Use config defaults if not provided
        subfolder = subfolder or (self.config.plots.figure_subfolder if hasattr(self.config, 'plots') and hasattr(self.config.plots, 'figure_subfolder') else "figures")
        fmt = fmt or (self.config.plots.figure_format if hasattr(self.config, 'plots') and hasattr(self.config.plots, 'figure_format') else "png")
        dpi = dpi or (self.config.plots.figure_dpi if hasattr(self.config, 'plots') and hasattr(self.config.plots, 'figure_dpi') else 300)
        
        try:
            figures_dir = self._create_figures_dir(subfolder)
            filename = self._get_filename(name, "plot", extension=f".{fmt}")
            full_path = figures_dir / filename
            
            fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {full_path}")
        except Exception as e:
            print(f"Error saving figure '{name}': {e}")

    def save_all_open_figures(self, prefix: str = "fig", subfolder: str = "figures",
                            fmt: str = "png", dpi: int = 300) -> None:
        """
        Save all currently open matplotlib figures.
        
        Parameters
        ----------
        prefix : str
            Prefix for filenames (default: 'fig')
        subfolder : str
            Subfolder within processed directory (default: 'figures')
        fmt : str
            Image format (default: 'png')
        dpi : int
            Resolution (default: 300)
        """
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
        
        figures_dir = self._create_figures_dir(subfolder)
        fig_nums = plt.get_fignums()
        
        if not fig_nums:
            print("No open figures to save.")
            return
        
        for i, num in enumerate(fig_nums):
            fig = plt.figure(num)
            filename = self._get_filename(f"{prefix}_{i+1:03d}", "plot", extension=f".{fmt}")
            full_path = figures_dir / filename
            fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {full_path}")
        
        print(f"Saved {len(fig_nums)} figures.")

    def save_emg_epochs(self, epochs: mne.Epochs, subfolder: str = "emg_processed") -> None:
        """
        Save EMG epochs to the processed directory with preserved annotations.
        
        Parameters
        ----------
        epochs : mne.Epochs
            EMG epochs object with annotations
        subfolder : str
            Subfolder name within processed directory (default: 'emg_processed')
        """
        # Check if export is enabled in configuration
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        filename = self._get_filename("emg_epochs", "processed")
        full_path = processed_dir / filename
        
        print(f"Saving EMG epochs to: {full_path}")
        epochs.save(full_path, overwrite=True)
        print(f"EMG epochs saved successfully!")
