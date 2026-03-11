import mne
import os
from pathlib import Path
from typing import Union, List


class EEGWriter:
    """Class for saving EEG data (raw, epochs, evoked) to the processed directory."""
    
    def __init__(self, config):
        """
        Initialize EEGWriter with project configuration.
        
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
        grand_average = mne.grand_average(evoked_list)  # type: ignore
        if hasattr(grand_average, 'comment'):
            grand_average.comment = 'Grand average TEPs'  # type: ignore
        self.save_evoked(grand_average, subfolder, "grand_average")  # type: ignore
        
        print(f"All evoked responses saved successfully!")
