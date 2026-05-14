import mne
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional

import re
import numpy as np
from scipy.io import savemat


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
        
    def save_epochs_to_mat(
        self,
        epochs: mne.Epochs,
        subfolder: str = "processed",
        post_pulse_window: tuple = (0.015, 0.450),
        pre_pos_pulse_window: tuple = (-0.050, 0.400),
    ) -> None:
        """
        Save epochs to a MATLAB .mat file with two time windows and context symbols.

        The output file contains:
            - X: post-pulse data, shape (n_channels, n_times_post, n_epochs)
            - X_pre_pos_pulse: pre/post-pulse data, shape (n_channels, n_times_pre, n_epochs)
            - X_ter: context symbol per epoch (0, 1 or 2), shape (n_epochs, 1)
            - fs: sampling frequency (Hz)
            - ch_names: list of channel names
            - times_post / times_pre: time vectors (seconds) for each window

        The mapping from epoch event name to symbol uses
        `config.analysis.name_to_symbol` after normalization
        (lowercase + whitespace removed), so it is robust to label
        variations like "8Bit 1", "8bit 1", "8 bit1", etc.

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object to be exported. Must contain event names that
            normalize to keys present in `config.analysis.name_to_symbol`.
        subfolder : str
            Subfolder within the subject's processed directory (default: 'processed').
        post_pulse_window : tuple
            (tmin, tmax) in seconds for the post-pulse window. Default: (0.015, 0.450).
        pre_pos_pulse_window : tuple
            (tmin, tmax) in seconds for the pre/post-pulse window. Default: (-0.050, 0.400).

        Raises
        ------
        ValueError
            If an epoch's event name cannot be mapped to a symbol.
        """
        # Check if export is enabled
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return

        # --- 1. Build symbol vector (X_ter) from epochs.events + epochs.event_id ---
        name_to_symbol = self.config.analysis.name_to_symbol

        def _normalize(name: str) -> str:
            """Lowercase and remove all whitespace."""
            return re.sub(r"\s+", "", str(name).lower())

        # Reverse map: numeric code → event name (string)
        code_to_name = {code: name for name, code in epochs.event_id.items()}

        symbols = np.empty(len(epochs), dtype=np.int32)
        for i, code in enumerate(epochs.events[:, 2]):
            name = code_to_name.get(int(code))
            if name is None:
                raise ValueError(
                    f"[Writer] Epoch {i}: event code {code} not found in "
                    f"epochs.event_id ({epochs.event_id})."
                )
            norm = _normalize(name)
            if norm not in name_to_symbol:
                raise ValueError(
                    f"[Writer] Epoch {i}: event name '{name}' (normalized: '{norm}') "
                    f"not found in config.analysis.name_to_symbol "
                    f"({list(name_to_symbol.keys())})."
                )
            symbols[i] = name_to_symbol[norm]

        X_ter = symbols.reshape(-1, 1)  # column vector (n_epochs, 1)

        # --- 2. Crop both time windows (without modifying the original) ---
        epochs_post = epochs.copy().crop(tmin=post_pulse_window[0],
                                        tmax=post_pulse_window[1])
        epochs_pre = epochs.copy().crop(tmin=pre_pos_pulse_window[0],
                                        tmax=pre_pos_pulse_window[1])

        # MNE shape: (n_epochs, n_channels, n_times)
        # MATLAB convention: (n_channels, n_times, n_epochs) → transpose (1, 2, 0)
        X = epochs_post.get_data().transpose(1, 2, 0)
        X_pre_pos_pulse = epochs_pre.get_data().transpose(1, 2, 0)

        # --- 3. Build output dictionary ---
        mat_dict = {
            "X": X.astype(np.float32),
            "X_pre_pos_pulse": X_pre_pos_pulse.astype(np.float32),
            "X_ter": X_ter,
            "fs": float(epochs.info["sfreq"]),
            "ch_names": np.array(epochs.ch_names, dtype=object),
            "times_post": epochs_post.times.astype(np.float64),
            "times_pre": epochs_pre.times.astype(np.float64),
        }

        # --- 4. Save ---
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)

        filename = self._get_filename("epochs", "contexttree", extension=".mat")
        full_path = processed_dir / filename

        print(f"Saving .mat file to: {full_path}")
        print(f"  X shape: {X.shape} (channels, times, epochs)")
        print(f"  X_pre_pos_pulse shape: {X_pre_pos_pulse.shape}")
        print(f"  X_ter shape: {X_ter.shape}  | unique symbols: {np.unique(X_ter).tolist()}")
        print(f"  fs: {mat_dict['fs']} Hz | n_channels: {len(epochs.ch_names)}")

        savemat(full_path, mat_dict, do_compression=True)
        print(".mat file saved successfully!")

