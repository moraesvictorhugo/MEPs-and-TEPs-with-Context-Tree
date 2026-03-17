# features.py

import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class FeatureExtractor:
    """Extrai features de amplitude de epochs EEG."""

    def __init__(
        self,
        epochs: mne.Epochs,
        channels: List[str],
        time_windows: Dict[str, Tuple[float, float]],
    ):
        self.epochs = epochs
        self.channels = channels
        self.time_windows = time_windows

    def get_evokeds(
        self,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, mne.Evoked]:
        """
        Retorna o Evoked (TEP médio) por condição, já filtrado nos canais de interesse.
        """
        if conditions is None:
            conditions = list(self.epochs.event_id.keys())

        return {
            cond: self.epochs[cond].average().pick(self.channels)
            for cond in conditions
        }

    def peak_to_peak(
        self,
        component1: str = "N15",
        component2: str = "P30",
        evokeds: Optional[Dict[str, mne.Evoked]] = None,
    ) -> pd.DataFrame:
        """
        Amplitude pico-a-pico entre dois componentes, calculada sobre
        o Evoked de cada condição/canal.
        """
        if evokeds is None:
            evokeds = self.get_evokeds()

        t1_min, t1_max = self.time_windows[component1]
        t2_min, t2_max = self.time_windows[component2]
        label = f"{component1}-{component2}"
        times = self.epochs.times
        mask1 = (times >= t1_min) & (times <= t1_max)
        mask2 = (times >= t2_min) & (times <= t2_max)

        rows = []
        for cond, evoked in evokeds.items():
            for ch in self.channels:
                signal = evoked.copy().pick([ch]).data.squeeze()  # (n_times,)
                val1 = signal[mask1].min()
                val2 = signal[mask2].max()

                rows.append({
                    "condition": cond,
                    "channel": ch,
                    "component": label,
                    "peak_to_peak": val2 - val1,
                })

        return pd.DataFrame(rows)
    
    def compute_gmfp(
        self,
        evokeds: Optional[Dict[str, mne.Evoked]] = None,
    ) -> Dict[str, np.ndarray]:
        if evokeds is None:
            evokeds = self.get_evokeds()

        return {
            cond: evoked.data.std(axis=0)
            for cond, evoked in evokeds.items()
        }