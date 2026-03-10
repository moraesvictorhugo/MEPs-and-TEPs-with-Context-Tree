from curses import raw

import mne
import numpy as np
from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.visualization.tep_plots import TEPPlotter


class EEGEpocher:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def find_events(self, raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict]:
        """Find events in raw data.

        Args:
            raw (mne.io.BaseRaw): Raw EEG data.

        Returns:
            tuple: (events ndarray, event_id dict)
        """
        events, event_id = mne.events_from_annotations(
            raw,
            event_id=self.config.events.trigger_id,  # type: ignore
        )
        return events, event_id

    def create_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        """Create epochs from raw data.

        Args:
            raw (mne.io.BaseRaw): Preprocessed raw EEG data.

        Returns:
            mne.Epochs: Epoched EEG data.
        """
        epoch_cfg = self.config.epochs

        events, event_id = self.find_events(raw)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=epoch_cfg.window[0],
            tmax=epoch_cfg.window[1],
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
        )

        if epoch_cfg.downsample:
            epochs.resample(sfreq=epoch_cfg.downsample_freq)

        return epochs

    @staticmethod
    def reject_bad(epochs: mne.Epochs, thresholds: dict, flat_thresholds: 
        dict = None) -> mne.Epochs:
        """Reject bad epochs based on amplitude thresholds.

        Args:
            epochs (mne.Epochs): Epochs to process.
            thresholds (dict): Max amplitude thresholds per channel type (e.g. 
                {"eeg": 150e-6}).
            flat_thresholds (dict, optional): Min amplitude thresholds per 
                channel type. Defaults to None.

        Returns:
            mne.Epochs: Cleaned epochs (copy).
        """
        epochs = epochs.copy()

        available = set(
            mne.channel_type(epochs.info, i)
            for i in range(len(epochs.ch_names))
        )

        reject = {k: v for k, v in thresholds.items() if k in available}
        flat = {k: v for k, v in flat_thresholds.items() if k in available
                } if flat_thresholds else None

        epochs.drop_bad(reject=reject, flat=flat)
        print(f"[EEGEpocher] Remaining epochs: {len(epochs)}/{len(epochs.drop_log)}")
        return epochs
    
class DualEventEpocher(EEGEpocher):
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.trigger_8bits = config.events.trigger_8bits
        self.stimulus_to_8bit_mapping = config.events.stimulus_to_8bit_mapping
    
    def create_epochs_with_8bit_grouping(self, raw: mne.io.BaseRaw) -> dict:
        """Create epochs using Stimulus A but group by 8-bit triggers."""
        try:
            # Step 1: Create epochs using Stimulus A (existing behavior)
            epochs = self.create_epochs(raw)
            
            if len(epochs) == 0:
                raise ValueError("No epochs found with Stimulus A events")
            
            # Step 2: Find 8-bit trigger events
            events_8bit, event_id_8bit = self.find_8bit_events(raw)
            
            if len(events_8bit) == 0:
                print("Warning: No 8-bit trigger events found")
                return {'stimulus_a_only': epochs}
            
            # Step 3: Map epochs to 8-bit triggers
            epochs_by_trigger = self._map_epochs_to_8bit_triggers(epochs, events_8bit)
            
            # Validate that we have epochs for each trigger
            for trigger in self.trigger_8bits:
                if trigger not in epochs_by_trigger or len(epochs_by_trigger[trigger]) == 0:
                    print(f"Warning: No epochs found for 8-bit trigger {trigger}")
            
            return epochs_by_trigger
            
        except Exception as e:
            print(f"Error in create_epochs_with_8bit_grouping: {e}")
            raise
    
    def find_8bit_events(self, raw: mne.io.BaseRaw) -> tuple:
        """Find 8-bit trigger events in raw data."""
        # Create event_id mapping for 8-bit triggers
        event_id_8bit = {f'8bits {i}': i for i in self.trigger_8bits}
        
        events_8bit, _ = mne.events_from_annotations(
            raw,
            event_id=event_id_8bit,
        )
        return events_8bit, event_id_8bit
    
    def _map_epochs_to_8bit_triggers(self, epochs: mne.Epochs, events_8bit: np.ndarray) -> dict:
        """Map epochs to their corresponding 8-bit trigger values."""
        epochs_by_trigger = {}
        
        # Initialize empty epochs for each trigger
        for trigger in self.trigger_8bits:
            epochs_by_trigger[trigger] = []
        
        # Find events that correspond to each epoch
        for i, epoch_info in enumerate(epochs.selection):
            epoch_start = epochs.events[i, 0]
            epoch_end = epoch_start + int(epochs.tmax * epochs.info['sfreq'])
            
            # Find 8-bit trigger events within this epoch
            mask = (events_8bit[:, 0] >= epoch_start) & (events_8bit[:, 0] <= epoch_end)
            epoch_8bit_events = events_8bit[mask]
            
            if len(epoch_8bit_events) > 0:
                # Use the first 8-bit trigger found in the epoch
                trigger_value = epoch_8bit_events[0, 2]
                if trigger_value in self.trigger_8bits:
                    # Add this epoch to the appropriate trigger group
                    epochs_by_trigger[trigger_value].append(epochs[i:i+1])
        
        # Convert lists to Epochs objects
        for trigger in self.trigger_8bits:
            if epochs_by_trigger[trigger]:
                epochs_by_trigger[trigger] = mne.concatenate_epochs(epochs_by_trigger[trigger])
            else:
                epochs_by_trigger[trigger] = None
        
        # Remove None entries
        epochs_by_trigger = {k: v for k, v in epochs_by_trigger.items() if v is not None}
        
        return epochs_by_trigger

    def get_trigger_statistics(self, epochs_by_trigger: dict) -> dict:
        """Get statistics about epochs per trigger."""
        stats = {}
        total_epochs = sum(len(epochs) for epochs in epochs_by_trigger.values())
        
        for trigger, epochs in epochs_by_trigger.items():
            stats[trigger] = {
                'count': len(epochs),
                'percentage': (len(epochs) / total_epochs) * 100 if total_epochs > 0 else 0
            }
        
        return stats

    def validate_trigger_mapping(self, raw: mne.io.BaseRaw) -> bool:
        """Validate that 8-bit triggers are properly mapped to Stimulus A events."""
        events_stim, _ = self.find_events(raw)
        events_8bit, _ = self.find_8bit_events(raw)
        
        if len(events_stim) == 0 or len(events_8bit) == 0:
            return False
        
        # Check if 8-bit events fall within reasonable time windows of Stimulus A
        for stim_event in events_stim:
            stim_time = stim_event[0]
            # Look for 8-bit events within ±100ms of Stimulus A
            nearby_8bit = events_8bit[
                (events_8bit[:, 0] >= stim_time - 100) & 
                (events_8bit[:, 0] <= stim_time + 100)
            ]
            if len(nearby_8bit) == 0:
                return False
        
        return True

# Analysis utility functions
def process_8bit_epochs(epochs_by_trigger: dict, config: ProjectConfig) -> dict:
    """Process all 8-bit trigger epochs with consistent preprocessing."""
    processed_epochs = {}
    for trigger_value, epochs in epochs_by_trigger.items():
        if epochs is None or len(epochs) == 0:
            continue
            
        # Apply preprocessing steps
        epochs = epochs.set_eeg_reference(config.channels.eeg_reference)
        epochs = EEGEpocher.reject_bad(epochs, 
                                      config.epochs.rejection_amplitude_threshold,
                                      config.epochs.rejection_flat_threshold)
        processed_epochs[trigger_value] = epochs
    
    return processed_epochs

def analyze_8bit_triggers(epochs_by_trigger: dict, config: ProjectConfig):
    """Analyze and plot all 8-bit trigger groups."""
    for trigger_value, epochs in epochs_by_trigger.items():
        if epochs is None or len(epochs) == 0:
            continue
            
        evoked = epochs.average()
        evoked.plot(title=f'TEPs - 8-bit Trigger {trigger_value}')
        
        # Additional analysis
        print(f"Trigger {trigger_value}: {len(epochs)} epochs")

     
     
     
     
