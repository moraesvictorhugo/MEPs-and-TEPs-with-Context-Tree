import mne
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
from src.tms_eeg.config.settings import ProjectConfig


@dataclass
class TextFileParser:
    """Parser for text files containing trial conditions."""
    
    subject_id: str
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory for the subject."""
        return Path(__file__).parents[3] / "data" / "raw" / f"{self.subject_id}_data"
    
    def find_text_file(self) -> Optional[Path]:
        """Find the text file with trial conditions in the subject's data folder."""
        try:
            # Look for .txt files in the subject's data folder
            txt_files = list(self.data_dir.glob("*.txt"))
            
            if not txt_files:
                return None
            
            # Return the first .txt file found
            return txt_files[0]
            
        except Exception as e:
            print(f"Error finding text file: {e}")
            return None
    
    def parse_trial_conditions(self, text_file: Path) -> List[str]:
        """Parse trial conditions from text file.
        
        Expected format: one condition per line, corresponding to each trial.
        Example:
        1
        2
        3
        1
        2
        ...
        """
        try:
            conditions = []
            with open(text_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        conditions.append(f"8Bit {int(line) +1}")
            
            return conditions
            
        except Exception as e:
            print(f"Error parsing text file {text_file}: {e}")
            return []


class AnnotationProcessor:
    """Processor for handling annotations with 8-bit triggers or text file fallback."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.text_parser = TextFileParser(config.subject_id)
    
    def process_annotations(self, data: Union[mne.io.Raw, mne.Epochs]) -> Union[mne.io.Raw, mne.Epochs]:
        """Process annotations to replace Stimulus A with condition labels.
        
        Args:
            data: Raw EEG data or epochs with annotations
            
        Returns:
            Data with processed annotations (same type as input)
        """
        if isinstance(data, mne.io.Raw):
            return self._process_raw_annotations(data)
        elif isinstance(data, mne.Epochs):
            return self._process_epochs_annotations(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected mne.io.Raw or mne.Epochs.")
    
    def _process_raw_annotations(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Process annotations for raw data."""
        # Step 1: Check for 8-bit triggers
        has_8bit_triggers = self._check_8bit_triggers(raw)
        
        if has_8bit_triggers:
            print("Found 8-bit trigger annotations. Using them for condition labeling.")
            raw_processed = self._replace_with_8bit_labels(raw)
        else:
            print("No 8-bit trigger annotations found. Checking for text file fallback.")
            raw_processed = self._replace_with_text_file_labels(raw)
        
        return raw_processed
    
    def _check_8bit_triggers(self, raw: mne.io.Raw) -> bool:
        """Check if 8-bit trigger annotations exist in the data."""
        try:
            # Look for 8-bit trigger annotations
            annotations = raw.annotations
            trigger_annotations = [desc for desc in annotations.description 
                                 if desc.startswith('8Bit')]
            
            return len(trigger_annotations) > 0
            
        except Exception as e:
            print(f"Error checking for 8-bit triggers: {e}")
            return False
    
    def _replace_with_8bit_labels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Replace Stimulus A annotations with 8-bit trigger labels.
        
        This method finds Stimulus A events and replaces them with the 
        corresponding 8-bit trigger label from the same time window.
        """
        try:
            # Get all annotations
            annotations = raw.annotations
            
            # Find Stimulus A events
            stimulus_a_indices = []
            stimulus_a_times = []
            for i, desc in enumerate(annotations.description):
                if desc == 'Stimulus A':
                    stimulus_a_indices.append(i)
                    stimulus_a_times.append(annotations.onset[i])
            
            if not stimulus_a_indices:
                print("No Stimulus A events found.")
                return raw
            
            # Find 8-bit trigger events
            trigger_indices = []
            trigger_times = []
            trigger_descriptions = []
            for i, desc in enumerate(annotations.description):
                if desc.startswith('8Bit'):
                    trigger_indices.append(i)
                    trigger_times.append(annotations.onset[i])
                    trigger_descriptions.append(desc)
            
            if not trigger_times:
                print("No 8-bit trigger events found.")
                return raw
            
            # Convert to numpy arrays for easier processing
            stimulus_a_times = np.array(stimulus_a_times)
            trigger_times = np.array(trigger_times)
            
            # Create new descriptions (start with original)
            new_descriptions = list(annotations.description)
            
            # For each Stimulus A event, find the closest 8-bit trigger
            for i, stim_time in enumerate(stimulus_a_times):
                # Find 8-bit triggers within ±100ms of Stimulus A
                time_diff = np.abs(trigger_times - stim_time)
                valid_triggers = time_diff < 0.1  # 100ms tolerance
                
                if np.any(valid_triggers):
                    # Use the closest 8-bit trigger
                    closest_idx = np.argmin(time_diff)
                    new_label = trigger_descriptions[closest_idx]
                    
                    # Replace the Stimulus A label with the 8-bit trigger label
                    stim_idx = stimulus_a_indices[i]
                    new_descriptions[stim_idx] = new_label
                    
                    print(f"Replaced Stimulus A at {stim_time:.3f}s with {new_label}")
                else:
                    print(f"No 8-bit trigger found near Stimulus A at {stim_time:.3f}s")
            
            # Create new annotations with updated descriptions
            new_annotations = mne.Annotations(
                onset=annotations.onset,
                duration=annotations.duration,
                description=new_descriptions
            )
            
            # Create new raw object with updated annotations
            raw_processed = raw.copy()
            raw_processed.set_annotations(new_annotations)
            
            return raw_processed
            
        except Exception as e:
            print(f"Error replacing with 8-bit labels: {e}")
            return raw
    
    def _replace_with_text_file_labels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Replace Stimulus A annotations using text file mapping.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Raw data with Stimulus A replaced by text file conditions
        """
        try:
            # Find text file
            text_file = self.text_parser.find_text_file()
            
            if text_file is None:
                print("No text file found. Keeping original Stimulus A annotations.")
                return raw
            
            print(f"Found text file: {text_file}")
            
            # Parse trial conditions from text file
            conditions = self.text_parser.parse_trial_conditions(text_file)
            
            if not conditions:
                print("No valid conditions found in text file. Keeping original annotations.")
                return raw
            
            # Get Stimulus A events using safer approach
            stimulus_a_indices = []
            stimulus_a_times = []
            for i, desc in enumerate(raw.annotations.description):
                if desc == 'Stimulus A':
                    stimulus_a_indices.append(i)
                    stimulus_a_times.append(raw.annotations.onset[i])
            
            if not stimulus_a_indices:
                print("No Stimulus A events found. Cannot apply text file mapping.")
                return raw
            
            # Ensure we have enough conditions for all Stimulus A events
            if len(conditions) < len(stimulus_a_indices):
                print(f"Warning: Only {len(conditions)} conditions in text file, "
                      f"but {len(stimulus_a_indices)} Stimulus A events. "
                      f"Repeating conditions to fill remaining events.")
                
                # Repeat conditions to match number of Stimulus A events
                conditions = (conditions * 
                             (len(stimulus_a_indices) // len(conditions) + 1))[
                             :len(stimulus_a_indices)]
            
            # Create new descriptions (start with original)
            new_descriptions = list(raw.annotations.description)
            
            # Replace Stimulus A events with text file conditions
            for i, stim_idx in enumerate(stimulus_a_indices):
                if i < len(conditions):
                    new_descriptions[stim_idx] = conditions[i]
                    print(f"Replaced Stimulus A at {raw.annotations.onset[stim_idx]:.3f}s "
                          f"with {conditions[i]}")
            
            # Create new annotations
            new_annotations = mne.Annotations(
                onset=raw.annotations.onset,
                duration=raw.annotations.duration,
                description=new_descriptions
            )
            
            # Create new raw object with updated annotations
            raw_processed = raw.copy()
            raw_processed.set_annotations(new_annotations)
            
            return raw_processed
            
        except Exception as e:
            print(f"Error replacing with text file labels: {e}")
            return raw
    
    def _process_epochs_annotations(self, epochs: mne.Epochs) -> mne.Epochs:
        """Process annotations for epochs data."""
        # For epochs, we need to work with the underlying raw data's annotations
        # since epochs don't store annotations directly, they reference the raw data
        
        # Check if the epochs have metadata with condition information
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            # If epochs already have metadata, we might not need to process annotations
            print("Epochs already have metadata. Skipping annotation processing.")
            return epochs
        
        # For epochs, we need to process the annotations on the underlying raw data
        # and then recreate epochs. However, this is complex and might not be the best approach.
        # Let's implement a simpler approach: work with the epochs' event_id and events
        
        # Check if we have 8-bit triggers in the event_id
        has_8bit_triggers = any(desc.startswith('8Bit') for desc in epochs.event_id.keys())
        
        if has_8bit_triggers:
            print("Found 8-bit trigger events in epochs. Using them for condition labeling.")
            # The epochs already have the correct event_id, so we can return them as is
            return epochs
        else:
            print("No 8-bit trigger events found in epochs. Checking for text file fallback.")
            return self._replace_epochs_with_text_file_labels(epochs)
    
    def _replace_epochs_with_text_file_labels(self, epochs: mne.Epochs) -> mne.Epochs:
        """Replace Stimulus A events in epochs using text file mapping."""
        try:
            # Find text file
            text_file = self.text_parser.find_text_file()
            
            if text_file is None:
                print("No text file found. Keeping original Stimulus A events in epochs.")
                return epochs
            
            print(f"Found text file: {text_file}")
            
            # Parse trial conditions from text file
            conditions = self.text_parser.parse_trial_conditions(text_file)
            
            if not conditions:
                print("No valid conditions found in text file. Keeping original events.")
                return epochs
            
            # Get Stimulus A events from epochs
            stimulus_a_events = []
            stimulus_a_indices = []
            
            # Find events with 'Stimulus A' description
            for i, (event_time, event_id) in enumerate(zip(epochs.events[:, 0], epochs.events[:, 2])):
                # Convert event time to time in seconds
                event_time_sec = epochs.times[event_time - epochs.tmin * epochs.info['sfreq']]
                
                # Check if this event corresponds to 'Stimulus A' in the event_id mapping
                event_desc = None
                for desc, id_val in epochs.event_id.items():
                    if id_val == event_id:
                        event_desc = desc
                        break
                
                if event_desc == 'Stimulus A':
                    stimulus_a_events.append(event_time_sec)
                    stimulus_a_indices.append(i)
            
            if not stimulus_a_indices:
                print("No Stimulus A events found in epochs. Cannot apply text file mapping.")
                return epochs
            
            # Ensure we have enough conditions for all Stimulus A events
            if len(conditions) < len(stimulus_a_indices):
                print(f"Warning: Only {len(conditions)} conditions in text file, "
                      f"but {len(stimulus_a_indices)} Stimulus A events. "
                      f"Repeating conditions to fill remaining events.")
                
                # Repeat conditions to match number of Stimulus A events
                conditions = (conditions * 
                             (len(stimulus_a_indices) // len(conditions) + 1))[
                             :len(stimulus_a_indices)]
            
            # Create new event_id mapping
            new_event_id = epochs.event_id.copy()
            if 'Stimulus A' in new_event_id:
                del new_event_id['Stimulus A']
            
            # Add new conditions to event_id
            for condition in set(conditions):  # Use set to avoid duplicates
                if condition not in new_event_id:
                    # Find a new ID that's not already used
                    new_id = max(new_event_id.values()) + 1 if new_event_id else 1
                    new_event_id[condition] = new_id
            
            # Update events array with new condition labels
            new_events = epochs.events.copy()
            for i, stim_idx in enumerate(stimulus_a_indices):
                if i < len(conditions):
                    condition = conditions[i]
                    new_events[stim_idx, 2] = new_event_id[condition]
                    print(f"Replaced Stimulus A event at {stimulus_a_events[i]:.3f}s "
                          f"with {condition}")
            
            # Create new epochs with updated events and event_id
            # We need to recreate the epochs object with the new events
            new_epochs = mne.EpochsArray(
                epochs.get_data(),
                epochs.info,
                tmin=epochs.tmin,
                events=new_events,
                event_id=new_event_id,
                metadata=epochs.metadata
            )
            
            # Return the EpochsArray directly - it's compatible with Epochs interface
            return new_epochs
            
        except Exception as e:
            print(f"Error replacing epochs with text file labels: {e}")
            return epochs
