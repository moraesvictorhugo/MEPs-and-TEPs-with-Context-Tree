import mne
from src.tms_eeg.config.settings import ProjectConfig

class ArtifactRemover:
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def remove_tms_artifact(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Remove TMS artifact from raw data using configured parameters.
    
        Uses the artifact removal window and mode from settings.
        """
        artifact_cfg = self.config.artifact
        
        # Use settings from config
        window = artifact_cfg.window_removal_artifact
        mode = artifact_cfg.mode_removal_artifact
        
        events, event_id = mne.events_from_annotations(raw)
        
        if mode == 'linear':
            raw_artifact_free = raw.copy()
            
            # Apply fix_stim_artifact using these events
            mne.preprocessing.fix_stim_artifact(
                raw_artifact_free,
                events=events,
                event_id=event_id,
                tmin=window[0],
                tmax=window[1],
                mode=mode
            )
            
        else:
            raise ValueError(f"Unsupported artifact removal mode yet.")
        
        return raw_artifact_free