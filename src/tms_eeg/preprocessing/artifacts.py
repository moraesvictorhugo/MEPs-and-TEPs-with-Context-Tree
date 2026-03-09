import mne
from src.tms_eeg.config.settings import ProjectConfig

class ArtifactRemover:
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def remove_tms_artifact(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        artifact_cfg = self.config.artifact
        window = artifact_cfg.window_removal_artifact
        mode = artifact_cfg.mode_removal_artifact
        
        events, _ = mne.events_from_annotations(raw)
        tms_trigger_id = list(self.config.events.trigger_id.values())[0]  # → 1
        
        if mode == 'linear':
            raw_artifact_free = raw.copy().load_data()
            
            mne.preprocessing.fix_stim_artifact(
                raw_artifact_free,
                events=events,
                event_id=tms_trigger_id,
                tmin=window[0],
                tmax=window[1],
                mode=mode
            )
        else:
            raise ValueError(f"Unsupported artifact removal mode yet.")
        
        return raw_artifact_free