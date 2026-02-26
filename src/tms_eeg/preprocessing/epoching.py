from src.tms_eeg.config.settings import ProjectConfig
import mne

class EEGEpocher:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def create_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        """Create epochs from raw data.

        Args:
            raw (mne.io.BaseRaw): Preprocessed raw EEG data.

        Returns:
            mne.Epochs: Epoched EEG data.
        """
        epoch_cfg = self.config.epochs
        event_cfg = self.config.events

        # Extract events from annotations
        events, event_id = mne.events_from_annotations(
            raw,
            event_id=event_cfg.trigger_id,  # type: ignore
        )

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

        # Downsample if configured
        if epoch_cfg.downsample:
            epochs.resample(sfreq=epoch_cfg.downsample_freq)

        return epochs
