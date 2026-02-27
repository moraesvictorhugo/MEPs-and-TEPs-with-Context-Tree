import mne
from src.tms_eeg.config.settings import ProjectConfig


class EEGEpocher:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def find_events(self, raw: mne.io.BaseRaw) -> tuple[mne.BaseEpochs, dict]:
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
     
     
     
     
