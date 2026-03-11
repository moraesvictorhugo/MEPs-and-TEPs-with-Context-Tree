import mne
from src.tms_eeg.config.settings import ProjectConfig


class PreprocessingPipeline:
    def __init__(self, config: ProjectConfig): ...
    def run(self, raw: mne.io.Raw) -> mne.Epochs: ...
