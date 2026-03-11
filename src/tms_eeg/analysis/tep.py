import mne


class TEPAnalyzer:
    def compute_erp(self, epochs: mne.Epochs) -> mne.Evoked: ...
    def extract_components(self, evoked: mne.Evoked) -> dict: ...
    # Components: N15, P30, N45, P60, N100, P180
