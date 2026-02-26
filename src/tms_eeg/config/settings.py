from dataclasses import dataclass, field
# field(default_factory=...) must be used for mutable types like lists
# or dicts to avoid shared state across instances

@dataclass
class IOConfig:
    export_data: bool = False
    change_channel_types: bool = False
    new_channel_names: dict = field(default_factory=dict)  # {"EEG 001": "Fp1"}

@dataclass
class EventConfig:
    event_ids: dict = field(default_factory=lambda: {
        'stimulus_0': 0,
        'stimulus_1': 1,
        'stimulus_2': 2,
    })
    trigger_id: dict = field(default_factory=lambda: {'Stimulus A': 1})
    trigger_8bits: tuple = (1, 2, 3)

@dataclass
class ArtifactConfig:
    window_removal_artifact: tuple = (-0.010, 0.008)
    mode_removal_artifact: str = 'linear'

@dataclass
class FilterConfig:
    bandpass: tuple = (1, 250)
    notch: tuple = (60, 120, 180, 240, 300)
    filter_epoched_data: bool = True
    bandpass_epochs: tuple = (None, 45)

@dataclass
class ChannelConfig:
    eeg_reference: str = 'average'
    remove_bad_channels: bool = True
    bad_channels: tuple = ('TP9', 'TP10', 'Oz', 'O1', 'O2')

@dataclass
class EpochConfig:
    window: tuple = (-0.8, 0.8)
    downsample: bool = True
    downsample_freq: float = 725.0

@dataclass
class ICAConfig:
    run_ica: bool = True
    plot_components: bool = True
    use_ica_label: bool = True
    components_to_remove: list = field(default_factory=list)  # user fills in

@dataclass
class AnalysisConfig:
    compute_lmfa: bool = True
    lmfa_channels: list = field(default_factory=list)  # empty = all channels
    compute_gmfa: bool = True
    gmfa_channels: list = field(default_factory=list)  # empty = all channels
    calculate_p2p_amplitude: bool = True

@dataclass
class PlotConfig:
    plot_raw: bool = True
    plot_raw_psd: bool = True
    plot_epochs: bool = True
    plot_teps_by_stimulus: bool = True
    plot_ica_components: bool = True

@dataclass
class ProjectConfig:
    io: IOConfig = field(default_factory=IOConfig)
    events: EventConfig = field(default_factory=EventConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    subject_id: str = ""
    session: str = "_400pulses_TMS-EEG_ContextTree"
