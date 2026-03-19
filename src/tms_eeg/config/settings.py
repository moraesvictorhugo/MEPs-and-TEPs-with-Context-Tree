from dataclasses import dataclass, field
# field(default_factory=...) must be used for mutable types like lists
# or dicts to avoid shared state across instances

@dataclass
class IOConfig:
    export_data: bool = False
    change_channel_types: bool = False
    new_channel_names: dict = field(default_factory=dict)  # {"EEG 001": "Fp1"}
    save_figs: bool = False

@dataclass
class EventConfig:
    event_ids: dict = field(default_factory=lambda: {
        'stimulus_0': 0,
        'stimulus_1': 1,
        'stimulus_2': 2,
    })
    trigger_id: dict = field(default_factory=lambda: {'Stimulus A': 1})
    trigger_8bits: tuple = (1, 2, 3)
    stimulus_to_8bit_mapping: dict = field(default_factory=lambda: {
        'Stimulus A': ['8bit 1', '8bit 2', '8bit 3'],
    })
    use_8bit_triggers: bool = False  # New field to control behavior
    text_file_pattern: str = "*.txt"  # Pattern to find condition files

@dataclass
class ArtifactConfig:
    window_removal_artifact: tuple = (-0.005, 0.012)
    mode_removal_artifact: str = 'linear' # 'cubic'

@dataclass
class FilterConfig:
    eeg_bandpass: tuple = (1, 250)
    emg_bandpass: tuple = (20, 500)
    notch: tuple = (60, 120, 180, 240, 300)
    eeg_filter_epoched_data: bool = True
    eeg_bandpass_epochs: tuple = (None, 45)

@dataclass
class ChannelConfig:
    eeg_reference: str = 'average'
    remove_bad_channels: bool = True
    bad_channels: tuple = ()
    eog_label: str = 'EOG'
    emg_label: str = 'EMG'
    eeg_montage: str = 'standard_1020'

@dataclass
class EpochConfig:
    window: tuple = (-0.8, 0.8)
    baseline: tuple = (-0.5, -0.01)  # ou None para desativar
    downsample: bool = True
    downsample_freq: float = 725.0
    emg_downsample_freq: float = 3000.0
    rejection_amplitude_threshold: dict = field(default_factory=lambda: {
        'eeg': 1000e-6, 'eog': 5000e-6})               # 50 and 100 microvolts
    rejection_flat_threshold: dict = field(
        default_factory=lambda: {'eeg': 1e-6})      # 1 microvolts

@dataclass
class ICAConfig:
    run_ica: bool = True
    plot_components: bool = True
    use_ica_label: bool = False

@dataclass
class AnalysisConfig:
    subjects: list = field(default_factory=lambda: [
        "V02", "V03"])
    channels_of_interest: list = field(default_factory=lambda: [
        "FC1", "FC5", "C3", "CP1", "CP5"])
    time_windows: dict = field(default_factory=lambda: {
        "N15":  (0.012, 0.020),
        "P30":  (0.020, 0.040),
        "N45":  (0.040, 0.055),
        "P60":  (0.050, 0.070),
        "N100": (0.070, 0.150),
        "P180": (0.150, 0.200), # Defined following Beck et al (2024)
    })
    # GMFP / LMFP
    compute_gmfp: bool = True
    compute_lmfp: bool = True
    # Features
    calculate_p2p_amplitude: bool = True
    # Context analysis
    context_definitions: dict = field(default_factory=lambda: {
        "ctx_0":  [0],        # atual=0, qualquer passado
        "ctx_2":  [2],        # atual=2, qualquer passado
        "ctx_01": [0, 1],     # anterior=0, atual=1
        "ctx_11": [1, 1],     # anterior=1, atual=1
        "ctx_21": [2, 1],     # anterior=2, atual=1
    })
    # Mapping: event code (from find_events) → context symbol
    event_to_symbol: dict = field(default_factory=lambda: {
        1: 0,   # 8Bit 1 → símbolo 0 (80% rMT)
        2: 1,   # 8Bit 2 → símbolo 1 (100% rMT)
        3: 2,   # 8Bit 3 → símbolo 2 (120% rMT)
    })

@dataclass
class PlotConfig:
    plot_raw: bool = True
    plot_raw_psd: bool = True
    plot_epochs: bool = True
    plot_teps_by_stimulus: bool = True
    plot_ica_components: bool = True
    # Figure saving options
    figure_format: str = "png"
    figure_dpi: int = 300
    figure_subfolder: str = "figures"
    # TEP plotting options
    tep_xlim: tuple = (-0.01, 0.2)
    tep_topo_times: list = field(default_factory=lambda: [
        0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.1
    ])
    tep_joint_times: list = field(default_factory=lambda: [
        0.015, 0.03, 0.045, 0.06, 0.1, 0.18
    ])
    tep_roi_channels: list = field(default_factory=lambda: [
        'C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'
    ])
    # EMG plotting options
    emg_xlim: tuple = (-0.01, 0.08)
    analysis_plots: bool = False

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
