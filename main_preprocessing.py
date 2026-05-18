from scipy.signal import detrend
from pytep import apply_sspsir, apply_sound

# Set backend
from src.tms_eeg.config.environment import setup_plotting_backend
setup_plotting_backend()

# Imports
from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_data
from src.tms_eeg.io.writer import Writer
from src.tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from src.tms_eeg.preprocessing.artifacts import ArtifactRemover
from src.tms_eeg.preprocessing.downsampling import Downsampler
from src.tms_eeg.preprocessing.epoching import EEGEpocher
from src.tms_eeg.preprocessing.filtering import Filter
from src.tms_eeg.preprocessing.ica import EEGICA
from src.tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter


"""
Steps
    Load data
    Find events and create epochs
    Baseline correction (-200 ms to -10 ms)
    Remove TMS Artifact using baseline data (window: -2 - 10 ms) cubic interpolation using 5 ms pre/pos pulse
    Remove bad channels (TP9, TP10, O1, O2, Iz)
    Remove bad trials (noise and blinks)
    Remove drifts without robust detrending
    ICA to remove ocular artifacts
    Baseline correction
    SOUND (lambda = 0.1)
    Rereference to average
    SSP-SIR with tweaked time-window identification and component rejection based on bandpower limits
    Low pass filter (80 Hz) and notch filter (58-62 Hz)
    Remove bad trials (noise)
    Export .mat and .fif file 
    
"""

# Settings
config = ProjectConfig(subject_id="V00")

# Load data
raw_data = load_data(config)

# Set EOG and EMG channels and set montage
raw_data.set_channel_types({
    config.channels.eog_label: 'eog', config.channels.emg_label: 'emg'})
raw_data.set_montage(config.channels.eeg_montage)

# Process annotations to replace Stimulus A with condition labels
annotation_processor = AnnotationProcessor(config)
raw_data = annotation_processor.process_annotations(raw_data)

# Create epochs using standard EEGEpocher
epocher = EEGEpocher(config)
epochs_eeg = epocher.create_epochs(raw_data)

# Create epochs for EMG data
raw_data_emg = raw_data.copy().pick("emg")
epochs_emg = epocher.create_epochs(raw_data_emg)

# Baseline correction
epochs_eeg.apply_baseline(baseline=(-0.2, -0.01))

# Artifact removal
epochs_eeg = ArtifactRemover(config).remove_tms_artifact(epochs_eeg)

# Remove bad channels (TP9, TP10, O1, O2, Iz)
epochs_eeg.drop_channels(["TP9", "TP10", "O1", "O2", "Iz"])

# Remove bad trials (noise and blinks) -> Manual Procedure
epochs_eeg.plot()

# Linear Detrend in each epoch and channel
epochs_eeg.apply_function(lambda x: detrend(x, type='linear'), picks='all')

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs_eeg)
ica_processor.plot_components(epochs_eeg)

# Check and remove eye component
epochs_eeg = ica_processor.apply_ica(epochs_eeg, components_to_remove=[0])
epochs_eeg.plot(block = False)

# Baseline correction
epochs_eeg.apply_baseline(baseline=(-0.5, -0.01))

# Apply SOUND
epochs_eeg = apply_sound(epochs_eeg, iter_num=5, lambda_val=0.1)

# Set average reference
epochs_eeg.set_eeg_reference(config.channels.eeg_reference)

# Apply SSP-SIR
epochs_eeg = apply_sspsir(epochs_eeg)

# Interpolate?

# Filter EEG data
epochs_eeg_filtered = Filter(config).eeg_bp_filter(epochs_eeg)
epochs_eeg_filtered = Filter(config).notch_filter_epochs(epochs_eeg, band=(58, 62))

# Filter EMG data
epochs_emg_filtered = Filter(config).emg_bp_filter(epochs_emg)
epochs_emg_filtered = Filter(config).notch_filter_epochs(epochs_emg, band=(58, 62))

# Remove bad trials (noise) -> Manual Procedure
epochs_eeg_filtered.plot()

# Downsampling
epochs_eeg_filtered = Downsampler(config).downsample(epochs_eeg_filtered)
epochs_emg_filtered = Downsampler(config).downsample_emg_channels(epochs_emg_filtered)

# Get epochs indexes and annotations from EEG and EMG epochs using the exporter
exporter = EpochAnnotationExporter(config)
eeg_epochs_indexes, eeg_epochs_annotations = exporter.extract_annotations(epochs_eeg_filtered)
emg_epochs_indexes, emg_epochs_annotations = exporter.extract_annotations(epochs_emg_filtered)

# Export to .mat for ContextTree analysis (MATLAB) using the exporter
symbols = exporter.map_annotations_to_symbols(eeg_epochs_annotations)
writer = Writer(config)
exporter.export_to_mat(writer, epochs_eeg_filtered, symbols)

# Export processed data
writer.save_emg_epochs(epochs_emg_filtered, 'emg_processed')

# Export epochs
writer.save_epochs(epochs_eeg_filtered, 'processed')