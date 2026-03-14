# Set backend
from src.tms_eeg.config.environment import setup_plotting_backend
setup_plotting_backend()

# Imports
from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
from src.tms_eeg.io.writer import Writer
from src.tms_eeg.preprocessing.epoching import EEGEpocher
from src.tms_eeg.preprocessing.artifacts import ArtifactRemover
from src.tms_eeg.preprocessing.filtering import Filter
from src.tms_eeg.preprocessing.ica import EEGICA
from src.tms_eeg.preprocessing.downsampling import Downsampler
from src.tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from src.tms_eeg.visualization.tep_plots import TEPPlotter
from src.tms_eeg.visualization.emg_plots import EMGPlotter

'''
Steps
    Load data                                                                                    
    Find/create events                                                                           
    Drop unused channels (e.g., EMG)                                                             
    Remove TMS artifact using baseline data (window: -2 - 5ms)                                   
    Filter raw EEG data (high-pass 1 Hz, low-pass: 250 Hz and notch filter 60 Hz)
    Create epochs (-0.8 to 0.8)
    Average reference
    Remove bad channels (manual or threshold=3)
    Remove bad epochs (manual or threshold=3)
    First ICA (FastICA)
    (Optional and very experimental) PARAFAC decomposition
    (Optional) Second ICA (Infomax)
    (Optional) SSP
    Filter epoched data (low-pass 45 Hz)
    Downsampling (725 Hz)
    TEP plotting
    PCIst
'''

# Settings
config = ProjectConfig(subject_id="V00test")

# Load data
raw_data = load_raw(config)

# Set EOG and EMG channels and set montage
raw_data.set_channel_types({
    config.channels.eog_label: 'eog', config.channels.emg_label: 'emg'})
raw_data.set_montage(config.channels.eeg_montage)

# Find events / create epochs
epocher = EEGEpocher(config)

# Drop unused channels
raw_data.drop_channels(config.channels.bad_channels)

# Artifact removal
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data)

# Filter raw EEG data
filtered_data = Filter(config).eeg_bp_filter(raw_data)
filtered_data = Filter(config).emg_bp_filter(filtered_data)
filtered_data = Filter(config).notch_filter(filtered_data)

# if config.subject_id == "V00test":
#     import mne
    
#     filtered_data.set_annotations(mne.Annotations([], [], []))
#     # Exemplo: distribui as annotations ao longo do sinal
#     duration_sec = filtered_data.times[-1]
#     print(f"Duração total: {duration_sec:.2f} s")

#     # Defina onsets válidos (menores que duration_sec)
#     my_annotations = mne.Annotations(
#         onset=[10, 10.1, 20, 20.1, 30, 30.1],   # ← ajuste para valores reais do seu experimento
#         duration=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#         description=["8Bit 1", "Stimulus A", "8Bit 2", "Stimulus A", "8Bit 3", "Stimulus A"],
#         orig_time=None
#     )

#     filtered_data.set_annotations(my_annotations)
#     print(filtered_data.annotations)

# Process annotations to replace Stimulus A with condition labels
annotation_processor = AnnotationProcessor(config)
filtered_data = annotation_processor.process_annotations(filtered_data)

# Create epochs using standard EEGEpocher
epochs = epocher.create_epochs(filtered_data)

# Writer initialization
writer = Writer(config)

# Downsampling EMG and export epochs of emg
emg_epochs = Downsampler(config).downsample_emg_channels(epochs)
writer.save_emg_epochs(emg_epochs, 'emg_processed')

# Set average reference
epochs.set_eeg_reference(config.channels.eeg_reference)

# Check and remove bad channels -> Manual Action Required
epochs.plot()

# Interpolate bad channels
epochs = epochs.interpolate_bads(reset_bads=True)

# Check and remove bad epochs
epochs.plot()

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs)
ica_processor.plot_components(epochs)
epochs = ica_processor.apply_ica(epochs, components_to_remove=[0, 3, 15, 18])
epochs.plot(block = False)

# Baseline correction
epochs.apply_baseline(baseline=(-0.5, -0.01))

# Filter epoched data
epochs = Filter(config).eeg_bp_filter_epoch(epochs)

# Downsampling
epochs = Downsampler(config).downsample(epochs)

# Baseline correction
epochs.apply_baseline(baseline=(-0.5, -0.01))

# TEP plots
tep_plotter = TEPPlotter(config=config, writer=writer)
tep_plotter.plot_all(epochs)

# Export processed data
writer.save_raw(filtered_data)

# Export epochs
writer.save_epochs(epochs, 'processed')

# Export average evoked
writer.save_evoked_conditions(epochs, 'processed')
