from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
from src.tms_eeg.io.writer import EEGWriter
from src.tms_eeg.preprocessing.epoching import EEGEpocher
from src.tms_eeg.preprocessing.artifacts import ArtifactRemover
from src.tms_eeg.preprocessing.filtering import EEGFilter
from src.tms_eeg.preprocessing.ica import EEGICA
from src.tms_eeg.preprocessing.downsampling import Downsampler
from src.tms_eeg.preprocessing.annotation_processor import AnnotationProcessor


# temp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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
config = ProjectConfig(subject_id="V00test")   # used to develop

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
filtered_data = EEGFilter(config).bp_filter(raw_data)
filtered_data = EEGFilter(config).notch_filter(filtered_data)

# NEW: Process annotations to replace Stimulus A with condition labels
annotation_processor = AnnotationProcessor(config)
processed_data = annotation_processor.process_annotations(filtered_data)

# Create epochs using standard EEGEpocher (now with condition labels!)
epochs = epocher.create_epochs(processed_data)

# Set average reference
epochs.set_eeg_reference(config.channels.eeg_reference)

# Check and remove bad channels
epochs.plot()

# Interpolate bad channels
epochs = epochs.interpolate_bads(reset_bads=True)

# Check and remove bad epochs
epochs.plot()

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs)
ica_processor.plot_components(epochs)
epochs = ica_processor.apply_ica(epochs, components_to_remove=[0, 1, 8, 10, 15])
epochs.plot(block = False)

# Baseline correction
epochs.apply_baseline(baseline=(-0.5, -0.01))

# Filter epoched data
epochs = EEGFilter(config).bp_filter_epoch(epochs)

# Downsampling
epochs = Downsampler(config).downsample(epochs)

# Baseline correction
epochs.apply_baseline(baseline=(-0.5, -0.01))

# Individual TEP plotting for validation
for condition in epochs.event_id.keys():
    evoked = epochs[condition].average()
    evoked.comment = f'TEPs for {condition}'
    title = f'TEPs - {condition}'
    
    # Add condition label to the evoked object
    evoked.plot(xlim=(-0.02, 0.2))
    evoked.plot(
        picks=['C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'],
        titles=f'TEPs for {condition}',
        xlim=(-0.02, 0.2)
    )
    evoked.plot_joint(
        times=[0.015, 0.03, 0.045, 0.06, 0.1, 0.18],
        title=title,
        ts_args=dict(xlim=(-0.02, 0.2))
    )
    evoked.plot_image(picks='eeg', xlim=(-0.02, 0.2), show_names='all')
    evoked.copy().crop(tmin=-0.02, tmax=0.2).animate_topomap(
        times=[0.015, 0.03, 0.045, 0.06, 0.1, 0.18], frame_rate=1)    
    evoked.plot(gfp=True, xlim=(-0.02, 0.2))
    evoked.plot_topomap(
    times=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    colorbar=True
    )
    evoked_crop = evoked.crop(tmin=0.02, tmax=0.2)
    evoked_crop.plot_topo(
        ylim=dict(eeg=[-10, 10]),
        vline=(0.0,),
        title=f'TEPs por canal - {condition}',
        color='blue',
        background_color='white'    
    )

# Export epochs
writer = EEGWriter(config)
writer.save_epochs(epochs, 'processed')

# Export average evoked
writer.save_evoked_conditions(epochs, 'processed')