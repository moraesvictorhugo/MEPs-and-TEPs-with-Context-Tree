from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
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

#def main():
# Settings
config = ProjectConfig(subject_id="V00test")   # used to develop

# Load data
raw_data = load_raw(config)

# Set EOG and EMG channels and set montage
raw_data.set_channel_types({config.channels.eog_label: 'eog', config.channels.emg_label: 'emg'})
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

# Filter epoched data
epochs = EEGFilter(config).bp_filter_epoch(epochs)

# Downsampling
epochs = Downsampler(config).downsample(epochs)

# ___________________________________________________
# TEP plotting (dynamically iterate over all conditions)
for condition in epochs.event_id.keys():
    evoked = epochs[condition].average()
    evoked.plot()
    evoked.plot(picks=['C3', 'FC1', 'CP1'])
    evoked.plot_topomap(times=[0.01, 0.05, 0.1, 0.2], ch_type='eeg')
    evoked.plot_joint(times=[0.05, 0.1, 0.2, 0.3])
    evoked.plot_image(picks='eeg')
    evoked.animate_topomap(times=None, frame_rate=1)
    evoked.plot(gfp=True)

    evoked_crop = evoked.crop(tmin=0.01, tmax=0.05)
    evoked_crop.plot_topo(
        ylim=dict(eeg=[-10, 10]),
        vline=(0.0,),
        title=f'TEPs por canal - {condition}',
        color='blue',
        background_color='white'    
    )




    
# if __name__ == "__main__":
#     main()
