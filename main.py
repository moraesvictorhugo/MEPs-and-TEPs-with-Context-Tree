from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
from src.tms_eeg.preprocessing.epoching import EEGEpocher
from src.tms_eeg.preprocessing.artifacts import ArtifactRemover
from src.tms_eeg.preprocessing.filtering import EEGFilter
from src.tms_eeg.preprocessing.ica import EEGICA
from src.tms_eeg.preprocessing.downsampling import Downsampler
from src.tms_eeg.preprocessing.epoching import DualEventEpocher, process_8bit_epochs, analyze_8bit_triggers


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

# Create epochs
epocher = DualEventEpocher(config)
epochs_by_trigger = epocher.create_epochs_with_8bit_grouping(filtered_data)

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

# NEW: Use utility functions instead of for loop
processed_epochs = process_8bit_epochs(epochs_by_trigger, config)
analyze_8bit_triggers(processed_epochs, config)

# ___________________________________________________
# TEP plotting
evoked_stim1 = epochs['Stimulus A'].average()
evoked_stim1.plot()

evoked_stim1.plot(picks=['Cz', 'Fz', 'Pz'])

# Plot topomap at specific time points
evoked_stim1.plot_topomap(times=[0.01, 0.05, 0.1, 0.2], ch_type='eeg')

# Plot joint plot at specific time points
evoked_stim1.plot_joint(times=[0.05, 0.1, 0.2, 0.3])

# Heatmap at specific time points
evoked_stim1.plot_image(picks='eeg')

# Animate topomap at specific time points
evoked_stim1.animate_topomap(times=None, frame_rate=1)

# GFP plot
evoked_stim1.plot(gfp=True)

# Plot topoplot and traces
evoked_crop = evoked_stim1.crop(tmin=0.01, tmax=0.05)
evoked_crop.plot_topo(
    ylim=dict(eeg=[-10, 10]),
    vline=(0.0,),
    title='TEPs por canal',
    color='blue',
    background_color='white'    
)




    
# if __name__ == "__main__":
#     main()
