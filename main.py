from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
from src.tms_eeg.preprocessing.epoching import EEGEpocher
from src.tms_eeg.preprocessing.artifacts import ArtifactRemover
from src.tms_eeg.preprocessing.filtering import EEGFilter
from src.tms_eeg.preprocessing.ica import EEGICA

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
raw_data = load_raw(config)  # should use load_raw instead

# Find events / create epochs
epocher = EEGEpocher(config)

# Drop unused channels
raw_data.drop_channels(config.channels.bad_channels)

# Artifact removal  ->>> not working properly (use event_id number)
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data) 

# # Filter raw EEG data
filtered_data = EEGFilter(config).bp_filter(raw_data)

# Create epochs
epochs = epocher.create_epochs(filtered_data)

# Set average reference
epochs.set_eeg_reference(config.channels.eeg_reference)

# Remove bad and epochs (manual or threshold=3) -> not working properly
# epochs = EEGEpocher.reject_bad(
#     epochs,
#     thresholds=config.epochs.rejection_amplitude_threshold,
#     flat_thresholds=config.epochs.rejection_flat_threshold
#     )

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs)
epochs = ica_processor.apply_ica(epochs)

# Optional: Plot components for manual inspection
if config.ica.plot_components:
    ica_processor.plot_components(epochs)





# # Access
# config.filters.bandpass        # [1, 250]
# config.ica.run_ica             # True
# config.channels.bad_channels   # ['TP9', 'TP10', 'Oz', 'O1', 'O2']

# # Override one value
# config.epochs.downsample_freq = 500.0
    
# if __name__ == "__main__":
#     main()
