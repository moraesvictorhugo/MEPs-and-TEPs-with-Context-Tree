from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_raw
from src.tms_eeg.preprocessing.epoching import EEGEpocher

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
raw = load_raw(config)  # should use load_raw instead

# Find/create events
epocher = EEGEpocher(config)
epochs = epocher.create_epochs(raw)

# Drop unused channels
epochs.drop_channels(config.channels.bad_channels)

# Artifact removal







# Access
config.filters.bandpass        # [1, 250]
config.ica.run_ica             # True
config.channels.bad_channels   # ['TP9', 'TP10', 'Oz', 'O1', 'O2']

# Override one value
config.epochs.downsample_freq = 500.0
