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

'''
Steps
    Load data                                                                                    
    Find/create events                                                                           
    Baseline correction (-200 ms to -10 ms)
    Remove TMS Artifact using baseline data (window: -2 - 5 ms) applying 0 padding
    Identify bad channels (max or min amplitude in baseline window exceeds median ± 10× IQR)
    Reject blinking trials (blink topography of ICA)
    SOUND (lambda = 0.1)
    SSP-SIR with tweaked time-window identification and component rejection based on bandpower limits
    Interpolate TMS pulse artifact time window (-2, 10 ms)
    Notch filter (56-64 Hz) and bandpass filter (1-100 Hz)
    Baseline correction
    Average reference
    Reject trial if initial number of epochs exceed n=20
'''

'''
To-Do for context tree retrieving:
preprocessed data as .mat file (V01.mat) with a data struct with these fields:
V01.mat
└── data (struct)
    ├── X_ter  →  [1 × N_ter]   sequência de símbolos ∈ {0, 1, 2}
    │
    └── Y_ter  →  {2 × E cell}  EEG segmentado
                  ├── linha 1: nomes dos eletrodos  ('Fz', 'Cz', ...)
                  └── linha 2: matrizes D × N_ter   (uma por eletrodo)
                  
Reformatar para o "molde" do .mat original               
    - X_ter: array (1, N) float64                            
    - Y_ter: array (2, E) object                             
        ├── linha 0: strings dos eletrodos                   
        └── linha 1: matrizes (D, N) float64 -> colunas são épocas e linhas são pontos no tempo em microVolts              
    - Empacotar tudo num dict: {'data': {...}}            
Salvar com scipy.io.savemat('V01.mat', {'data': ...})

Exemplo em: /home/victomoraes/Documents/GitHub/EEG_Retrieving/statistical_analysis/EEGretrieving_pre/preprocessed_data/V01.mat

- EEG epochs (15 ms to 415 ms) -> adapted to cut TMS artifact at 250 Hz of sampling rate
'''


# Settings
config = ProjectConfig(subject_id="V00")

# Load data
raw_data = load_raw(config)

# Set EOG and EMG channels and set montage
raw_data.set_channel_types({
    config.channels.eog_label: 'eog', config.channels.emg_label: 'emg'})
raw_data.set_montage(config.channels.eeg_montage)

# # Drop unused channels
# raw_data.drop_channels(config.channels.bad_channels)

# Artifact removal -> só criação de eventos para baseline
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data)

# Filter raw EEG data
filtered_data = Filter(config).eeg_bp_filter(raw_data)
filtered_data = Filter(config).emg_bp_filter(filtered_data)
filtered_data = Filter(config).notch_filter(filtered_data)

# Process annotations to replace Stimulus A with condition labels
annotation_processor = AnnotationProcessor(config)
filtered_data = annotation_processor.process_annotations(filtered_data)

# Create epochs using standard EEGEpocher
epocher = EEGEpocher(config)
epochs = epocher.create_epochs(filtered_data)

# Apply SOUND
try:
    import sys
    import os
    
    # Adiciona o caminho do diretório PyTEP-SOUND-SSP-SIR ao Python path
    sound_dir = os.path.join(os.path.dirname(__file__), 'PyTEP-SOUND-SSP-SIR')
    sys.path.append(sound_dir)
    
    # Importa o módulo sound
    from sound import apply_sound
    
    print("Aplicando SOUND para remoção de artefatos...")
    epochs = apply_sound(epochs, iter_num=5, lambda_val=0.1)
    print("SOUND aplicado com sucesso!")
    
except ImportError as e:
    print(f"Erro ao importar SOUND: {e}")
    print("Certifique-se de que o diretório PyTEP-SOUND-SSP-SIR está presente.")
    print("Continuando sem SOUND...")
except Exception as e:
    print(f"Erro ao aplicar SOUND: {e}")
    print("Continuando sem SOUND...")

# Writer initialization
writer = Writer(config)

# Downsampling EMG and export epochs of emg
emg_epochs = Downsampler(config).downsample_emg_channels(epochs)
writer.save_emg_epochs(emg_epochs, 'emg_processed')

# Check and remove bad channels -> Manual Action Required
epochs.plot()

# Interpolate bad channels
epochs = epochs.interpolate_bads(reset_bads=True)

# Set average reference
epochs.set_eeg_reference(config.channels.eeg_reference)

# Check and remove bad epochs
epochs.plot()

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs)
ica_processor.plot_components(epochs)
epochs = ica_processor.apply_ica(epochs, components_to_remove=[0])
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