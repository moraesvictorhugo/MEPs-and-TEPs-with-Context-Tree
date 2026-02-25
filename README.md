# MEPs-and-TEPs-with-Context-Tree
 Created to store Python codes used to PD-NEUROMAT

# Project Structure

tms_eeg_analysis/
├── data/
│   ├── raw/                  # Original, unmodified EEG files
│   └── processed/            # Output from preprocessing/processing steps
│
├── src/
│   └── tms_eeg/
│       ├── __init__.py
│       ├── config/           # Configuration files
│       ├── io/               # File loading and saving
│       ├── preprocessing/    # Filtering, artifact removal, ICA, etc.
│       ├── analysis/         # TEP extraction, TMS-related analysis
│       └── visualization/    # All visualization functions
│
├── notebooks/                # Jupyter notebooks for exploration
│
├── outputs/
│   └── figures/              # Saved plots and reports
│
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py # One test file per module (as needed)
│
├── main.py                   # Entry point / pipeline runner
├── pyproject.toml            # Project metadata and dependencies
└── README.md

# Overal data flow

Raw File
   │
   ▼
EEGReader.load_raw()
   │
   ▼
PreprocessingPipeline.run()
   ├── ArtifactRemover.remove_tms_pulse()
   ├── EEGFilter.bandpass() + notch()
   ├── ArtifactRemover.run_ica() / apply_ica()
   └── EEGEpocher.create_epochs() + apply_baseline()
   │
   ▼
mne.Epochs
   ├──► TEPAnalyzer.compute_erp() ──► TEPPlotter
   └──► TFRAnalyzer.compute_tfr() ──► TEPPlotter
   |
   ▼
Export data (figures and csv files)