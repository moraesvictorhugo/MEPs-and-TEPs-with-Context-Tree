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
│       ├── io.py             # File loading and saving
│       ├── preprocessing.py  # Filtering, artifact removal, ICA, etc.
│       ├── processing.py     # TEP extraction, TMS-related analysis
│       └── plotting.py       # All visualization functions
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
