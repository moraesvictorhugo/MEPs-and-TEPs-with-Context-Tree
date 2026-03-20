"""Group-level visualization for TMS-EEG metrics."""

import pandas as pd
from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.visualization.group_plots import GroupPlotter

config = ProjectConfig()
df = pd.read_csv("data/group/all_subjects_metrics.csv")

plotter = GroupPlotter(config=config)

# Boxplots por condição
plotter.plot_boxplots(df, groupby="condition")

# Boxplots por contexto
plotter.plot_boxplots(df, groupby="context")
