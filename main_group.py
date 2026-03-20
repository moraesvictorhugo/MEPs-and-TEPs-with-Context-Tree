"""Group-level visualization for TMS-EEG metrics."""

import pandas as pd
from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.visualization.group_plots import GroupPlotter

config = ProjectConfig()
df = pd.read_csv("data/group/database.csv")

plotter = GroupPlotter(config=config)

# Loop pelos canais de interesse
for ch in config.analysis.channels_of_interest:
    plotter.plot_boxplots(df, groupby="condition", channel=ch, component="N15-P30")
    plotter.plot_boxplots(df, groupby="context", channel=ch, component="N15-P30")
    plotter.plot_boxplots(df, groupby="condition", channel=ch, component="N15-P60")
    plotter.plot_boxplots(df, groupby="context", channel=ch, component="N15-P60")
    plotter.plot_boxplots(df, groupby="condition", channel=ch, component="N100-P180")
    plotter.plot_boxplots(df, groupby="context", channel=ch, component="N100-P180")
    
# Boxplots for GMFP and LMFP
plotter.plot_p30_amplitude(df)
