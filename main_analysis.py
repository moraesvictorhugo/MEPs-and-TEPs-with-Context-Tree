"""Main analysis pipeline for TMS-EEG data."""

# Set backend
# from src.tms_eeg.config.environment import setup_plotting_backend
# setup_plotting_backend()

from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_data
from src.tms_eeg.io.writer import Writer
from src.tms_eeg.analysis.features import FeatureExtractor
from src.tms_eeg.visualization.tep_plots import TEPPlotter
# from src.tms_eeg.analysis.gfp import compute_gfp
# from src.tms_eeg.analysis.lmfp import compute_lmfp
# from src.tms_eeg.analysis.group import aggregate_subjects
# from src.tms_eeg.visualization.tep_plots import plot_tep
# from src.tms_eeg.visualization.gfp_plots import plot_gfp
# from src.tms_eeg.visualization.group_plots import plot_group_comparison

# def main():
all_results = []
subjects = ProjectConfig().analysis.subjects

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")

    # Extract features object
    extractor = FeatureExtractor(
        epochs, config.analysis.channels_of_interest,
        config.analysis.time_windows
    )

    # 1. Evokeds por condição
    evokeds = extractor.get_evokeds()

    # 2. Plot TEP (adaptar plotter para receber Dict[str, Evoked])
    writer = Writer(config)
    tep_plotter = TEPPlotter(config=config, writer=writer)
    tep_plotter.plot_mean_tep(evokeds=evokeds)

    # 3. Amplitude P2P sobre o TEP médio
    amplitude_N15_P30 = extractor.peak_to_peak("N15", "P30", evokeds=evokeds)

    # Amplitude P2P N100-P180 sobre o TEP médio
    amplitude_N100_P180 = extractor.peak_to_peak("N100", "P180", evokeds=evokeds)
        
    # Get GMFP _ FIX
    gmfp = extractor.compute_gmfp(evokeds=evokeds)
    
    
    gfp = compute_gfp(epochs)
    lmfp = compute_lmfp(epochs, config.analysis.channels_of_interest)

    writer = Writer(config)
    writer.save_csv(features, filename="peak_to_peak.csv")
    writer.save_csv(gfp, filename="gfp.csv")
    writer.save_csv(lmfp, filename="lmfp.csv")

    fig_tep = plot_tep(epochs, config.analysis.channels_of_interest)
    fig_gfp = plot_gfp(gfp)
    save_figure(fig_tep, config, filename="tep_butterfly.png")
    save_figure(fig_gfp, config, filename="gfp.png")

    all_results.append({
        "subject_id": subject_id,
        "features": features,
        "gfp": gfp,
        "lmfp": lmfp,
    })

group_df = aggregate_subjects(all_results)
save_csv(group_df, path="outputs/stats/group_features.csv")

fig_group = plot_group_comparison(group_df)
save_figure(fig_group, path="outputs/figures/group_comparison.png")

print("\n✅ Análise concluída!")


# if __name__ == "__main__":
#     main()
