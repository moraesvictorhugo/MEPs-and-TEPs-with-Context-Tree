"""Main analysis pipeline for TMS-EEG data."""

# Set backend
# from src.tms_eeg.config.environment import setup_plotting_backend
# setup_plotting_backend()

from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_data
from src.tms_eeg.io.writer import Writer
from src.tms_eeg.analysis.features import FeatureExtractor
from src.tms_eeg.visualization.tep_plots import TEPPlotter
from src.tms_eeg.visualization.gfp_plots import MFPPlotter

all_results = []
subjects = ProjectConfig().analysis.subjects

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")
    writer = Writer(config)

    # ── Feature Extractor ────────────────────────────────────────────
    extractor = FeatureExtractor(
        epochs,
        config.analysis.channels_of_interest,
        config.analysis.time_windows,
    )

    # ── Evokeds (ROI) ───────────────────────────────────────────────
    evokeds = extractor.get_evokeds()

    # ── TEP plots ────────────────────────────────────────────────────
    tep_plotter = TEPPlotter(config=config, writer=writer)
    tep_plotter.plot_mean_tep(evokeds=evokeds)

    # ── Peak-to-Peak calculation ─────────────────────────────────────
    amplitude_N15_P30 = extractor.peak_to_peak("N15", "P30", evokeds=evokeds)
    amplitude_N100_P180 = extractor.peak_to_peak("N100", "P180", evokeds=evokeds)

    # ── GMFP & LMFP calculation ──────────────────────────────────────
    gmfp = extractor.compute_gmfp()
    lmfp = extractor.compute_lmfp()

    # ── MFP peak extraction ──────────────────────────────────────────
    df_gmfp_peaks = extractor.extract_mfp_peaks(gmfp, label="GMFP")
    df_lmfp_peaks = extractor.extract_mfp_peaks(lmfp, label="LMFP")

    # ── Full summary (optional — runs everything at once) ────────────
    # df_gmfp, df_lmfp, df_p2p = extractor.compute_summary()

    # ── MFP Plots ────────────────────────────────────────────────────
    mfp_plotter = MFPPlotter(times=epochs.times, config=config, writer=writer)

    # Side-by-side GMFP vs LMFP per condition
    mfp_plotter.plot_gmfp_lmfp(
        gmfp, lmfp,
        time_windows=config.analysis.time_windows,
    )

    # Overlay all conditions (one plot for GMFP, one for LMFP)
    mfp_plotter.plot_overlay(gmfp, label="GMFP", time_windows=config.analysis.time_windows)
    mfp_plotter.plot_overlay(lmfp, label="LMFP", time_windows=config.analysis.time_windows)

    # ── Collect results ──────────────────────────────────────────────
    all_results.append({
        "subject": subject_id,
        "amplitude_N15_P30": amplitude_N15_P30,
        "amplitude_N100_P180": amplitude_N100_P180,
        "gmfp_peaks": df_gmfp_peaks,
        "lmfp_peaks": df_lmfp_peaks,
    })


group_df = aggregate_subjects(all_results)
save_csv(group_df, path="outputs/stats/group_features.csv")

fig_group = plot_group_comparison(group_df)
save_figure(fig_group, path="outputs/figures/group_comparison.png")

print("\n✅ Análise concluída!")


# if __name__ == "__main__":
#     main()
