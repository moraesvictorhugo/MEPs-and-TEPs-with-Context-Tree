"""Main analysis pipeline for TMS-EEG data."""

# Set backend
from src.tms_eeg.config.environment import setup_plotting_backend
setup_plotting_backend()

from src.tms_eeg.config.settings import ProjectConfig
from src.tms_eeg.io.reader import load_data, get_raw_path
from src.tms_eeg.io.writer import Writer
from src.tms_eeg.analysis.features import FeatureExtractor
from src.tms_eeg.analysis.context import ContextMapper
from src.tms_eeg.visualization.tep_plots import TEPPlotter
from src.tms_eeg.visualization.gfp_plots import MFPPlotter

all_results = []
subjects = ProjectConfig().analysis.subjects

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")
    writer = Writer(config)

    # ── Shared objects ───────────────────────────────────────────────
    extractor = FeatureExtractor(
        epochs,
        config.analysis.channels_of_interest,
        config.analysis.time_windows,
    )
    tep_plotter = TEPPlotter(config=config, writer=writer)
    mfp_plotter = MFPPlotter(times=epochs.times, config=config, writer=writer)

    # ================================================================ #
    #  PART 1 — ANÁLISE POR CONDIÇÃO (8Bit 1 / 2 / 3)
    # ================================================================ #

    # ── Evokeds (ROI) ───────────────────────────────────────────────
    evokeds = extractor.get_evokeds()

    # ── TEP plots ────────────────────────────────────────────────────
    tep_plotter.plot_mean_tep(evokeds=evokeds)

    # ── Peak-to-Peak calculation ─────────────────────────────────────
    amplitude_N15_P30 = extractor.peak_to_peak("N15", "P30", evokeds=evokeds)
    amplitude_N100_P180 = extractor.peak_to_peak(
        "N100", "P180", evokeds=evokeds)

    # ── GMFP & LMFP calculation ─────────────────────────────────────
    gmfp = extractor.compute_gmfp()
    lmfp = extractor.compute_lmfp()

    # ── MFP peak extraction ──────────────────────────────────────────
    df_gmfp_peaks = extractor.extract_mfp_peaks(gmfp, label="GMFP")
    df_lmfp_peaks = extractor.extract_mfp_peaks(lmfp, label="LMFP")
   
    # ── MFP Plots ────────────────────────────────────────────────────
    # Side-by-side GMFP vs LMFP per condition
    mfp_plotter.plot_gmfp_lmfp(
        gmfp, lmfp,
        time_windows=config.analysis.time_windows,
    )

    # Overlay all conditions (one plot for GMFP, one for LMFP)
    mfp_plotter.plot_overlay(
        gmfp, label="GMFP", time_windows=config.analysis.time_windows)
    mfp_plotter.plot_overlay(
        lmfp, label="LMFP", time_windows=config.analysis.time_windows)

    # ================================================================ #
    #  PART 2 — ANÁLISE POR CONTEXTO (árvore de contexto)
    # ================================================================ #

    raw_path = get_raw_path(config)
    context_mapper = ContextMapper(config)
    context_epochs = context_mapper.get_context_epochs(epochs, raw_path)

    # ── Evokeds por contexto (ROI) ──────────────────────────────────
    ctx_evokeds = {
        ctx_name: ctx_ep.average().pick(config.analysis.channels_of_interest)
        for ctx_name, ctx_ep in context_epochs.items()
    }

    # ── TEP plots por contexto ───────────────────────────────────────
    tep_plotter.plot_mean_tep(evokeds=ctx_evokeds)

    # ── GMFP & LMFP por contexto ────────────────────────────────────
    ctx_gmfp = {
        ctx_name: ctx_ep.average().pick("eeg").data.std(axis=0)
        for ctx_name, ctx_ep in context_epochs.items()
    }
    ctx_lmfp = {
        ctx_name: ctx_ep.average().pick(
            config.analysis.channels_of_interest
        ).data.std(axis=0)
        for ctx_name, ctx_ep in context_epochs.items()
    }
     
    ####
    # ── Feature extraction per context (reusing FeatureExtractor) ──
    ctx_gmfp_peaks = {}
    ctx_lmfp_peaks = {}
    ctx_p2p_N15_P30 = {}
    ctx_p2p_N100_P180 = {}
    
    for ctx_name, ctx_ep in context_epochs.items():
        # Create a FeatureExtractor for this context subset
        ctx_extractor = FeatureExtractor(
            ctx_ep,
            config.analysis.channels_of_interest,
            config.analysis.time_windows,
        )
        
        # Compute GMFP and LMFP using the extractor
        ctx_gmfp_dict = ctx_extractor.compute_gmfp()
        ctx_lmfp_dict = ctx_extractor.compute_lmfp()
        
        # Extract peaks
        ctx_gmfp_peaks[ctx_name] = ctx_extractor.extract_mfp_peaks(
            ctx_gmfp_dict, label="GMFP"
        )
        ctx_lmfp_peaks[ctx_name] = ctx_extractor.extract_mfp_peaks(
            ctx_lmfp_dict, label="LMFP"
        )
        
        # Compute peak-to-peak amplitudes
        ctx_evokeds_single = ctx_extractor.get_evokeds()
        ctx_p2p_N15_P30[ctx_name] = ctx_extractor.peak_to_peak(
            "N15", "P30", evokeds=ctx_evokeds_single
        )
        ctx_p2p_N100_P180[ctx_name] = ctx_extractor.peak_to_peak(
            "N100", "P180", evokeds=ctx_evokeds_single
        )
    ####    
    
    # ── MFP Plots por contexto ───────────────────────────────────────
    mfp_plotter.plot_overlay(
        ctx_gmfp, label="GMFP_context",
        time_windows=config.analysis.time_windows,
    )
    mfp_plotter.plot_overlay(
        ctx_lmfp, label="LMFP_context",
        time_windows=config.analysis.time_windows,
    )

    # ── Contexts branch 1 comparison ──────────────────────────────────────────
    tep_plotter.plot_context_comparison(context_epochs)
    
    # ── Time evolution comparison ────────────────────────────────────
    # Metades (default)
    tep_plotter.plot_context_temporal_comparison(context_epochs)

    # Apenas ctx_01, ctx_11, ctx_21 em terços com rótulos customizados
    tep_plotter.plot_context_temporal_comparison(
        context_epochs,
        n_splits=3,
        split_labels=["Início", "Meio", "Fim"],
        contexts=["ctx_01", "ctx_11", "ctx_21"],
    )
    
    # ================================================================ #
    #  COLLECT RESULTS
    # ================================================================ #

    all_results.append({        
        "subject": subject_id,
        # Per-condition results
        "amplitude_N15_P30": amplitude_N15_P30,
        "amplitude_N100_P180": amplitude_N100_P180,
        "gmfp_peaks": df_gmfp_peaks,
        "lmfp_peaks": df_lmfp_peaks,
        # Per-context results
        "context_evokeds": ctx_evokeds,
        "context_gmfp": ctx_gmfp,
        "context_lmfp": ctx_lmfp,
        "context_gmfp_peaks": ctx_gmfp_peaks,
        "context_lmfp_peaks": ctx_lmfp_peaks,
        "context_p2p_N15_P30": ctx_p2p_N15_P30,
        "context_p2p_N100_P180": ctx_p2p_N100_P180,
    })

# if __name__ == "__main__":
#     main()
