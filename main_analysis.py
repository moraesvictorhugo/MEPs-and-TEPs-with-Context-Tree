"""Main analysis pipeline for TMS-EEG data."""

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data, get_raw_path
from tms_eeg.io.writer import Writer
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.group import MetricsCollector
from tms_eeg.visualization.tep_plots import TEPPlotter
from tms_eeg.visualization.gfp_plots import MFPPlotter

# Set backend
from tms_eeg.config.environment import setup_plotting_backend
setup_plotting_backend()

collector = MetricsCollector()
subjects = ProjectConfig().analysis.subjects

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")
    
    # ── Renomear condições ───────────────────────────────────────────
    label_map = {"8bit 1": "0", "8bit 2": "1", "8bit 3": "2"}
    epochs.event_id = {label_map[k.lower()]: v for k, v in epochs.event_id.items()}

    # ── Shared objects ───────────────────────────────────────────────
    writer = Writer(config)
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
    amplitude_N15_P60 = extractor.peak_to_peak("N15", "P60", evokeds=evokeds)
    amplitude_N100_P180 = extractor.peak_to_peak(
        "N100", "P180", evokeds=evokeds)
    
    # ── Collect peak-to-peak rows ──
    collector.collect_peak_to_peak_from_df(
        subject_id, "condition", amplitude_N15_P30, "N15-P30"
    )
    collector.collect_peak_to_peak_from_df(
        subject_id, "condition", amplitude_N15_P60, "N15-P60"
    )
    collector.collect_peak_to_peak_from_df(
        subject_id, "condition", amplitude_N100_P180, "N100-P180"
    )

    # ── GMFP & LMFP calculation ─────────────────────────────────────
    gmfp = extractor.compute_gmfp()
    lmfp = extractor.compute_lmfp()

    # ── MFP peak extraction ──────────────────────────────────────────
    df_gmfp_peaks = extractor.extract_mfp_peaks(gmfp, label="GMFP")
    df_lmfp_peaks = extractor.extract_mfp_peaks(lmfp, label="LMFP")
    
    # ── Collect MFP peaks rows ──
    collector.collect_mfp_peaks_from_df(
        subject_id, "condition", df_gmfp_peaks, "GMFP"
    )
    collector.collect_mfp_peaks_from_df(
        subject_id, "condition", df_lmfp_peaks, "LMFP"
    )
   
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

    # ── Feature extraction per context (reusing FeatureExtractor) ──
    ctx_gmfp = {}
    ctx_lmfp = {}
    ctx_gmfp_peaks = {}
    ctx_lmfp_peaks = {}
    ctx_p2p_N15_P30 = {}
    ctx_p2p_N15_P60 = {}
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
        
        # Store for plotting (single context → single curve)
        ctx_gmfp[ctx_name] = list(ctx_gmfp_dict.values())[0]
        ctx_lmfp[ctx_name] = list(ctx_lmfp_dict.values())[0]
        
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
        ctx_p2p_N15_P60[ctx_name] = ctx_extractor.peak_to_peak(
            "N15", "P60", evokeds=ctx_evokeds_single
        )
        ctx_p2p_N100_P180[ctx_name] = ctx_extractor.peak_to_peak(
            "N100", "P180", evokeds=ctx_evokeds_single
        )
    
    # ── Collect context metrics ──
    for ctx_name, df in ctx_p2p_N15_P30.items():
        df = df.copy()
        df["condition"] = ctx_name
        collector.collect_peak_to_peak_from_df(
            subject_id, "context", df, "N15-P30"
        )
    
    for ctx_name, df in ctx_p2p_N15_P60.items():
        df = df.copy()
        df["condition"] = ctx_name
        collector.collect_peak_to_peak_from_df(
            subject_id, "context", df, "N15-P60"
        )

    for ctx_name, df in ctx_p2p_N100_P180.items():
        df = df.copy()
        df["condition"] = ctx_name
        collector.collect_peak_to_peak_from_df(
            subject_id, "context", df, "N100-P180"
        )

    for ctx_name, df in ctx_gmfp_peaks.items():
        df = df.copy()
        df["condition"] = ctx_name
        collector.collect_mfp_peaks_from_df(
            subject_id, "context", df, "GMFP"
        )

    for ctx_name, df in ctx_lmfp_peaks.items():
        df = df.copy()
        df["condition"] = ctx_name
        collector.collect_mfp_peaks_from_df(
            subject_id, "context", df, "LMFP"
        )

    
    
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
    #  COLLECT RESULTS (tidy format - rows already collected above)
    # ================================================================ #

# ── Export to CSV if enabled ──
config_check = ProjectConfig()
database = collector.export_csv(
    output_path="data/group/database.csv",
    export_enabled=config_check.io.export_data,
)