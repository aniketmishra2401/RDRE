"""Generate reproducible plots and tabular outputs from the RDRE Python testbench.

These are synthetic-benchmark outputs, not experimental combustion measurements.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_NAMES = {
    0: "QUENCH", 1: "DEFLAGRATION", 2: "AXIAL_PULSE",
    3: "W1+", 4: "W1-", 5: "W2+", 6: "W2-",
    7: "W3+", 8: "W3-", 9: "REV"
}

def _save_fig(fig, outdir, stem):
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def generate_report(
    outdir, t_raw, sig1_raw, sig2_raw, gt_raw,
    freqs, t_frames, z1_flat, n_cont, phase, frame_power,
    gt_frame, pred_state_indices, pred_n_labels, metrics,
    fs, target_fs, delta_theta_deg, snr_db
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # FIG. 1 — Architecture diagram
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    blocks = [
        (0.02, 0.57, 0.15, 0.23, "Synthetic dual-sensor\nsignal generator"),
        (0.21, 0.57, 0.15, 0.23, "Signal conditioning\nand resampling"),
        (0.40, 0.57, 0.15, 0.23, "Time-frequency\nand phase features"),
        (0.59, 0.57, 0.15, 0.23, "State emissions\nand HMM decoding"),
        (0.78, 0.57, 0.19, 0.23, "State timeline and\nevent evaluation"),
    ]
    for x, y, w, h, label in blocks:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)
    for i in range(len(blocks)-1):
        x1 = blocks[i][0] + blocks[i][2]
        x2 = blocks[i+1][0]
        ax.annotate("", xy=(x2-0.005, 0.685), xytext=(x1+0.005, 0.685),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(0.5, 0.30, "Automated outputs: signal plots • state comparison • metrics • CSV data",
            ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.12, "Current benchmark uses synthetic signals; no injector or combustion model is included.",
            ha="center", va="center", fontsize=9)
    ax.set_title("FIG. 1 — RDRE monitoring testbench architecture", fontsize=13)
    _save_fig(fig, outdir, "fig01_architecture")

    # FIG. 2 — raw dual-sensor signals (plot decimated points to keep files small)
    stride = max(1, len(t_raw)//50000)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_raw[::stride]*1000, sig1_raw[::stride], label="Sensor 1", lw=0.65)
    ax.plot(t_raw[::stride]*1000, sig2_raw[::stride], label="Sensor 2", lw=0.65, alpha=0.8)
    for x in [40, 80, 120, 160]:
        ax.axvline(x, ls="--", lw=0.9)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Synthetic signal amplitude (arbitrary units)")
    ax.set_title("FIG. 2 — Dual-sensor synthetic input and programmed transitions")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_fig(fig, outdir, "fig02_dual_sensor_input")

    # FIG. 3 — scalogram-like CWT magnitude image and phase/frequency traces
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    extent = [t_frames[0]*1000, t_frames[-1]*1000, freqs[0]/1000, freqs[-1]/1000]
    axes[0].imshow(z1_flat, aspect="auto", origin="lower", extent=extent, interpolation="nearest")
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_title("CWT magnitude (arbitrary units)")
    axes[1].plot(t_frames*1000, phase, lw=0.8)
    axes[1].set_ylabel("Inter-sensor phase (rad)")
    axes[2].plot(t_frames*1000, n_cont, lw=0.8)
    axes[2].set_ylabel("Continuous wave-count proxy")
    axes[2].set_xlabel("Time (ms)")
    for ax in axes:
        for x in [40, 80, 120, 160]:
            ax.axvline(x, ls="--", lw=0.8, alpha=0.7)
        ax.grid(True, alpha=0.2)
    fig.suptitle("FIG. 3 — Time-frequency and phase-derived features", y=1.01)
    _save_fig(fig, outdir, "fig03_time_frequency_phase")

    # FIG. 4 — ground truth versus predicted state sequence
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.step(t_frames*1000, gt_frame, where="post", label="Ground-truth signed wave label", lw=1.5)
    ax.step(t_frames*1000, pred_n_labels, where="post", label="Predicted signed wave label", lw=0.9, alpha=0.85)
    for x in [40, 80, 120, 160]:
        ax.axvline(x, ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Signed wave label")
    ax.set_title("FIG. 4 — Ground truth versus predicted signed wave labels")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_fig(fig, outdir, "fig04_state_comparison")

    # FIG. 5 — state-index timeline preserves all 10 state identities
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(t_frames*1000, pred_state_indices, where="post", lw=0.9)
    ax.set_yticks(list(STATE_NAMES.keys()), [STATE_NAMES[i] for i in STATE_NAMES])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Decoded HMM state")
    ax.set_title("FIG. 5 — Full decoded ten-state timeline")
    ax.grid(True, alpha=0.25)
    _save_fig(fig, outdir, "fig05_full_state_timeline")

    # FIG. 6 — metrics summary chart where numeric values are available
    metric_labels, metric_values = [], []
    for k in ["True Transition Events", "Matched Transitions", "Missed Transitions", "False Alarm Events"]:
        if k in metrics:
            try:
                metric_values.append(float(metrics[k]))
                metric_labels.append(k)
            except (TypeError, ValueError):
                pass
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if metric_values:
        bars = ax.bar(metric_labels, metric_values)
        ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Count")
    ax.set_title("FIG. 6 — Preliminary event-evaluation counts")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    _save_fig(fig, outdir, "fig06_event_counts")

    # Export downsampled raw data, frame-level features and JSON metrics.
    raw_stride = max(1, len(t_raw)//100000)
    with open(outdir / "dual_sensor_input.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "sensor1_arb", "sensor2_arb", "ground_truth_signed_wave_label"])
        w.writerows(zip(t_raw[::raw_stride], sig1_raw[::raw_stride],
                        sig2_raw[::raw_stride], gt_raw[::raw_stride]))
    with open(outdir / "frame_features_and_states.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "ground_truth_signed_label", "predicted_signed_label",
                    "predicted_state_index", "continuous_wave_count_proxy", "phase_rad", "frame_power"])
        w.writerows(zip(t_frames, gt_frame, pred_n_labels, pred_state_indices,
                        n_cont, phase, frame_power))
    report = {
        "evidence_type": "synthetic software benchmark; not experimental combustion data",
        "sampling_rate_hz": float(fs),
        "target_sampling_rate_hz": float(target_fs),
        "sensor_separation_deg": float(delta_theta_deg),
        "configured_snr_db": float(snr_db),
        "metrics_note": "Metrics use the current event_evaluator implementation and should be treated as preliminary until evaluator corrections are applied.",
        "metrics": metrics,
        "figures": [
            "fig01_architecture.png/.pdf",
            "fig02_dual_sensor_input.png/.pdf",
            "fig03_time_frequency_phase.png/.pdf",
            "fig04_state_comparison.png/.pdf",
            "fig05_full_state_timeline.png/.pdf",
            "fig06_event_counts.png/.pdf"
        ]
    }
    (outdir / "run_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFigure/report outputs saved to: {outdir.resolve()}")
    print("Figures exported as PNG (300 dpi) and PDF; data exported as CSV.")
