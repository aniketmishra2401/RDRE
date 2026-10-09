import numpy as np

from signal_sythesis import generate_dual_sensor_rdre_signal
from flattened_front_end import compute_flattened_cwt_with_noise_gate
from scale_free_acoustic_refinement import ScaleFreeAcousticDiscriminator
from signed_wave_tracker import decompose_co_counter_waves, SignedReversalHMM
from sensor_calib_and_expanded_nodet import SensorDelayCalibrator, ExpandedStateObservationEngine
from event_evaluator import evaluate_event_metrics

def apply_state_hysteresis(state_seq: np.ndarray, min_hold_frames: int = 15) -> np.ndarray:
    """Holds a state transition for at least min_hold_frames (75 μs) to reject transient glitches."""
    smoothed = np.copy(state_seq)
    current_state = state_seq[0]
    hold_counter = 0

    for i in range(1, len(state_seq)):
        candidate = state_seq[i]
        if candidate == current_state:
            hold_counter += 1
        else:
            if hold_counter >= min_hold_frames:
                current_state = candidate
                hold_counter = 0
            else:
                smoothed[i] = current_state
                hold_counter += 1
    return smoothed

def run_full_rdre_pipeline():
    print("=== 1. Generating Dual-Sensor Pressure Signals ===")
    fs = 10e6
    target_fs = 200e3
    duration = 0.2
    delta_theta_deg = 60.0
    
    t, sig1_raw, sig2_raw, gt_labels = generate_dual_sensor_rdre_signal(
        duration=duration, 
        fs=fs, 
        delta_theta_deg=delta_theta_deg, 
        snr_db=15.0
    )

    print("=== 2. High-Pass Filtering & Causal L1-CWT Front End ===")
    calibrator = SensorDelayCalibrator(fs=fs, fc_highpass=500.0)
    sig1_clean = calibrator.remove_thermal_drift(sig1_raw)
    sig2_clean = calibrator.remove_thermal_drift(sig2_raw)

    freqs, t_frames, Z1_flat, Z1_complex, Z2_complex, n_cont, smooth_phi12, frame_power = compute_flattened_cwt_with_noise_gate(
        sig1=sig1_clean, 
        sig2=sig2_clean, 
        fs=fs, 
        target_fs=target_fs,
        delta_theta_deg=delta_theta_deg
    )

    print("=== 3. Mode Refinement & Continuous Emission Generation ===")
    obs_engine = ExpandedStateObservationEngine(delta_theta_deg=delta_theta_deg)
    
    n_frames = len(t_frames)
    emission_log_B = np.zeros((10, n_frames))

    p_quench_thresh = 1e-6

    for i in range(n_frames):
        f_peak_idx = np.argmax(Z1_flat[:, i])
        f_candidate = freqs[f_peak_idx]

        phi12_raw = smooth_phi12[i]
        phi12_corr = calibrator.correct_phase(f_candidate, phi12_raw)

        emission_log_B[:, i] = obs_engine.evaluate_emissions(
            n_cont=n_cont[i], 
            phi12_rad=phi12_corr, 
            total_power=frame_power[i],
            p_quench_thresh=p_quench_thresh, 
            C_harm=1.0
        )

    print("=== 4. Viterbi State Decoding & Persistence Debouncing ===")
    hmm = SignedReversalHMM(n_states=10)
    pred_path_indices = hmm.decode_viterbi_log_B(emission_log_B)
    
    smoothed_path_indices = apply_state_hysteresis(pred_path_indices, min_hold_frames=15)

    # State mapping: 0:QUENCH, 1:DEFLAG, 2:AXIAL, 3:W1+, 4:W1-, 5:W2+, 6:W2-, 7:W3+, 8:W3-, 9:REV
    state_to_N_map = np.array([0, 0, 0, +1, -1, +2, -2, +3, -3, 0])
    pred_N_labels = state_to_N_map[smoothed_path_indices]

    print("=== 5. Evaluating Performance ===")
    gt_frame_labels = gt_labels[::50][:n_frames]
    
    metrics = evaluate_event_metrics(
        t_frames=t_frames, 
        gt_labels=gt_frame_labels, 
        pred_labels=pred_N_labels,
        tolerance_sec=0.001, 
        search_window_sec=0.005
    )

    print("\n--- TESTBENCH RESULTS ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

if __name__ == "__main__":
    run_full_rdre_pipeline()
