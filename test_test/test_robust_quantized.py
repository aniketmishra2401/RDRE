import os
import sys
import numpy as np
import pytest

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_sythesis import generate_transition_signal
from wavelet_analysis import compute_wavelet
from metrics import calculate_accuracy, calculate_far
from trackers import fsm_tracker, hmm_tracker
from trackers_quantized import quantized_hmm_tracker


@pytest.fixture
def adversarial_schedule():
    """Generates a physical sequential wave transition schedule (1->2->3->2->1)."""
    return [
        (0.0, 1),
        (0.02, 2),
        (0.04, 3),
        (0.06, 2),
        (0.08, 1)
    ]


@pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.10, 0.15])
def test_trackers_under_noise(adversarial_schedule, noise_std):
    t, sig, truth_states = generate_transition_signal(
        adversarial_schedule, duration=0.1, noise_std=noise_std
    )
    
    f, t_frames, Zxx = compute_wavelet(sig, fs=10e6)
    truth_frames = np.interp(t_frames, t, truth_states).round()

    estimate_fsm = fsm_tracker(Zxx, f, rotation_freq_hz=5000)
    estimate_hmm = hmm_tracker(Zxx, f, rotation_freq_hz=5000)
    estimate_qhmm = quantized_hmm_tracker(Zxx, f, rotation_freq_hz=5000, scale_factor=256)
    
    acc_fsm = calculate_accuracy(truth_frames, estimate_fsm)
    far_fsm = calculate_far(truth_frames, estimate_fsm)
    
    acc_hmm = calculate_accuracy(truth_frames, estimate_hmm)
    far_hmm = calculate_far(truth_frames, estimate_hmm)
    
    acc_qhmm = calculate_accuracy(truth_frames, estimate_qhmm)
    far_qhmm = calculate_far(truth_frames, estimate_qhmm)
    
    print(
        f"\n[CWT] Noise: {noise_std:.2f} | "
        f"FSM Acc: {acc_fsm:.1f}%, FAR: {far_fsm:.1f}% | "
        f"Float HMM Acc: {acc_hmm:.1f}%, FAR: {far_hmm:.1f}% | "
        f"Q8.8 HMM Acc: {acc_qhmm:.1f}%, FAR: {far_qhmm:.1f}%"
    )
    
    if noise_std < 0.10:
        assert acc_qhmm >= 95.0, f"Quantized Q8.8 HMM accuracy fell below threshold: {acc_qhmm:.1f}%"
        assert far_qhmm <= 2.0, f"Quantized Q8.8 HMM FAR exceeded threshold: {far_qhmm:.1f}%"