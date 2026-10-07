import pytest
import numpy as np
import sys
import os

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_sythesis import generate_transition_signal
from wavelet_analysis import compute_wavelet
from metrics import calculate_accuracy, calculate_far
from trackers import fsm_tracker, hmm_tracker

@pytest.fixture
def adversarial_schedule():
    """Generates a sequential wave transition schedule (1->2->3->2->1)."""
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
    
    acc_fsm = calculate_accuracy(truth_frames, estimate_fsm)
    far_fsm = calculate_far(truth_frames, estimate_fsm)
    
    acc_hmm = calculate_accuracy(truth_frames, estimate_hmm)
    far_hmm = calculate_far(truth_frames, estimate_hmm)
    
    print(f"\n[CWT] Noise: {noise_std} | FSM Acc: {acc_fsm:.1f}%, FAR: {far_fsm:.1f}% | HMM Acc: {acc_hmm:.1f}%, FAR: {far_hmm:.1f}%")
    
    if noise_std < 0.10:
        assert acc_hmm > 85.0, f"HMM failed accuracy at noise {noise_std}"








# import pytest
# import numpy as np
# import sys
# import os

# # This strictly tells Python and Pylance to look in the parent folder for your files
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from signal_sythesis import generate_transition_signal
# from stft_analysis import compute_stft
# from metrics import calculate_accuracy, calculate_far, calculate_latency
# from trackers import fsm_tracker, hmm_tracker

# @pytest.fixture
# def adversarial_schedule():
#     """Generates a sequential wave transition schedule (1->2->3->2->1)."""
#     return [
#         (0.0, 1),
#         (0.02, 2),
#         (0.04, 3),
#         (0.06, 2),
#         (0.08, 1)
#     ]

# @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.10, 0.15])
# def test_trackers_under_noise(adversarial_schedule, noise_std):
    
#     t, sig, truth_states = generate_transition_signal(
#         adversarial_schedule, duration=0.1, noise_std=noise_std
#     )
    
#     f, t_frames, Zxx = compute_stft(sig, fs=10e6)
   
    
#     truth_frames = np.interp(t_frames, t, truth_states).round()

#     estimate_fsm = fsm_tracker(Zxx, f, rotation_freq_hz=5000)
#     estimate_hmm = hmm_tracker(Zxx, f, rotation_freq_hz=5000)
    
#     acc_fsm = calculate_accuracy(truth_frames, estimate_fsm)
#     far_fsm = calculate_far(truth_frames, estimate_fsm)
    
#     acc_hmm = calculate_accuracy(truth_frames, estimate_hmm)
#     far_hmm = calculate_far(truth_frames, estimate_hmm)
    
#     print(f"\nNoise: {noise_std} | FSM Acc: {acc_fsm:.1f}%, FAR: {far_fsm:.1f}% | HMM Acc: {acc_hmm:.1f}%, FAR: {far_hmm:.1f}%")
    
#     if noise_std < 0.10:
#         assert acc_fsm > 70.0, f"FSM failed accuracy at noise {noise_std}"
#         assert far_fsm < 20.0, f"FSM FAR too high at noise {noise_std}"