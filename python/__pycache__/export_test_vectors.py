import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
cwd = os.getcwd()

for p in [file_dir, parent_dir, cwd]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from signal_sythesis import generate_transition_signal
from wavelet_analysis import compute_wavelet
from trackers_quantized import (
    build_quantized_transition_matrix,
    compute_quantized_emissions,
)


def viterbi_decode_forward_quantized(log_A_q, log_B_q, max_waves=8):
    """
    Real-Time Forward Viterbi Filter (Zero-Latency Hardware Parity).
    Matches cycle-by-cycle ACS hardware register execution bit-for-bit.
    """
    max_waves, num_frames = log_B_q.shape
    V = np.zeros((max_waves, num_frames), dtype=np.int32)

    initial_log_prob_q = int(np.round(np.log(1.0 / max_waves) * 256))
    V[:, 0] = initial_log_prob_q + log_B_q[:, 0]

    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[0] = np.argmax(V[:, 0]) + 1

    for t in range(1, num_frames):
        for j in range(max_waves):
            prob = V[:, t - 1] + log_A_q[:, j]
            V[j, t] = np.max(prob) + log_B_q[j, t]
        best_path[t] = np.argmax(V[:, t]) + 1

    return best_path


def export_test_vectors(output_dir="test_vectors", scale_factor=256):
    os.makedirs(output_dir, exist_ok=True)

    schedule = [(0.0, 1), (0.02, 2), (0.04, 3), (0.06, 2), (0.08, 1)]
    t, sig, truth_states = generate_transition_signal(
        schedule, duration=0.1, noise_std=0.05
    )

    f, t_frames, Zxx = compute_wavelet(sig, fs=10e6)

    magnitude_q = np.round(np.abs(Zxx) * scale_factor).astype(np.int32)
    log_A_q = build_quantized_transition_matrix(max_waves=8, scale_factor=scale_factor)
    log_B_q = compute_quantized_emissions(
        magnitude_q / scale_factor,
        f,
        rotation_freq_hz=5000,
        max_waves=8,
        scale_factor=scale_factor,
    )

    # Use streaming forward decoding for bit-exact hardware parity
    golden_states = viterbi_decode_forward_quantized(log_A_q, log_B_q, max_waves=8)

    path_A = os.path.join(output_dir, "transition_matrix_A.hex")
    with open(path_A, "w") as f_a:
        for row in log_A_q:
            for val in row:
                f_a.write(f"{(int(val) & 0xFFFFFFFF):08X}\n")

    path_B = os.path.join(output_dir, "emission_matrix_B.hex")
    with open(path_B, "w") as f_b:
        num_frames = log_B_q.shape[1]
        for frame_idx in range(num_frames):
            for state_idx in range(8):
                val = log_B_q[state_idx, frame_idx]
                f_b.write(f"{(int(val) & 0xFFFFFFFF):08X}\n")

    path_G = os.path.join(output_dir, "golden_states.hex")
    with open(path_G, "w") as f_g:
        for state in golden_states:
            f_g.write(f"{(int(state) & 0xFFFF):04X}\n")

    print(f"\n[SUCCESS] Exported test vectors to '{output_dir}/'")


if __name__ == "__main__":
    export_test_vectors()