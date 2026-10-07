import numpy as np

def build_quantized_transition_matrix(max_waves=8, p_stay=0.90, p_step=0.04, scale_factor=256):
    """
    Builds fixed-point integer log-transition matrix (Q8.8 representation).
    """
    epsilon = 1e-5
    A = np.full((max_waves, max_waves), epsilon)
    for i in range(max_waves):
        A[i, i] = p_stay
        if i > 0:
            A[i, i - 1] = p_step
        if i < max_waves - 1:
            A[i, i + 1] = p_step
        A[i, :] = A[i, :] / np.sum(A[i, :])
        
    log_A = np.log(np.maximum(A, 1e-12))
    log_A_q = np.round(log_A * scale_factor).astype(np.int32)
    return log_A_q


def compute_quantized_emissions(stft_magnitude, freq_bins, rotation_freq_hz, max_waves=8, beta=20.0, scale_factor=256):
    """
    Computes integer-quantized log-emission matrix B using Q8.8 scaling.
    """
    target_bin_indices = []
    for wave_count in range(1, max_waves + 1):
        target_freq = wave_count * rotation_freq_hz
        idx = np.argmin(np.abs(freq_bins - target_freq))
        target_bin_indices.append(idx)
        
    state_mags = stft_magnitude[target_bin_indices, :]
    
    col_max = np.max(state_mags, axis=0, keepdims=True)
    norm_mags = state_mags / (col_max + 1e-12)
    
    shift_mag = norm_mags * beta - np.max(norm_mags * beta, axis=0, keepdims=True)
    exp_mag = np.exp(shift_mag)
    B = exp_mag / np.sum(exp_mag, axis=0, keepdims=True)
    
    log_B = np.log(np.maximum(B, 1e-12))
    log_B_q = np.round(log_B * scale_factor).astype(np.int32)
    return log_B_q


def viterbi_decode_quantized(log_A_q, log_B_q, max_waves=8):
    """
    Integer Viterbi Add-Compare-Select (ACS) engine.
    """
    max_waves, num_frames = log_B_q.shape
    
    V = np.zeros((max_waves, num_frames), dtype=np.int32)
    backpointer = np.zeros((max_waves, num_frames), dtype=np.int16)
    
    initial_log_prob_q = int(np.round(np.log(1.0 / max_waves) * 256))
    V[:, 0] = initial_log_prob_q + log_B_q[:, 0]
    
    for t in range(1, num_frames):
        for j in range(max_waves):
            prob = V[:, t - 1] + log_A_q[:, j]
            best_prev_state = np.argmax(prob)
            V[j, t] = prob[best_prev_state] + log_B_q[j, t]
            backpointer[j, t] = best_prev_state
            
    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[-1] = np.argmax(V[:, -1])
    
    for t in range(num_frames - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]
        
    return best_path + 1


def quantized_hmm_tracker(Zxx, freq_bins, rotation_freq_hz=5000, max_waves=8, scale_factor=256):
    """
    16-bit Fixed-Point HMM Tracker wrapper.
    """
    magnitude_q = np.round(np.abs(Zxx) * scale_factor).astype(np.int32)
    
    log_A_q = build_quantized_transition_matrix(max_waves=max_waves, scale_factor=scale_factor)
    log_B_q = compute_quantized_emissions(magnitude_q / scale_factor, freq_bins, rotation_freq_hz, max_waves=max_waves, scale_factor=scale_factor)
    
    estimated_states = viterbi_decode_quantized(log_A_q, log_B_q, max_waves=max_waves)
    return estimated_states