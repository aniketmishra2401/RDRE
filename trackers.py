import numpy as np

# ==========================================
# SAKSHAM'S WORKSPACE: FSM BASELINE TRACKER
# ==========================================

def fsm_tracker(Zxx, freq_bins, rotation_freq_hz=5000, debounce_limit=3, max_waves=8):
    """
    Finite State Machine with Target-Bin Peak Tracking and Debounce logic.
    """
    magnitude = np.abs(Zxx)
    num_frames = magnitude.shape[1]
    estimated_states = np.zeros(num_frames, dtype=int)
    
    # 1. Pre-calculate bin indices corresponding strictly to fundamental wave states (1..max_waves)
    target_bin_indices = []
    for wave_count in range(1, max_waves + 1):
        target_freq = wave_count * rotation_freq_hz
        idx = np.argmin(np.abs(freq_bins - target_freq))
        target_bin_indices.append(idx)
        
    current_state = 1
    candidate_state = 1
    debounce_counter = 0
    
    for t in range(num_frames):
        # 2. Target Peak Detection: Evaluate magnitude ONLY at valid wave state bins
        state_magnitudes = magnitude[target_bin_indices, t]
        
        # 3. Map max target bin index to wave state count (0-indexed bin -> 1..8 wave count)
        raw_state = int(np.argmax(state_magnitudes) + 1)
        
        # 4. FSM Debounce Logic: Ignore single-frame noise spikes
        if raw_state == current_state:
            debounce_counter = 0
        else:
            if raw_state == candidate_state:
                debounce_counter += 1
            else:
                candidate_state = raw_state
                debounce_counter = 1
                
            if debounce_counter >= debounce_limit:
                current_state = candidate_state
                debounce_counter = 0
                
        estimated_states[t] = current_state
        
    return estimated_states

# ==========================================
# ADITYA'S WORKSPACE: PHYSICS-INFORMED HMM
# ==========================================

def build_transition_matrix(max_waves=8, p_stay=0.90, p_step=0.04, epsilon=1e-5):
    """Builds an (N x N) physics-informed state transition matrix A."""
    A = np.full((max_waves, max_waves), epsilon)
    for i in range(max_waves):
        A[i, i] = p_stay # 1. Stay in same wave state (N -> N)
        if i > 0:
            A[i, i - 1] = p_step # 2. Wave merge (N -> N - 1)
        if i < max_waves - 1:
            A[i, i + 1] = p_step # 3. Wave split (N -> N + 1)
        A[i, :] = A[i, :] / np.sum(A[i, :]) # 4. Normalize
    return A

def compute_emission_probabilities(stft_magnitude, freq_bins, rotation_freq_hz, max_waves=8, beta=20.0):
    """
    Computes emission probability matrix B using Softmax.
    Includes column-wise normalization to prevent probability wash-out.
    """
    target_bin_indices = []
    for wave_count in range(1, max_waves + 1):
        target_freq = wave_count * rotation_freq_hz
        idx = np.argmin(np.abs(freq_bins - target_freq))
        target_bin_indices.append(idx)
        
    # Extract the raw magnitudes for the 8 wave states
    state_magnitudes = stft_magnitude[target_bin_indices, :]
    
    # --- THE FIX: Normalize per time-frame ---
    # Divide every column by its maximum value so the peak is always 1.0
    col_max = np.max(state_magnitudes, axis=0, keepdims=True)
    normalized_mags = state_magnitudes / (col_max + 1e-12) 
    
    # Apply Softmax with the new beta scaling
    shift_mag = normalized_mags * beta - np.max(normalized_mags * beta, axis=0, keepdims=True)
    exp_mag = np.exp(shift_mag)
    B = exp_mag / np.sum(exp_mag, axis=0, keepdims=True)
    
    return B

def viterbi_decode(A, B):
    """Runs log-space Viterbi decoding to find optimal path."""
    max_waves, num_frames = B.shape
    log_A = np.log(np.maximum(A, 1e-12))
    log_B = np.log(np.maximum(B, 1e-12))
    
    V = np.zeros((max_waves, num_frames))
    backpointer = np.zeros((max_waves, num_frames), dtype=int)
    
    V[:, 0] = np.log(1.0 / max_waves) + log_B[:, 0]
    
    for t in range(1, num_frames):
        for j in range(max_waves):
            prob = V[:, t - 1] + log_A[:, j]
            best_prev_state = np.argmax(prob)
            V[j, t] = prob[best_prev_state] + log_B[j, t]
            backpointer[j, t] = best_prev_state
            
    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[-1] = np.argmax(V[:, -1])
    
    for t in range(num_frames - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]
        
    return best_path + 1

def hmm_tracker(Zxx, freq_bins, rotation_freq_hz=5000, max_waves=8):
    """Main HMM pipeline."""
    # Ensure magnitude is extracted from complex STFT
    stft_magnitude = np.abs(Zxx)
    
    A = build_transition_matrix(max_waves=max_waves, p_stay=0.90, p_step=0.04)
    B = compute_emission_probabilities(stft_magnitude, freq_bins, rotation_freq_hz, max_waves=max_waves)
    estimated_states = viterbi_decode(A, B)
    
    return estimated_states