import numpy as np

def decompose_co_counter_waves(
    Z1_complex: complex,
    Z2_complex: complex,
    N_candidate: int,
    delta_theta_deg: float = 60.0,
    eps: float = 1e-9
) -> tuple[float, float, float]:
    phi = np.abs(N_candidate) * np.radians(delta_theta_deg)
    denom = 2j * np.sin(phi)
    if np.abs(np.sin(phi)) < 1e-3:
        denom = 2j * 1e-3
        
    A = (Z1_complex * np.exp(1j * phi) - Z2_complex) / denom
    B = (Z2_complex - Z1_complex * np.exp(-1j * phi)) / denom
    
    power_A = np.abs(A) ** 2
    power_B = np.abs(B) ** 2
    R_backward = power_B / (power_A + power_B + eps)
    
    return np.abs(A), np.abs(B), R_backward


class SignedReversalHMM:
    def __init__(self, n_states: int = 10):
        self.n_states = n_states
        self.A = np.zeros((self.n_states, self.n_states))
        
        # High self-transition inertia (0.999) suppresses frame chatter
        np.fill_diagonal(self.A, 0.999)
        self.A[0, 0] = 0.999  # QUENCH dwell
        self.A[9, 9] = 0.900  # REV dwell
        
        for i in range(self.n_states):
            rem_prob = (1.0 - np.sum(self.A[i, :])) / (self.n_states - 1)
            self.A[i, :] += rem_prob
            
        self.pi = np.ones(self.n_states) / self.n_states

    def decode_viterbi_log_B(self, emission_log_B: np.ndarray) -> np.ndarray:
        n_states, T = emission_log_B.shape
        viterbi = np.zeros((n_states, T))
        backpointer = np.zeros((n_states, T), dtype=int)
        
        log_pi = np.log(self.pi + 1e-12)
        log_A = np.log(self.A + 1e-12)
        
        viterbi[:, 0] = log_pi + emission_log_B[:, 0]
        
        for t in range(1, T):
            for s in range(n_states):
                trans_prob = viterbi[:, t - 1] + log_A[:, s]
                best_prev = np.argmax(trans_prob)
                viterbi[s, t] = trans_prob[best_prev] + emission_log_B[s, t]
                backpointer[s, t] = best_prev
                
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[:, -1])
        
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[best_path[t + 1], t + 1]
            
        return best_path
