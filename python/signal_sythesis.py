import numpy as np

def generate_dual_sensor_rdre_signal(
    duration: float = 0.2,          # Total duration in seconds
    fs: float = 10e6,               # Sampling rate (10 MHz)
    delta_theta_deg: float = 60.0,  # Sensor angular separation (degrees)
    f_base: float = 5000.0,         # Base rotation frequency (Hz)
    snr_db: float = 15.0,           # Target SNR in dB
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synchronized dual-sensor pressure traces (PT1, PT2) for an RDRE
    with exact phase delays reflecting wave count (N) and direction (+1 CCW, -1 CW).
    """
    np.random.seed(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    delta_theta_rad = np.radians(delta_theta_deg)
    
    # State sequence definition: (Mode N, Direction [+1/-1], Start Time, End Time)
    states = [
        (1,  1, 0.00, 0.04),   # W1+ (CCW)
        (2,  1, 0.04, 0.08),   # W2+ (CCW)
        (3,  1, 0.08, 0.12),   # W3+ (CCW)
        (2, -1, 0.12, 0.16),   # W2- (CW Reversal)
        (1,  1, 0.16, 0.20),   # W1+ (CCW)
    ]
    
    sig1 = np.zeros(n_samples)
    sig2 = np.zeros(n_samples)
    labels = np.zeros(n_samples, dtype=int)
    
    for N, direction, t_start, t_end in states:
        idx = (t >= t_start) & (t < t_end)
        if not np.any(idx):
            continue
            
        t_sub = t[idx]
        labels[idx] = N * direction  # Signed ground-truth label
        
        # Fundamental target frequency with a 5% speed wobble
        f_target = N * f_base
        wobble = 0.05 * np.sin(2 * np.pi * 50 * (t_sub - t_start))
        f_inst = f_target * (1.0 + wobble)
        
        # Accumulate instantaneous phase for Sensor 1
        phase1 = 2 * np.pi * np.cumsum(f_inst) / fs
        
        # Spatial phase shift for Sensor 2: Δφ = direction * N * Δθ
        spatial_phase_shift = direction * N * delta_theta_rad
        phase2 = phase1 - spatial_phase_shift
        
        # Synthesize steep detonation profile using 3 harmonics
        wave1 = np.zeros_like(t_sub)
        wave2 = np.zeros_like(t_sub)
        harmonics = [1, 2, 3]
        weights = [1.0, 0.5, 0.25]
        
        for k, w in zip(harmonics, weights):
            wave1 += w * np.cos(k * phase1)
            wave2 += w * np.cos(k * phase2)
            
        sig1[idx] = wave1
        sig2[idx] = wave2

    # Add white Gaussian noise based on signal power
    p_sig = np.mean(sig1**2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise1 = np.random.normal(0, np.sqrt(p_noise), n_samples)
    noise2 = np.random.normal(0, np.sqrt(p_noise), n_samples)
    
    return t, sig1 + noise1, sig2 + noise2, labels