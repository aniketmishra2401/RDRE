import numpy as np

def generate_transition_signal(
    schedule, 
    duration=0.1, 
    noise_std=0.05, 
    fs=10e6, 
    base_freq_hz=5000, 
    num_harmonics=3,
    jitter_pct=0.05,
    fade_duration_ms=1.5
):
    """
    High-Fidelity RDRE Synthetic Signal Generator.
    
    Physics Features Added:
    1. Steep Shock Fronts: Summation of 1st, 2nd, and 3rd harmonics (1/k amplitude scaling).
    2. Detonation Speed Drift: Continuous frequency jitter (+/- 5%) simulating flow fluctuations.
    3. Mode Competition: Smooth cosine fading during wave transitions (1.5 ms overlap).
    4. Broadband Noise: Additive background combustion/sensor noise.
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    sig = np.zeros(n_samples)
    truth_states = np.zeros(n_samples)
    
    schedule = sorted(schedule, key=lambda x: x[0])
    fade_samples = int((fade_duration_ms / 1000.0) * fs)
    
    # Detonation speed drift (low-frequency jitter simulating chamber pressure drift)
    jitter = jitter_pct * (0.6 * np.sin(2 * np.pi * 120 * t) + 0.4 * np.sin(2 * np.pi * 310 * t))
    
    for i, (t_start, wave_count) in enumerate(schedule):
        t_end = schedule[i + 1][0] if i + 1 < len(schedule) else duration
        
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)
        
        # Ground truth transitions at interval midpoints
        truth_states[idx_start:idx_end] = wave_count
        
        # Envelope matrix for transition fading
        env = np.ones(n_samples)
        
        # Cosine fade-in envelope
        if i > 0:
            fi_start = max(0, idx_start - fade_samples // 2)
            fi_end = min(n_samples, idx_start + fade_samples // 2)
            fi_len = fi_end - fi_start
            env[:fi_start] = 0.0
            env[fi_start:fi_end] = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fi_len)))
        else:
            env[:idx_start] = 0.0

        # Cosine fade-out envelope
        if i + 1 < len(schedule):
            fo_start = max(0, idx_end - fade_samples // 2)
            fo_end = min(n_samples, idx_end + fade_samples // 2)
            fo_len = fo_end - fo_start
            env[fo_start:fo_end] = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fo_len)))
            env[fo_end:] = 0.0
            
        # Instantaneous wave frequency: f(t) = N * f_base * (1 + jitter)
        inst_freq = wave_count * base_freq_hz * (1 + jitter)
        
        # Continuous phase integration: phi(t) = 2*pi * integral(f(t) dt)
        phase = 2 * np.pi * np.cumsum(inst_freq) / fs
        
        # Construct shock waveform using Fourier harmonic decay: 1 / k^1.1
        state_sig = np.zeros(n_samples)
        for k in range(1, num_harmonics + 1):
            state_sig += (0.5 / (k ** 1.1)) * np.sin(k * phase)
            
        sig += env * state_sig

    # Superimpose broadband Gaussian combustion noise
    sig += np.random.normal(0, noise_std, n_samples)
    
    return t, sig, truth_states









# import numpy as np

# def generate_transition_signal(schedule, duration=0.1, noise_std=0.01, fs=10e6):
#     """
#     Generates a synthetic pressure signal and the ground-truth wave states.
#     Corrected to ensure previous wave states shut off during transitions.
#     """
#     t = np.arange(0, duration, 1/fs)
    
#     # Start with baseline white noise
#     sig = np.random.normal(0, noise_std, len(t))
#     truth_states = np.ones(len(t))
    
#     base_freq_hz = 5000  # Assume 1-wave rotates at 5000 Hz
    
#     # Ensure schedule is sorted by time
#     schedule = sorted(schedule, key=lambda x: x[0])
    
#     for i in range(len(schedule)):
#         time_start, wave_count = schedule[i]
#         idx_start = int(time_start * fs)
        
#         # Figure out when this wave state ends
#         if i + 1 < len(schedule):
#             time_end = schedule[i + 1][0]
#             idx_end = int(time_end * fs)
#         else:
#             idx_end = len(t) # Last state goes to the end of the array
            
#         # Update the ground truth array for just this window
#         truth_states[idx_start:idx_end] = wave_count
        
#         # Inject the frequency ONLY during this specific time window
#         target_freq = wave_count * base_freq_hz
#         sig[idx_start:idx_end] += 0.5 * np.sin(2 * np.pi * target_freq * t[idx_start:idx_end])
        
#     return t, sig, truth_states