import numpy as np
from scipy.signal import decimate

def compute_causal_cwt_and_phase(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float = 10e6,
    target_fs: float = 200e3,
    delta_theta_deg: float = 60.0,
    w0: float = 6.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a strictly causal CWT on dual sensor signals and extracts
    speed-independent signed wave count estimates (N) via cross-phase.
    
    Parameters:
        sig1, sig2      : Dual-sensor pressure signals (PT1, PT2)
        fs              : Raw sampling rate (10 MHz)
        target_fs       : Decimated sampling rate (200 kHz)
        delta_theta_deg : Angular spacing between sensors in degrees
        w0              : Morlet central frequency parameter
        
    Returns:
        freqs           : Evaluated frequency bins (2 kHz - 42 kHz)
        t_frames        : Causal time vector (seconds)
        Z1_mag          : CWT magnitude spectrogram for Sensor 1
        signed_N_raw    : Instantaneous signed wave count per frame
    """
    delta_theta_rad = np.radians(delta_theta_deg)
    
    # 1. Decimate both signals (Pre-decimation pipeline)
    decimation_factor = max(1, int(fs / target_fs))
    sig1_dec = decimate(sig1, decimation_factor) if decimation_factor > 1 else sig1
    sig2_dec = decimate(sig2, decimation_factor) if decimation_factor > 1 else sig2
    fs_dec = fs / decimation_factor
    
    n_samples = len(sig1_dec)
    freqs = np.linspace(2000, 42000, 100)
    
    Z1_complex = np.zeros((len(freqs), n_samples), dtype=complex)
    Z2_complex = np.zeros((len(freqs), n_samples), dtype=complex)
    
    # 2. Compute Causal CWT per frequency bin
    for idx, f_target in enumerate(freqs):
        scale_sec = w0 / (2 * np.pi * f_target)
        
        # Window length (10-sigma span)
        M = int(10 * scale_sec * fs_dec)
        if M % 2 == 0:
            M += 1
            
        t_wavelet = (np.arange(M) - M // 2) / fs_dec
        norm = (1.0 / np.sqrt(scale_sec)) * (np.pi ** -0.25)
        
        # Complex Morlet Kernel
        wavelet = norm * np.exp(1j * 2 * np.pi * f_target * t_wavelet) * np.exp(-0.5 * (t_wavelet / scale_sec) ** 2)
        
        # Causal FIR Group Delay Shift:
        # Pad front of signal with M-1 zeros so convolution output at index 'k'
        # depends strictly on samples <= k (zero lookahead into future samples)
        pad_len = M - 1
        sig1_padded = np.pad(sig1_dec, (pad_len, 0), mode='constant')
        sig2_padded = np.pad(sig2_dec, (pad_len, 0), mode='constant')
        
        # Perform 1D time-domain FIR convolution
        conv1 = np.convolve(sig1_padded, np.conj(wavelet[::-1]), mode='valid')
        conv2 = np.convolve(sig2_padded, np.conj(wavelet[::-1]), mode='valid')
        
        # Store complex outputs (truncated to match signal length)
        Z1_complex[idx, :] = conv1[:n_samples]
        Z2_complex[idx, :] = conv2[:n_samples]

    # 3. Speed-Independent Cross-Spectral Phase Extraction
    # Calculate cross-spectrum: Z1 * conj(Z2)
    cross_spectrum = Z1_complex * np.conj(Z2_complex)
    
    # Extract dominant frequency index per frame
    Z1_mag = np.abs(Z1_complex)
    peak_freq_indices = np.argmax(Z1_mag, axis=0)
    
    signed_N_raw = np.zeros(n_samples)
    
    for t_idx in range(n_samples):
        f_idx = peak_freq_indices[t_idx]
        
        # Compute cross-spectral phase at the dominant peak
        phi_12 = np.angle(cross_spectrum[f_idx, t_idx])
        
        # Unambiguous wave count calculation: N = phi_12 / delta_theta
        # Direction (+1 CCW, -1 CW) is preserved in the sign of phi_12
        n_continuous = phi_12 / delta_theta_rad
        signed_N_raw[t_idx] = np.round(n_continuous)

    # Causal time vector accounting for decimation
    t_frames = np.arange(n_samples) / fs_dec

    return freqs, t_frames, Z1_mag, signed_N_raw