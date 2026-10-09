import numpy as np
from scipy.signal import decimate, medfilt

def compute_flattened_cwt_with_noise_gate(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float = 10e6,
    target_fs: float = 200e3,
    delta_theta_deg: float = 60.0,
    w0: float = 6.0
):
    delta_theta_rad = np.radians(delta_theta_deg)
    
    dec_factor = max(1, int(fs / target_fs))
    sig1_dec = decimate(sig1, dec_factor) if dec_factor > 1 else sig1
    sig2_dec = decimate(sig2, dec_factor) if dec_factor > 1 else sig2
    fs_dec = fs / dec_factor
    
    n_samples = len(sig1_dec)
    freqs = np.linspace(2000, 42000, 100)
    
    Z1_complex = np.zeros((len(freqs), n_samples), dtype=complex)
    Z2_complex = np.zeros((len(freqs), n_samples), dtype=complex)
    
    for idx, f_target in enumerate(freqs):
        scale_sec = w0 / (2 * np.pi * f_target)
        M = int(10 * scale_sec * fs_dec)
        if M % 2 == 0: M += 1
            
        t_wavelet = (np.arange(M) - M // 2) / fs_dec
        norm_L1 = 1.0 / scale_sec
        wavelet = norm_L1 * np.exp(1j * 2 * np.pi * f_target * t_wavelet) * np.exp(-0.5 * (t_wavelet / scale_sec) ** 2)
        
        pad_len = M - 1
        s1_pad = np.pad(sig1_dec, (pad_len, 0), mode='constant')
        s2_pad = np.pad(sig2_dec, (pad_len, 0), mode='constant')
        
        Z1_complex[idx, :] = np.convolve(s1_pad, np.conj(wavelet[::-1]), mode='valid')[:n_samples]
        Z2_complex[idx, :] = np.convolve(s2_pad, np.conj(wavelet[::-1]), mode='valid')[:n_samples]

    Z1_flat = np.abs(Z1_complex)
    frame_power = np.mean(Z1_flat ** 2, axis=0)
    
    cross_spectrum = Z1_complex * np.conj(Z2_complex)
    peak_freq_indices = np.argmax(Z1_flat, axis=0)
    
    raw_phi12 = np.zeros(n_samples)
    for t_idx in range(n_samples):
        f_idx = peak_freq_indices[t_idx]
        raw_phi12[t_idx] = np.angle(cross_spectrum[f_idx, t_idx])

    # 21-frame median filter (105 μs window) for noise-immune phase tracking
    smooth_phi12 = medfilt(raw_phi12, kernel_size=21)

    # Continuous wave ratio (un-rounded to eliminate quantization chatter)
    n_cont = smooth_phi12 / delta_theta_rad

    t_frames = np.arange(n_samples) / fs_dec
    return freqs, t_frames, Z1_flat, Z1_complex, Z2_complex, n_cont, smooth_phi12, frame_power
