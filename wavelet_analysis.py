import numpy as np
from scipy.signal import decimate

def compute_wavelet(sig, fs=10e6, target_fs=200e3):
    """
    Computes Continuous Wavelet Transform (CWT) using a custom Morlet wavelet kernel.
    Compatible with SciPy 1.15+ (Python 3.13+).
    
    1. Decimates signal from 10 MHz -> 200 kHz for fast computation.
    2. Builds scale-normalized complex Morlet wavelets across 2 kHz - 42 kHz.
    3. Performs 1D time-domain convolutions to build the scalogram matrix.
    """
    # 1. Downsample signal to reduce computational complexity
    decimation_factor = max(1, int(fs / target_fs))
    sig_dec = decimate(sig, decimation_factor) if decimation_factor > 1 else sig
    fs_dec = fs / decimation_factor

    # 2. Define frequency range covering 1-wave through 8-wave states (5 kHz - 40 kHz)
    freqs = np.linspace(2000, 42000, 100)
    w0 = 6.0
    
    n_samples = len(sig_dec)
    Zxx_wavelet = np.zeros((len(freqs), n_samples))

    # 3. Compute Morlet CWT via convolution for each target frequency
    for idx, f_target in enumerate(freqs):
        # Determine wavelet scale in seconds
        scale_sec = w0 / (2 * np.pi * f_target)
        
        # Define window length (10 sigma span)
        M = int(10 * scale_sec * fs_dec)
        if M % 2 == 0:
            M += 1
            
        t_wavelet = (np.arange(M) - M // 2) / fs_dec
        
        # Complex Morlet formula with energy normalization (1 / sqrt(scale))
        norm = (1.0 / np.sqrt(scale_sec)) * (np.pi ** -0.25)
        wavelet = norm * np.exp(1j * 2 * np.pi * f_target * t_wavelet) * np.exp(-0.5 * (t_wavelet / scale_sec) ** 2)
        
        # Convolve decimated signal with complex conjugate of wavelet
        conv = np.convolve(sig_dec, np.conj(wavelet[::-1]), mode='same')
        Zxx_wavelet[idx, :] = np.abs(conv)

    # 4. Generate matching time vector for decimated frames
    t_frames = np.arange(n_samples) / fs_dec

    return freqs, t_frames, Zxx_wavelet