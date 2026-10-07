import numpy as np
from scipy.signal import stft

def compute_stft(sig, fs=10e6):
    """
    Computes the Short-Time Fourier Transform of the raw engine signal.
    """
    # 10000 samples @ 10MHz = 1 ms window. 
    # This provides exactly 1000 Hz frequency resolution per bin.
    nperseg = 10000 
    noverlap = 5000
    
    f, t_frames, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    return f, t_frames, Zxx