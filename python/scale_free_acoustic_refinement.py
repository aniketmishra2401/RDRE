import numpy as np
from scipy.signal import decimate

class ScaleFreeAcousticDiscriminator:
    def __init__(
        self,
        D_mean_m: float = 0.083,          # Mean combustor diameter (83 mm)
        D_cj_m_per_s: float = 1980.0,      # CJ detonation velocity (m/s)
        delta_theta_deg: float = 60.0
    ):
        self.D_mean = D_mean_m
        self.D_cj = D_cj_m_per_s
        self.delta_theta_rad = np.radians(delta_theta_deg)
        
        # Issue 4 Fix: Calculate fundamental CJ rotation frequency
        self.f_cj = self.D_cj / (np.pi * self.D_mean)
        
        # Non-dimensional frequency grid: 0.25 f_CJ to 4.5 f_CJ (100 bins)
        self.freqs_coarse = np.linspace(0.25 * self.f_cj, 4.5 * self.f_cj, 100)
        self.w0_coarse = 6.0
        self.w0_fine = 24.0  # Issue 3 Fix: High-resolution kernel

    def compute_coarse_bank(
        self,
        sig1_dec: np.ndarray,
        sig2_dec: np.ndarray,
        fs_dec: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast coarse CWT scan (w0 = 6.0) across the dimensionless frequency bank."""
        n_samples = len(sig1_dec)
        Z1_coarse = np.zeros((len(self.freqs_coarse), n_samples), dtype=complex)
        Z2_coarse = np.zeros((len(self.freqs_coarse), n_samples), dtype=complex)

        for idx, f_target in enumerate(self.freqs_coarse):
            scale = self.w0_coarse / (2 * np.pi * f_target)
            M = int(10 * scale * fs_dec)
            if M % 2 == 0: M += 1
            
            t_wavelet = (np.arange(M) - M // 2) / fs_dec
            norm_L1 = 1.0 / scale
            wavelet = norm_L1 * np.exp(1j * 2 * np.pi * f_target * t_wavelet) * np.exp(-0.5 * (t_wavelet / scale) ** 2)
            
            pad_len = M - 1
            s1_pad = np.pad(sig1_dec, (pad_len, 0), mode='constant')
            s2_pad = np.pad(sig2_dec, (pad_len, 0), mode='constant')
            
            Z1_coarse[idx, :] = np.convolve(s1_pad, np.conj(wavelet[::-1]), mode='valid')[:n_samples]
            Z2_coarse[idx, :] = np.convolve(s2_pad, np.conj(wavelet[::-1]), mode='valid')[:n_samples]
            
        return Z1_coarse, Z2_coarse

    def refine_and_discriminate(
        self,
        sig1_dec: np.ndarray,
        sig2_dec: np.ndarray,
        fs_dec: float,
        candidate_f_peak: float,
        t_frame_idx: int
    ) -> tuple[float, float, str]:
        """
        Executes narrow high-resolution refinement (w0 = 24) around candidate peak
        and computes harmonic coherence C_harm to reject acoustic sloshing modes.
        """
        # Define 6 narrow channels around candidate peak (±10% span)
        freqs_fine = np.linspace(candidate_f_peak * 0.90, candidate_f_peak * 1.10, 6)
        
        Z1_fine = np.zeros(len(freqs_fine), dtype=complex)
        Z2_fine = np.zeros(len(freqs_fine), dtype=complex)
        
        for idx, f_target in enumerate(freqs_fine):
            scale = self.w0_fine / (2 * np.pi * f_target)
            M = int(10 * scale * fs_dec)
            if M % 2 == 0: M += 1
            
            t_wavelet = (np.arange(M) - M // 2) / fs_dec
            norm_L1 = 1.0 / scale
            wavelet = norm_L1 * np.exp(1j * 2 * np.pi * f_target * t_wavelet) * np.exp(-0.5 * (t_wavelet / scale) ** 2)
            
            # Local window convolution around time index
            start_i = max(0, t_frame_idx - M)
            end_i = min(len(sig1_dec), t_frame_idx + M)
            
            s1_sub = sig1_dec[start_i:end_i]
            s2_sub = sig2_dec[start_i:end_i]
            
            if len(s1_sub) >= M:
                c1 = np.convolve(s1_sub, np.conj(wavelet[::-1]), mode='same')
                c2 = np.convolve(s2_sub, np.conj(wavelet[::-1]), mode='same')
                Z1_fine[idx] = c1[len(c1) // 2]
                Z2_fine[idx] = c2[len(c2) // 2]

        # Extract refined peak frequency
        refined_f_idx = np.argmax(np.abs(Z1_fine))
        refined_f_peak = freqs_fine[refined_f_idx]
        
        # Fundamental phase
        phi_1 = np.angle(Z1_fine[refined_f_idx] * np.conj(Z2_fine[refined_f_idx]))
        
        # Evaluate 2nd and 3rd Harmonic Coherence (C_harm)
        f_h2 = 2.0 * refined_f_peak
        f_h3 = 3.0 * refined_f_peak
        
        mag_fund = np.abs(Z1_fine[refined_f_idx])
        
        # Simple local harmonic magnitude and phase estimates
        scale_h2 = self.w0_coarse / (2 * np.pi * f_h2)
        M_h2 = int(10 * scale_h2 * fs_dec)
        if M_h2 % 2 == 0: M_h2 += 1
        t_w2 = (np.arange(M_h2) - M_h2 // 2) / fs_dec
        w2 = (1.0 / scale_h2) * np.exp(1j * 2 * np.pi * f_h2 * t_w2) * np.exp(-0.5 * (t_w2 / scale_h2) ** 2)
        
        sub1 = sig1_dec[max(0, t_frame_idx - M_h2): min(len(sig1_dec), t_frame_idx + M_h2)]
        sub2 = sig2_dec[max(0, t_frame_idx - M_h2): min(len(sig2_dec), t_frame_idx + M_h2)]
        
        if len(sub1) >= M_h2:
            z1_h2 = np.convolve(sub1, np.conj(w2[::-1]), mode='same')[len(sub1)//2]
            z2_h2 = np.convolve(sub2, np.conj(w2[::-1]), mode='same')[len(sub2)//2]
            mag_h2 = np.abs(z1_h2)
            phi_2 = np.angle(z1_h2 * np.conj(z2_h2))
        else:
            mag_h2, phi_2 = 0.0, 0.0

        # Phase deviation: Δφ2 = |φ2 - 2*φ1|
        d_phi2 = np.abs(np.arctan2(np.sin(phi_2 - 2 * phi_1), np.cos(phi_2 - 2 * phi_1)))
        
        # Harmonic Coherence Score
        C_harm = (mag_h2 / (mag_fund + 1e-9)) * np.cos(d_phi2)
        
        # Decision logic: Detonation requires strong locked harmonics
        if C_harm > 0.15:
            classification = "DETONATION"
        else:
            classification = "ACOUSTIC_MODE"
            
        return refined_f_peak, C_harm, classification