import numpy as np
from scipy.signal import butter, filtfilt

class SensorDelayCalibrator:
    def __init__(self, fs: float = 200e3, fc_highpass: float = 500.0):
        self.fs = fs
        self.fc_highpass = fc_highpass
        self.tau_delay = 0.0
        b, a = butter(2, self.fc_highpass / (0.5 * self.fs), btype='high')
        self.b = b
        self.a = a

    def remove_thermal_drift(self, sig: np.ndarray) -> np.ndarray:
        if len(sig) < 15:
            return sig - np.mean(sig)
        return filtfilt(self.b, self.a, sig)

    def calibrate_online(self, f_w1_hz: float, phi12_raw_rad: float, delta_theta_rad: float) -> float:
        phase_error = phi12_raw_rad - delta_theta_rad
        phase_error = np.arctan2(np.sin(phase_error), np.cos(phase_error))
        self.tau_delay = phase_error / (2 * np.pi * f_w1_hz)
        return self.tau_delay

    def correct_phase(self, f_hz: float, phi12_raw_rad: float) -> float:
        phi_corrected = phi12_raw_rad - (2 * np.pi * f_hz * self.tau_delay)
        return np.arctan2(np.sin(phi_corrected), np.cos(phi_corrected))


class ExpandedStateObservationEngine:
    STATES = [
        "QUENCH", "DEFLAGRATION", "AXIAL_PULSE",
        "W1+", "W1-", "W2+", "W2-", "W3+", "W3-", "REV"
    ]

    def __init__(self, delta_theta_deg: float = 60.0):
        self.delta_theta_rad = np.radians(delta_theta_deg)

    def evaluate_emissions(
        self,
        n_cont: float,
        phi12_rad: float,
        total_power: float,
        p_quench_thresh: float,
        C_harm: float
    ) -> np.ndarray:
        log_B = np.zeros(10)
        
        # 1. QUENCH State (Zero-power floor)
        if total_power < p_quench_thresh:
            log_B[0] = 0.0
            log_B[1:] = -15.0
            return log_B
        else:
            log_B[0] = -15.0

        # 2. DEFLAGRATION State (Low acoustic coherence)
        if C_harm < 0.15:
            log_B[1] = 0.0
            log_B[2:] = -15.0
            return log_B
        else:
            log_B[1] = -15.0

        # 3. AXIAL PULSE State (Strict in-phase zero-wave condition)
        if abs(n_cont) < 0.15:
            log_B[2] = 0.0
        else:
            log_B[2] = -15.0

        # 4. REVERSAL Transient State (Penalized during normal active wave modes)
        log_B[9] = -15.0

        # 5. Detonation Wave States (W1+, W1-, W2+, W2-, W3+, W3-)
        target_N = [None, None, None, +1, -1, +2, -2, +3, -3, None]
        for idx in range(3, 9):
            target = target_N[idx]
            err = (n_cont - target) ** 2
            log_B[idx] = -0.5 * (err / 0.5)

        return log_B
