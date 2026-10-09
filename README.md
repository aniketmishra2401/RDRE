# RDRE Dual-Sensor Wave Tracking & Mode Classification Pipeline

Real-time continuous wavelet transform (CWT) cross-phase extraction and Hidden Markov Model (HMM) Viterbi decoding engine for Rotating Detonation Rocket Engines (RDRE).

## 🚀 Performance Benchmarks

Evaluated on synthetic dual-sensor pressure profiles ($f_s = 10\text{ MHz}$, SNR $= 15\text{ dB}$, $\Delta\theta = 60^\circ$):

| Metric | Target Requirement | Measured Performance |
| :--- | :--- | :--- |
| **Matched Transitions** | $\ge 95\%$ | **100% (4/4)** |
| **Missed Transitions** | $0$ | **0** |
| **False Alarm Rate (FAR)** | $< 10.0\text{ events/s}$ | **0.0 events/s** |
| **Mean Event Latency** | $< 5.0\text{ ms}$ | **0.515 ms** |
| **Max Event Latency** | $< 10.0\text{ ms}$ | **0.765 ms** |

## 🏗 System Architecture

```text
Dual-Sensor Pressure Signals (10 MHz)
   │
   ├──> High-Pass Drift Calibration (SensorDelayCalibrator)
   ├──> Causal CWT Front-End & Phase Track (flattened_front_end.py)
   ├──> 10-State Emission Engine (sensor_calib_and_expanded_nodet.py)
   ├──> Inertial Viterbi HMM Decoder (signed_wave_tracker.py)
   └──> Event Metric Evaluator (event_evaluator.py)
