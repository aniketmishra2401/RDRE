# RDRE Monitoring and Signal Analysis Testbench

A Python-based research prototype for signal processing, wave-state estimation, and transition analysis for a **Rotating Detonation Rocket Engine (RDRE)** monitoring system.

The project provides a software testbench for processing simulated sensor signals, examining wave-related signal characteristics, comparing estimated states against programmed reference labels, and generating plots and evaluation reports.

> **Status: Prototype under development.** The current benchmark uses synthetic signals. Performance has not yet been validated using measurements from a physical RDRE.

## 1. Objectives

The project aims to develop and evaluate a signal-processing workflow for RDRE monitoring. Its current objectives are to:

- Process simulated sensor signals.
- Analyse signal characteristics in the time and frequency domains.
- Extract phase-related information.
- Estimate signed wave states.
- Compare predicted states against programmed reference labels.
- Generate preliminary transition-detection statistics.
- Export figures and structured data for engineering analysis.
- Establish a foundation for future validation using experimental sensor measurements.

## 2. Processing Workflow

The current software workflow comprises the following stages:

1. Synthetic sensor signal generation.
2. Signal conditioning and preprocessing.
3. Signal feature extraction and analysis.
4. Wave-state estimation.
5. Comparison with available reference labels.
6. Preliminary event evaluation.
7. Figure generation and data export.

The repository also contains supporting modules for phase extraction, wave tracking, sensor calibration, and related signal-processing development. Not every module is necessarily integrated into the main pipeline.

## 3. Technology Stack

- Python
- NumPy
- SciPy
- Matplotlib
- CSV and JSON for data export

## 4. Repository Structure

The repository includes the Python processing pipeline, supporting modules, test data, and reporting resources.

```text
RDRE/
├── python/
│   ├── run_pipeline.py
│   ├── reporting.py
│   ├── flattened_front_end.py
│   ├── scale_free_acoustic_refinement.py
│   ├── causal_phase_extractor.py
│   ├── signed_wave_tracker.py
│   ├── event_evaluator.py
│   ├── signal_sythesis.py
│   └── sensor_calib_and_expanded_nodet.py
├── test_vectors/
├── results/
├── requirements.txt
├── README.md
└── FIGURE_OUTPUTS.md
```

*The directory tree is representative. Refer to the actual repository for the complete file structure.*

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/aniketmishra2401/RDRE.git
cd RDRE
```

Create a virtual environment.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

Alternatively, if `requirements.txt` accurately lists the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Pipeline

Navigate to the Python source directory:

```bash
cd python
```

Run the main pipeline:

```bash
python run_pipeline.py
```

The pipeline executes the configured synthetic-data workflow and generates the outputs enabled in the current implementation.

Output locations and filenames may depend on the repository version and configuration. Consult `FIGURE_OUTPUTS.md` for additional information about the reporting outputs.

## 7. Current Synthetic Test Configuration

The initial benchmark uses the following configuration:

| Parameter | Value |
|---|---:|
| Observation duration | 0.2 seconds |
| Original sampling frequency | 10 MHz |
| Target processing sampling frequency | 200 kHz |
| Configured signal-to-noise ratio | 15 dB |
| Simulated sensor angular spacing | 60° |
| Random seed | 42 |
| Programmed reference transitions | 4 |

The programmed sequence contains five successive intervals with assigned wave states and directions. The sequence is used as a controlled reference for examining the pipeline's predicted signed wave-state labels.

These values describe the current synthetic benchmark and are not specifications or validated performance characteristics of a physical engine.

## 8. Generated Outputs

The reporting workflow generates graphical and machine-readable outputs for inspection and further analysis.

| Output | Description |
|---|---|
| `fig01_architecture` | Processing architecture diagram |
| `fig02_dual_sensor_input` | Simulated sensor signals and a selected transition detail |
| `fig03_time_frequency_phase` | Time-frequency, phase, and signal-power representations |
| `fig04_state_comparison` | Comparison of programmed reference and predicted signed states |
| `fig05_full_state_timeline` | Timeline of predicted states |
| `fig06_event_counts` | Preliminary event-evaluation summary |
| `dual_sensor_input.csv` | Simulated sensor samples |
| `frame_features_and_states.csv` | Frame-level features and predicted state estimates |
| `run_summary.json` | Run configuration and preliminary evaluation summary |

Figures are exported as PNG and PDF files where supported by the reporting module.

See [`FIGURE_OUTPUTS.md`](FIGURE_OUTPUTS.md) for figure descriptions and reporting details.

## 9. Preliminary Benchmark Results

The current synthetic test run produced the following event-evaluator outputs:

| Metric | Reported value |
|---|---:|
| Programmed reference transitions | 4 |
| Matched transitions | 4 |
| Missed transitions | 0 |
| False alarms | 0 |
| Mean reported transition-timing difference | 0.515 ms |
| Maximum reported transition-timing difference | 0.765 ms |

**Important:** These values are preliminary outputs from the current event evaluator. Transition matching and state-label handling require further verification. They must not be interpreted as independently validated detection accuracy, a confirmed false-alarm rate, or real-time response latency.

The timing differences represent retrospective comparisons between estimated and reference event timestamps, not complete computational or end-to-end latency.

## 10. Current Limitations

The current implementation has several limitations:

- **Synthetic data:** The benchmark does not establish performance on a physical RDRE.
- **Simplified signal generation:** The generated signals do not comprehensively reproduce combustion physics, injector dynamics, chamber acoustics, or real sensor effects.
- **Reference-label coverage:** The available reference sequence primarily represents signed wave states rather than independently verified ground truth for every possible operating state.
- **Event evaluation:** Transition matching and state-label handling require further verification.
- **Real-time operation:** Retrospective timing comparisons do not establish causal processing or end-to-end latency.
- **Hardware validation:** The current software benchmark does not establish performance on a physical sensor array, embedded processor, or FPGA.

These limitations should be considered when interpreting the generated figures and numerical results.

## 11. Planned Development

Potential future work includes:

1. Improving and verifying event-matching logic.
2. Adding independently labelled reference data for supported operating states.
3. Testing robustness across different noise levels, signal frequencies, channel delays, and sensor configurations.
4. Comparing the pipeline against appropriate baseline methods.
5. Investigating phase ambiguity and sensor-array observability.
6. Evaluating confidence scores and rejection behaviour.
7. Measuring computational runtime and end-to-end latency separately from retrospective event timing.
8. Validating the software using experimental sensor measurements.
9. Investigating embedded or FPGA deployment after software verification.

These are proposed development activities, not claims that the corresponding capabilities have already been validated.

## 12. Research Scope and Intended Use

This repository is intended for research, prototyping, and engineering evaluation of RDRE-related signal-processing and monitoring methods.

The current results document software behaviour under controlled synthetic conditions. They do not replace experimental engine testing, independent validation, or safety qualification.

Any technical or patent documentation based on this project should distinguish implemented features from planned work and should substantiate performance claims with appropriate test evidence.

## 13. Contributing

Potential contributions include:

- Signal-processing improvements.
- Additional synthetic or experimental datasets.
- Verified evaluation procedures.
- Robustness and baseline testing.
- Documentation and reproducibility improvements.

Contributions should include suitable tests and clearly identify whether results were obtained using synthetic or experimental data.

## 14. License

No license is specified in this README. Add a `LICENSE` file once the project owners have decided how the source code may be used, modified, and redistributed. Until then, do not assume unrestricted reuse.

## 15. Acknowledgements and References

Relevant research publications and prior art concerning rotating detonation, wave detection, acoustic sensing, and engine monitoring should be documented as the technical investigation progresses.


