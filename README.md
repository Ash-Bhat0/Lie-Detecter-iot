# Kinski IoT Polygraph & Physiological Lie Detector (Raspberry Pi 4 Model B)

An end-to-end IoT and Edge Machine Learning Polygraph Lie Detection system built for the Raspberry Pi 4 Model B (with cross-platform simulation mode for PC/Mac).

The system reads physiological biosignals from hardware sensors, extracts Autonomic Nervous System (ANS) features (Heart Rate Variability and Galvanic Skin Response), trains and evaluates Edge Machine Learning classifiers, and runs live interrogation sessions with real-time AI deception detection and physical VU meter / buzzer feedback.

---

## System Architecture

```text
Lie-Detecter-iot/
├── app.py                         # Unified interactive CLI launcher (Modes 1-7)
├── polygraph_pi4b.py              # Hardware driver: MCP3008 ADC + GPIO VU meters & buzzer
├── NanoLieDetector_original.ino   # Original Arduino Nano reference sketch
│
├── ml/                            # Machine Learning Pipeline
│   ├── __init__.py
│   ├── data_collector.py          # Labeled interrogation session recorder (Baseline + Trials)
│   ├── feature_extraction.py      # Autonomic Nervous System feature extractor (HRV + GSR EDA)
│   ├── train_model.py             # 5-Fold Stratified CV trainer (Random Forest, SVM, GBM)
│   ├── polygraph_ml.py            # Live real-time ML interrogation & prediction engine
│   ├── generate_synthetic_data.py # Benchmark physiological dataset generator (18 subjects)
│   └── models/                    # Exported model artifacts (.pkl & metadata.json)
│
├── data/
│   ├── raw/                       # Continuous session CSVs (Pulse, GSR, BPM, Labels)
│   └── processed/                 # Extracted trial feature tables for ML training
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Physiological Signals & Feature Extraction

Polygraph lie detection measures involuntary autonomic arousal (fight-or-flight sympathetic response) that occurs when a subject attempts deception.

The pipeline extracts **24 physiological features** across two primary biosensors:

### 1. Cardiovascular / Pulse Sensor (MCP3008 CH0)
- **$\Delta\text{BPM}$ & Peak Shifts**: Mean BPM, peak BPM, and deviation from the subject's resting baseline ($\Delta\text{BPM} = \text{BPM}_{\text{trial}} - \text{BPM}_{\text{baseline}}$).
- **Heart Rate Variability (HRV)** (derived from detected peak-to-peak RR intervals):
  - `hrv_sdnn`: Standard deviation of normal RR intervals.
  - `hrv_rmssd`: Root Mean Square of Successive Differences (parasympathetic suppression marker; drops significantly under acute deception stress).
  - `hrv_pnn50`: Percentage of successive RR differences $> 50\text{ ms}$.
- **Pulse Waveform Dynamics**: Amplitude mean, standard deviation, and systolic peak envelope.

### 2. Galvanic Skin Response / Electrodermal Activity (MCP3008 CH1)
- **Tonic Conductance (SCL)**: Mean GSR, peak GSR, difference and ratio relative to resting baseline ($\Delta\text{GSR}$).
- **Phasic Responses (SCR)**:
  - `gsr_max_slope`: Maximum positive rise slope ($d(\text{GSR})/dt$), capturing sudden sympathetic sweat gland activation.
  - `gsr_scr_peaks`: Count of distinct Skin Conductance Responses during the question window.
  - `gsr_auc`: Area Under the Curve above baseline (integrated sympathetic energy).
- **Autonomic Stress Index**: Combined normalized cardiac and electrodermal arousal score.

---

## Hardware Setup & Pinout (Raspberry Pi 4B)

The Raspberry Pi 4 Model B communicates with the MCP3008 10-bit ADC via SPI.

| MCP3008 Pin | Signal / Description | Raspberry Pi 4 Pin |
|---|---|---|
| **VDD / VREF** | 3.3 V Power & Reference | Pin 1 (3.3 V) |
| **AGND / DGND**| Ground | Pin 6 (GND) |
| **CLK** | SPI Clock (SCLK) | GPIO11 (Pin 23) |
| **DOUT** | SPI MISO | GPIO9 (Pin 21) |
| **DIN** | SPI MOSI | GPIO10 (Pin 19) |
| **CS/SHDN** | SPI Chip Select (CE0) | GPIO8 (Pin 24) |
| **CH0** | Analog Pulse Sensor Output | Analog In 0 |
| **CH1** | GSR Electrode Circuit Node | Analog In 1 |

### Actuators (PWM & Buzzer)
| Function | BCM GPIO | Physical Header Pin |
|---|---|---|
| Pulse VU Meter (PWM needle) | **GPIO18** | Physical Pin 12 |
| GSR VU Meter (PWM needle) | **GPIO19** | Physical Pin 35 |
| Warning Buzzer Driver | **GPIO23** | Physical Pin 16 |

> [!CAUTION]
> **Electrical Safety:** Always power the MCP3008 at **3.3 V** when wired directly to the Pi. Never apply 5 V logic directly to Raspberry Pi GPIOs. Analog VU meters and buzzers must be driven through appropriate transistor/MOSFET driver circuits with current-limiting resistors.

---

## Quickstart Guide

### 1. Installation

On Raspberry Pi OS:
```bash
sudo apt update
sudo apt install -y python3-spidev python3-gpiozero python3-pip
git clone https://github.com/Ash-Bhat0/Lie-Detecter-iot.git
cd Lie-Detecter-iot
pip install -r requirements.txt
```

*(On Windows / macOS, the system automatically runs in high-fidelity simulation mode if `spidev` or `gpiozero` are not available).*

### 2. Run the Main Launcher

```bash
python app.py
```

You will be presented with the interactive menu:

```text
=== KINSKI IOT POLYGRAPH & LIE DETECTION SYSTEM (Raspberry Pi 4B) ===
  1. Live Sensor Hardware Monitor (Raw Pulse/GSR & VU needle sweep)
  2. Record New Training Session (Interrogation data logger with labels)
  3. Train / Retrain Edge ML Models (Random Forest, SVM, GBM)
  4. Run Real-Time ML Lie Detector (Interrogation mode + Live AI)
  5. Generate / Refresh Benchmark Training Dataset
  6. LIAR-PLUS Text Claim Classifier (NLP Requirements)
  7. Exit
```

---

## Step-by-Step Workflow

### Step 1: Record Custom Training Data (`Option 2`)
Run `python -m ml.data_collector` to conduct an interrogation session:
1. **Calibration Phase**: The subject sits quietly for 20s to record resting baseline BPM and GSR.
2. **Question Trials**: The examiner asks questions (control or relevant questions). Tag each trial with ground truth (`0 = Truth`, `1 = Lie`).
3. Continuous 100 Hz sensor streams are automatically logged to `data/raw/polygraph_session_<subject>_<timestamp>.csv`.

*(If you don't have recorded hardware sessions yet, use `Option 5` to instantly generate an 18-subject benchmark dataset).*

### Step 2: Train & Evaluate the ML Model (`Option 3`)
Run `python -m ml.train_model`:
- Automatically aggregates all raw session CSVs and extracts 24 autonomic features.
- Evaluates candidate models via 5-Fold Stratified Cross-Validation:
  - **Random Forest Classifier** (Top Edge performer)
  - **Gradient Boosting Classifier**
  - **Support Vector Machine (RBF kernel)**
  - **Logistic Regression**
- Displays feature importance ranking and exports the best pipeline to `ml/models/best_polygraph_model.pkl`.

### Step 3: Run Real-Time ML Lie Detection (`Option 4`)
Run `python -m ml.polygraph_ml`:
1. Establishes the subject's individual resting baseline.
2. Examiner prompts a question.
3. System records a 10-second response window, extracts HRV/GSR features, and feeds them into the trained ML model.
4. Outputs real-time deception probability $P(\text{Lie})$, autonomic breakdown, and activates hardware (VU needle deflection + alert buzzer if deception is detected).

---

## Attribution & Scientific Note

- Hardware polygraph concept adapted from Roni Bandini's *Nano Lie Detector* (2020).
- **Scientific Notice:** Involuntary physiological arousal (heart rate acceleration and skin conductance responses) indicates autonomic activation, cognitive load, or stress. It is not an infallible proof of deception. Use this project as an educational IoT biosensing and Edge ML demonstration.
