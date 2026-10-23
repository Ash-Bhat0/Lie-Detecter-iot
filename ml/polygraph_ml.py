#!/usr/bin/env python3
"""
Real-Time ML Polygraph & Interrogation Engine
=============================================
Runs on Raspberry Pi 4 Model B (with fallback mock mode for PC testing).

Workflow:
  1. Loads the trained physiological ML model (`best_polygraph_model.pkl`).
  2. Runs baseline calibration to establish subject-specific resting BPM and GSR.
  3. Conducts real-time interrogation trials:
     - Prompts questions to the examiner
     - Records 10-second response window from MCP3008 ADC (CH0 Pulse, CH1 GSR)
     - Extracts autonomic nervous system features (HRV, BPM shift, SCR peaks, GSR delta)
     - Runs the ML classifier to predict P(Lie) and confidence verdict
     - Actuates hardware: drives VU meters via PWM and alerts buzzer on deception.
"""

import os
import sys
import time
import math
import joblib
from pathlib import Path
import numpy as np

# Hardware and Feature Extraction imports
from ml.feature_extraction import extract_features_from_window
from ml.data_collector import SensorReader, SimpleBeatDetector, DEFAULT_SAMPLE_RATE_HZ, SPI_AVAILABLE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_polygraph_model.pkl"

# Hardware Pin Definitions (BCM numbering)
PULSE_METER_GPIO = 18
GSR_METER_GPIO = 19
BUZZER_GPIO = 23

# Check gpiozero availability for Raspberry Pi PWM and Buzzer
GPIO_AVAILABLE = False
pulse_meter = None
gsr_meter = None
buzzer = None

try:
    from gpiozero import PWMOutputDevice
    pulse_meter = PWMOutputDevice(PULSE_METER_GPIO, frequency=1000, initial_value=0)
    gsr_meter = PWMOutputDevice(GSR_METER_GPIO, frequency=1000, initial_value=0)
    buzzer = PWMOutputDevice(BUZZER_GPIO, frequency=50, initial_value=0)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


def set_meter(meter_device, value_0_to_1: float):
    """Safely writes normalized PWM value to meter."""
    if GPIO_AVAILABLE and meter_device is not None:
        try:
            meter_device.value = max(0.0, min(1.0, value_0_to_1))
        except Exception:
            pass


def beep_buzzer(duration_sec: float = 0.1, count: int = 1):
    """Beeps buzzer if hardware is connected."""
    if GPIO_AVAILABLE and buzzer is not None:
        try:
            for _ in range(count):
                buzzer.value = 0.5
                time.sleep(duration_sec)
                buzzer.value = 0.0
                if count > 1:
                    time.sleep(0.08)
        except Exception:
            pass
    else:
        # Terminal beep fallback
        print("\a", end="", flush=True)


def load_trained_model():
    """Loads the trained ML model artifact."""
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Trained model artifact not found at:\n  {MODEL_PATH}")
        print("Please train the model first by running:")
        print("  python -m ml.train_model")
        return None

    try:
        artifact = joblib.load(MODEL_PATH)
        print(f"[Model Loaded] Type: {artifact.get('model_name', 'Classifier')}")
        metrics = artifact.get("metrics", {})
        print(f"  * Cross-Val Accuracy : {metrics.get('cv_accuracy', 0.0)*100:.1f}%")
        print(f"  * Cross-Val F1-Score : {metrics.get('cv_f1', 0.0):.3f}")
        return artifact
    except Exception as e:
        print(f"[ERROR] Failed to load model artifact: {e}")
        return None


def run_live_interrogation():
    """Main live interrogation loop with real-time ML prediction."""
    print("=" * 68)
    print("     KINSKI POLYGRAPH - REAL-TIME ML LIE DETECTION SYSTEM")
    print("=" * 68)

    model_artifact = load_trained_model()
    if model_artifact is None:
        return

    pipeline = model_artifact["pipeline"]
    feature_names = model_artifact["feature_names"]

    hw_status = "Physical Raspberry Pi (MCP3008 + GPIO)" if (SPI_AVAILABLE and GPIO_AVAILABLE) else "Simulation / PC Test Mode"
    print(f"Operating Mode: {hw_status}")

    # Startup Needle Sweep if hardware present
    if GPIO_AVAILABLE:
        print("Performing analog VU meter calibration sweep...")
        for v in np.linspace(0, 0.8, 15):
            set_meter(pulse_meter, v)
            set_meter(gsr_meter, v)
            time.sleep(0.03)
        set_meter(pulse_meter, 0.0)
        set_meter(gsr_meter, 0.0)
        beep_buzzer(0.05, 1)

    reader = SensorReader(mock=not SPI_AVAILABLE)

    # 1. Baseline Calibration
    print("\n" + "-" * 68)
    print("PHASE 1: BASELINE PHYSIOLOGICAL CALIBRATION (15 seconds)")
    print("Instruct subject to sit still, breathe normally, and relax.")
    print("-" * 68)
    input("Press Enter to begin calibration...")

    detector = SimpleBeatDetector()
    calib_duration = 15
    calib_samples = int(calib_duration * DEFAULT_SAMPLE_RATE_HZ)
    bpm_list = []
    gsr_list = []

    start_calib = time.time()
    for i in range(calib_samples):
        now = time.time()
        elapsed = now - start_calib
        pulse, gsr = reader.read_raw()
        beat, bpm = detector.update(pulse, elapsed)

        bpm_list.append(bpm)
        gsr_list.append(gsr)

        # Pulse meter tick on beat
        if beat:
            set_meter(pulse_meter, 0.3)
        else:
            set_meter(pulse_meter, 0.05)

        if i % (DEFAULT_SAMPLE_RATE_HZ * 2) == 0:
            print(f"  Calibrating... {calib_duration - int(elapsed):2d}s | BPM: {bpm:5.1f} | GSR: {gsr:4d}")

        target_t = start_calib + (i + 1) * (1.0 / DEFAULT_SAMPLE_RATE_HZ)
        delay = target_t - time.time()
        if delay > 0:
            time.sleep(delay)

    baseline_bpm = float(np.mean(bpm_list)) if bpm_list else 72.0
    baseline_gsr = float(np.mean(gsr_list)) if gsr_list else 400.0
    print(f"\n[Baseline Established]")
    print(f"  --> Resting Heart Rate : {baseline_bpm:.1f} BPM")
    print(f"  --> Resting Skin Cond. : {baseline_gsr:.1f} ADC units")
    beep_buzzer(0.08, 1)

    # 2. Interrogation Loop
    print("\n" + "=" * 68)
    print("PHASE 2: LIVE INTERROGATION & REAL-TIME ML INFERENCE")
    print("=" * 68)

    q_idx = 0
    trial_duration = 10  # 10s window

    while True:
        q_idx += 1
        print(f"\n--------------------------------------------------------------")
        print(f"Trial #{q_idx}")
        q_prompt = input("Enter Question to ask subject (or 'exit' to quit): ").strip()
        if q_prompt.lower() in ["exit", "quit", "q"]:
            break
        if not q_prompt:
            q_prompt = f"Question #{q_idx}"

        # In mock mode, ask examiner whether to simulate Truth or Lie for demonstration
        if not SPI_AVAILABLE:
            sim_choice = input("  [Sim Mode] Simulate subject answering with (t=Truth / l=Lie)? [default: t]: ").strip().lower()
            sim_state = "lie" if sim_choice == 'l' else "truth"
            reader.set_simulation_state(sim_state)

        print(f"\n--> ASK SUBJECT: \"{q_prompt}\"")
        input("Press Enter immediately after speaking the question to begin 10s analysis...")

        print(f"Analyzing physiological response ({trial_duration}s)...")
        trial_detector = SimpleBeatDetector()
        samples_count = int(trial_duration * DEFAULT_SAMPLE_RATE_HZ)

        pulse_arr = np.zeros(samples_count)
        gsr_arr = np.zeros(samples_count)
        bpm_arr = np.zeros(samples_count)

        start_trial = time.time()
        for i in range(samples_count):
            now = time.time()
            elapsed = now - start_trial

            pulse, gsr = reader.read_raw()
            beat, bpm = trial_detector.update(pulse, elapsed)

            pulse_arr[i] = pulse
            gsr_arr[i] = gsr
            bpm_arr[i] = bpm

            # Hardware feedback: needle reacts to instant changes
            if beat:
                set_meter(pulse_meter, 0.45)
            else:
                set_meter(pulse_meter, 0.08)

            # Map GSR relative to baseline to GSR meter needle (0 to 1)
            norm_gsr = max(0.0, min(1.0, (gsr - baseline_gsr + 100) / 400.0))
            set_meter(gsr_meter, norm_gsr)

            # Live terminal status bar
            if i % (DEFAULT_SAMPLE_RATE_HZ // 2) == 0:
                print(f"\r  [Analyzing] {elapsed:4.1f}s/{trial_duration}s | Pulse: {pulse:4d} | GSR: {gsr:4d} | BPM: {bpm:5.1f}   ", end="", flush=True)

            target_t = start_trial + (i + 1) * (1.0 / DEFAULT_SAMPLE_RATE_HZ)
            delay = target_t - time.time()
            if delay > 0:
                time.sleep(delay)

        print("\r  [Analysis Complete] Computing autonomic features...          ")

        # 3. Extract Features
        feats = extract_features_from_window(
            pulse_arr=pulse_arr,
            gsr_arr=gsr_arr,
            bpm_arr=bpm_arr,
            baseline_bpm=baseline_bpm,
            baseline_gsr=baseline_gsr,
            sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ
        )

        # Build feature vector matching trained model schema
        x_vec = np.array([[feats.get(fn, 0.0) for fn in feature_names]])

        # 4. ML Model Prediction
        pred_label = pipeline.predict(x_vec)[0]
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(x_vec)[0]
            p_lie = float(probs[1])
        else:
            p_lie = 1.0 if pred_label == 1 else 0.0

        p_truth = 1.0 - p_lie

        # 5. Determine Verdict
        if p_lie >= 0.70:
            verdict = "DECEPTION DETECTED (LIE)"
            color_mark = "[ ! DECEPTION ! ]"
            is_deceptive = True
        elif p_lie >= 0.45:
            verdict = "INCONCLUSIVE / ELEVATED STRESS"
            color_mark = "[ ? INCONCLUSIVE ? ]"
            is_deceptive = False
        else:
            verdict = "TRUTHFUL RESPONSE"
            color_mark = "[ + TRUTH + ]"
            is_deceptive = False

        # Visual confidence bar
        meter_len = 30
        filled = int(p_lie * meter_len)
        bar = "[" + "#" * filled + "-" * (meter_len - filled) + "]"

        # 6. Actuate Hardware
        if is_deceptive:
            set_meter(gsr_meter, 0.95)
            set_meter(pulse_meter, 0.85)
            beep_buzzer(0.12, 3)  # Alert buzzer bursts
        else:
            set_meter(gsr_meter, 0.20)
            set_meter(pulse_meter, 0.15)

        # 7. Print Comprehensive Forensic Report
        print("\n" + "=" * 60)
        print(f"             QUESTION RESULT: {color_mark}")
        print("=" * 60)
        print(f"  Question        : \"{q_prompt}\"")
        print(f"  VERDICT         : {verdict}")
        print(f"  Deception Prob  : {p_lie*100:5.1f}%  {bar}")
        print(f"  Truth Prob      : {p_truth*100:5.1f}%")
        print("\n  Autonomic Physiological Breakdown:")
        print(f"    * Heart Rate Shift (ΔBPM) : {feats['bpm_delta']:+5.1f} BPM  (Mean: {feats['bpm_mean']:.1f} vs Base: {baseline_bpm:.1f})")
        print(f"    * GSR Conductance Shift   : {feats['gsr_delta']:+5.1f} units (Max Δ: {feats['gsr_max_delta']:+.1f})")
        print(f"    * Phasic SCR Bursts       : {feats['gsr_scr_peaks']} peak(s) detected")
        print(f"    * Heart Rate Var. (RMSSD) : {feats['hrv_rmssd']:5.1f} ms")
        print(f"    * Autonomic Stress Index  : {feats['stress_index']:0.3f}")
        print("=" * 60)

    # Cleanup hardware on exit
    if GPIO_AVAILABLE:
        set_meter(pulse_meter, 0.0)
        set_meter(gsr_meter, 0.0)


if __name__ == "__main__":
    run_live_interrogation()
