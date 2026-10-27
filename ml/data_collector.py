#!/usr/bin/env python3
"""
Data Collector for IoT Physiological Lie Detector (Pulse + GSR)
================================================================
This script guides an interrogation session to record labeled physiological data:
  1. Calibration / Resting baseline phase (establishes baseline BPM & GSR).
  2. Question-by-question response trials (prompts questions, records sensor streams,
     and tags each trial with ground truth: 'truth', 'lie', or 'control').

Hardware Support:
  - Raspberry Pi 4B with MCP3008 ADC (SPI bus 0, CE0):
      CH0 = Pulse Sensor
      CH1 = GSR Sensor
  - Simulation / Mock Mode: Auto-activates if spidev is unavailable (for testing on PC/Mac).
"""

import os
import sys
import time
import math
import random
import serial
import csv
from datetime import datetime
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Configuration (MCP3008 + Raspberry Pi)
PULSE_CHANNEL = 0
GSR_CHANNEL = 1
DEFAULT_SAMPLE_RATE_HZ = 100  # 100 Hz sampling = 10ms per sample
DEFAULT_QUESTION_DURATION_SEC = 10  # 10 seconds per question window
DEFAULT_BASELINE_DURATION_SEC = 20  # 20 seconds baseline calibration

# Check SPI hardware availability
SPI_AVAILABLE = False
try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1_000_000
    SPI_AVAILABLE = True
except Exception:
    SPI_AVAILABLE = False


class SensorReader:
    """Reads Pulse from ESP8266 (MAX30102) over USB Serial, with fallback simulation."""

    def __init__(self, mock: bool = False, serial_port: str = "/dev/ttyUSB0", baud_rate: int = 115200):
        self.mock = mock
        self.ser = None
        self.last_pulse = 512
        
        # Simulation variables (used only if ESP8266 is disconnected)
        self.sim_phase = 0.0
        self.sim_base_bpm = 72.0
        self.sim_base_gsr = 350.0
        self.current_state = "neutral"
        self.sim_stress_factor = 0.0

        # Try connecting to the ESP8266 over USB
        if not self.mock:
            try:
                self.ser = serial.Serial(serial_port, baud_rate, timeout=0.05)
                print(f"[Hardware] Connected to ESP8266 Pulse Sensor on {serial_port}")
            except Exception as e:
                print(f"[Notice] Could not open {serial_port} ({e}). Running in Simulation Mode.")
                self.mock = True

    def set_simulation_state(self, state: str):
        """Used in mock mode to simulate physiological arousal for truth vs lie."""
        self.current_state = state
        if state == "lie":
            self.sim_stress_factor = random.uniform(0.65, 1.0)
        elif state == "truth":
            self.sim_stress_factor = random.uniform(0.05, 0.25)
        else:
            self.sim_stress_factor = 0.0

    def read_raw(self):
        """Returns tuple of (pulse_raw [0-1023], gsr_raw [0-1023])."""
        # --- 1. PHYSICAL HARDWARE MODE (ESP8266 + MAX30102) ---
        if not self.mock and self.ser is not None:
            try:
                # Read incoming lines from ESP8266 (format: "IR:98421\n")
                while self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if "IR:" in line:
                        raw_ir = int(line.split("IR:")[1])
                        if raw_ir > 50000:
                            # Normalize 18-bit MAX30102 IR reflection into 0-1023 range
                            # Centers typical resting finger pulse around ~512
                            normalized = int(512 + (raw_ir - 90000) / 100.0)
                            self.last_pulse = max(0, min(1023, normalized))

                # Baseline GSR value (350-450) until your physical GSR arrives
                gsr_val = 400
                return self.last_pulse, gsr_val
            except Exception as e:
                print(f"[Warning] Serial read error ({e}), switching to simulation.")
                self.mock = True

        # --- 2. FALLBACK SIMULATION MODE (If ESP8266 is unplugged) ---
        target_bpm = self.sim_base_bpm + (22.0 * self.sim_stress_factor)
        freq = target_bpm / 60.0
        self.sim_phase += (2.0 * math.pi * freq) * (1.0 / DEFAULT_SAMPLE_RATE_HZ)

        pulse_wave = (math.sin(self.sim_phase) + 0.35 * math.sin(2.0 * self.sim_phase))
        pulse_raw = int(512 + 180 * pulse_wave + random.uniform(-10, 10))
        pulse_raw = max(0, min(1023, pulse_raw))

        target_gsr = self.sim_base_gsr + (350.0 * self.sim_stress_factor)
        self.sim_base_gsr += 0.01 * (target_gsr - self.sim_base_gsr)
        gsr_raw = int(self.sim_base_gsr + random.uniform(-4, 4))
        gsr_raw = max(0, min(1023, gsr_raw))

        return pulse_raw, gsr_raw

        # Mock / Simulation mode
        # Simulate realistic cardiac waveform with stress modulation
        target_bpm = self.sim_base_bpm + (22.0 * self.sim_stress_factor)
        freq = target_bpm / 60.0
        self.sim_phase += (2.0 * math.pi * freq) * (1.0 / DEFAULT_SAMPLE_RATE_HZ)

        # Cardiac pulse waveform approximation (systolic peak + dicrotic notch)
        pulse_wave = (math.sin(self.sim_phase) + 0.35 * math.sin(2.0 * self.sim_phase))
        pulse_raw = int(512 + 180 * pulse_wave + random.uniform(-10, 10))
        pulse_raw = max(0, min(1023, pulse_raw))

        # GSR simulation: tonic conductance level + phasic sympathetic surge during deception
        target_gsr = self.sim_base_gsr + (350.0 * self.sim_stress_factor)
        # Slow drift towards target + noise
        self.sim_base_gsr += 0.01 * (target_gsr - self.sim_base_gsr)
        gsr_raw = int(self.sim_base_gsr + random.uniform(-4, 4))
        gsr_raw = max(0, min(1023, gsr_raw))

        return pulse_raw, gsr_raw


class SimpleBeatDetector:
    """Detects heartbeats from raw pulse sensor stream to estimate instant BPM."""

    def __init__(self, threshold: int = 510, refractory_sec: float = 0.30):
        self.threshold = threshold
        self.refractory_sec = refractory_sec
        self.last_beat_time = 0.0
        self.was_above = False
        self.current_bpm = 72.0
        self.rr_intervals = []

    def update(self, sample_value: int, current_time: float):
        above = sample_value >= self.threshold
        beat_detected = (above and not self.was_above and (current_time - self.last_beat_time) >= self.refractory_sec)
        self.was_above = above

        if beat_detected:
            if self.last_beat_time > 0:
                interval = current_time - self.last_beat_time
                if 0.33 <= interval <= 1.5:  # Valid human heart rate (40 to 180 BPM)
                    bpm = 60.0 / interval
                    self.current_bpm = 0.8 * self.current_bpm + 0.2 * bpm
                    self.rr_intervals.append(interval)
            self.last_beat_time = current_time
            return True, self.current_bpm

        return False, self.current_bpm


def record_baseline(reader: SensorReader, duration_sec: int = DEFAULT_BASELINE_DURATION_SEC):
    """Measures resting baseline physiological parameters before testing."""
    print(f"\n[1/2] Establishing Baseline Calibration ({duration_sec} seconds)...")
    print("      Instruct the subject to sit comfortably, breathe normally, and relax.")
    input("Press Enter to begin baseline recording...")

    reader.set_simulation_state("neutral")
    detector = SimpleBeatDetector()
    sample_interval = 1.0 / DEFAULT_SAMPLE_RATE_HZ
    samples_count = int(duration_sec * DEFAULT_SAMPLE_RATE_HZ)

    pulse_vals = []
    gsr_vals = []
    bpm_vals = []

    start_time = time.monotonic()
    for i in range(samples_count):
        now = time.monotonic()
        t = now - start_time
        pulse, gsr = reader.read_raw()
        beat, bpm = detector.update(pulse, t)

        pulse_vals.append(pulse)
        gsr_vals.append(gsr)
        bpm_vals.append(bpm)

        # Progress indicator
        if i % (DEFAULT_SAMPLE_RATE_HZ * 2) == 0:
            remaining = duration_sec - int(t)
            print(f"      Calibrating... {remaining:2d}s remaining | BPM: {bpm:5.1f} | GSR: {gsr:4d}")

        elapsed = time.monotonic() - now
        sleep_time = sample_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    baseline_bpm = sum(bpm_vals) / len(bpm_vals) if bpm_vals else 72.0
    baseline_gsr = sum(gsr_vals) / len(gsr_vals) if gsr_vals else 400.0

    print(f"\n[Baseline Established]")
    print(f"  * Baseline Resting BPM : {baseline_bpm:.1f}")
    print(f"  * Baseline GSR Level   : {baseline_gsr:.1f}")
    return baseline_bpm, baseline_gsr


def run_data_collection_session():
    """Runs a full labeled interrogation session with questions and saves CSV data."""
    print("=" * 65)
    print("     IoT POLYGRAPH SENSOR DATA RECORDER & CALIBRATION")
    print("=" * 65)

    mode_str = "Physical MCP3008 Hardware (Raspberry Pi)" if SPI_AVAILABLE else "Simulation / Mock Mode"
    print(f"Active Mode: {mode_str}")

    subject_id = input("\nEnter Subject ID / Name (e.g. subject_01): ").strip() or f"sub_{int(time.time())}"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = RAW_DATA_DIR / f"polygraph_session_{subject_id}_{session_id}.csv"

    reader = SensorReader(mock=not SPI_AVAILABLE)

    # 1. Baseline Calibration
    baseline_bpm, baseline_gsr = record_baseline(reader)

    # 2. Setup Question Protocol
    default_questions = [
        {"id": "Q1", "text": "Is today a weekday?", "type": "control", "expected": 0},
        {"id": "Q2", "text": "Did you take money from the secure drawer?", "type": "relevant", "expected": 1},
        {"id": "Q3", "text": "Are you sitting in a chair right now?", "type": "control", "expected": 0},
        {"id": "Q4", "text": "Did you alter the test results secretly?", "type": "relevant", "expected": 1},
        {"id": "Q5", "text": "Have you ever been outside this building?", "type": "control", "expected": 0},
        {"id": "Q6", "text": "Did you break the hardware component intentionally?", "type": "relevant", "expected": 1},
    ]

    print("\n[2/2] Interrogation Protocol Phase")
    print("You can use default preset questions or enter custom questions on the fly.")
    use_preset = input("Use standard preset questions? (Y/n): ").strip().lower() != 'n'

    questions_list = default_questions if use_preset else []

    # CSV File Setup
    headers = [
        "timestamp_ms", "subject_id", "session_id", "baseline_bpm", "baseline_gsr",
        "question_id", "question_text", "question_type", "ground_truth_label",
        "pulse_raw", "gsr_raw", "instant_bpm"
    ]

    records = []

    trial_idx = 0
    while True:
        trial_idx += 1
        if use_preset and trial_idx <= len(questions_list):
            q_info = questions_list[trial_idx - 1]
            q_id = q_info["id"]
            q_text = q_info["text"]
            q_type = q_info["type"]
            default_label = q_info["expected"]
        else:
            print(f"\n--- Question #{trial_idx} ---")
            q_text = input("Enter Question text (or 'done' to finish session): ").strip()
            if q_text.lower() == 'done':
                break
            q_id = f"Q{trial_idx}"
            q_type = "relevant" if trial_idx % 2 == 0 else "control"
            default_label = 0 if q_type == "control" else 1

        print(f"\n--------------------------------------------------------------")
        print(f"Trial #{trial_idx} [{q_id}]: \"{q_text}\"")
        label_input = input(f"Ground Truth Label for this answer [0=Truth, 1=Lie] (default {default_label}): ").strip()
        label = int(label_input) if label_input in ["0", "1"] else default_label
        label_name = "LIE" if label == 1 else "TRUTH"

        print(f"--> Question queued. Label: {label_name} ({label})")
        input("Ask the question to the subject, then Press Enter to record response (10s)...")

        # Set simulation state if in mock mode
        reader.set_simulation_state("lie" if label == 1 else "truth")

        detector = SimpleBeatDetector()
        sample_interval = 1.0 / DEFAULT_SAMPLE_RATE_HZ
        samples_count = int(DEFAULT_QUESTION_DURATION_SEC * DEFAULT_SAMPLE_RATE_HZ)

        start_time = time.time()
        print(f"Recording sensor stream for {DEFAULT_QUESTION_DURATION_SEC} seconds...")

        for i in range(samples_count):
            now_epoch = time.time()
            elapsed_sec = now_epoch - start_time
            pulse, gsr = reader.read_raw()
            _, bpm = detector.update(pulse, elapsed_sec)

            records.append({
                "timestamp_ms": round(elapsed_sec * 1000, 2),
                "subject_id": subject_id,
                "session_id": session_id,
                "baseline_bpm": round(baseline_bpm, 2),
                "baseline_gsr": round(baseline_gsr, 2),
                "question_id": q_id,
                "question_text": q_text,
                "question_type": q_type,
                "ground_truth_label": label,
                "pulse_raw": pulse,
                "gsr_raw": gsr,
                "instant_bpm": round(bpm, 2)
            })

            # Real-time console update every 1 second
            if i % DEFAULT_SAMPLE_RATE_HZ == 0:
                print(f"  [{int(elapsed_sec)}s/{DEFAULT_QUESTION_DURATION_SEC}s] Pulse: {pulse:4d} | GSR: {gsr:4d} | BPM: {bpm:5.1f}")

            # Timing control
            target_next = start_time + (i + 1) * sample_interval
            delay = target_next - time.time()
            if delay > 0:
                time.sleep(delay)

        print(f"[Done] Recorded trial #{trial_idx} ({label_name}).")

        if use_preset and trial_idx >= len(questions_list):
            more = input("\nAll preset questions recorded. Add more custom questions? (y/N): ").strip().lower()
            if more != 'y':
                break

    # Save to CSV
    if records:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)

        print("\n" + "=" * 65)
        print(f"SUCCESS! Session saved to:")
        print(f"--> {output_csv}")
        print(f"Total samples recorded: {len(records)} across {trial_idx} questions.")
        print("=" * 65)
        return str(output_csv)
    else:
        print("\nNo records saved.")
        return None


if __name__ == "__main__":
    run_data_collection_session()
