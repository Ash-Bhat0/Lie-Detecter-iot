#!/usr/bin/env python3
"""
Realistic Polygraph Benchmark Dataset Generator
================================================
Generates scientifically calibrated physiological sessions across multiple subjects.
Each subject has:
  - Unique resting baseline BPM (60-85 BPM)
  - Unique resting baseline GSR (250-550 ADC units)
  - Inter-subject variability in sympathetic reactivity
  - Control / Truth trials (minimal change, natural respiration sinus arrhythmia)
  - Deceptive / Lie trials (BPM surge of 10-25%, skin conductance rise, SCR bursts, HRV drop)

Saves both raw time-series CSVs and aggregated feature CSVs.
"""

import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from ml.feature_extraction import extract_features_from_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_POOL = [
    ("Q1", "Are you currently in the testing room?", "control", 0),
    ("Q2", "Did you intentionally steal the USB drive from the lab?", "relevant", 1),
    ("Q3", "Is your age greater than 15?", "control", 0),
    ("Q4", "Did you access the confidential grading server without permission?", "relevant", 1),
    ("Q5", "Are you wearing shoes right now?", "control", 0),
    ("Q6", "Did you fabricate the experimental data points in the report?", "relevant", 1),
    ("Q7", "Is the current year 2026?", "control", 0),
    ("Q8", "Did you take money from the petty cash box?", "relevant", 1),
    ("Q9", "Have you ever drunk a glass of water?", "control", 0),
    ("Q10", "Did you tamper with the hardware circuits before the demo?", "relevant", 1),
    ("Q11", "Is the sun currently in the sky?", "control", 0),
    ("Q12", "Did you leak the private authentication keys online?", "relevant", 1),
]


def generate_subject_session(
    subject_id: str,
    session_id: str,
    num_questions: int = 10,
    sample_rate_hz: int = 100,
    question_duration_sec: int = 10
):
    """Generates continuous time-series for a single subject session."""
    # Subject-specific baseline traits
    base_bpm = random.uniform(62.0, 82.0)
    base_gsr = random.uniform(280.0, 500.0)

    # Reactivity factor (some subjects show higher EDA responses, some higher cardiac)
    cardiac_reactivity = random.uniform(0.7, 1.3)
    gsr_reactivity = random.uniform(0.7, 1.4)

    raw_records = []
    features_records = []

    selected_questions = random.sample(QUESTIONS_POOL, min(num_questions, len(QUESTIONS_POOL)))

    for q_idx, (q_id, q_text, q_type, expected_label) in enumerate(selected_questions):
        # Determine actual label (mostly truthful for control, lie for relevant with some noise)
        is_lie = expected_label == 1
        label = 1 if is_lie else 0

        # Physiological response target for this trial
        if is_lie:
            # During deception: Sympathetic fight-or-flight activation
            target_bpm = base_bpm + random.uniform(10.0, 22.0) * cardiac_reactivity
            gsr_surge = random.uniform(120.0, 280.0) * gsr_reactivity
            num_scr_peaks = random.randint(2, 5)
        else:
            # During truth: Relaxed or slight orienting reflex
            target_bpm = base_bpm + random.uniform(-2.0, 4.0)
            gsr_surge = random.uniform(5.0, 35.0)
            num_scr_peaks = random.randint(0, 1)

        samples_count = int(question_duration_sec * sample_rate_hz)
        dt = 1.0 / sample_rate_hz

        pulse_samples = np.zeros(samples_count)
        gsr_samples = np.zeros(samples_count)
        bpm_samples = np.zeros(samples_count)

        phase = 0.0
        current_bpm = base_bpm
        current_gsr = base_gsr

        for i in range(samples_count):
            t = i * dt

            # Dynamic BPM curve: rises 2-4 seconds after question, then slowly plateaus
            if is_lie:
                arousal_profile = math.exp(-0.5 * ((t - 4.5) / 2.5) ** 2)
                inst_target_bpm = base_bpm + (target_bpm - base_bpm) * arousal_profile
            else:
                inst_target_bpm = target_bpm

            current_bpm += 0.03 * (inst_target_bpm - current_bpm)
            bpm_samples[i] = current_bpm + random.uniform(-0.5, 0.5)

            # Cardiac pulse photoplethysmogram (PPG) waveform
            freq = current_bpm / 60.0
            phase += 2.0 * math.pi * freq * dt
            # BVP wave: systolic peak + dicrotic notch
            wave = math.sin(phase) + 0.35 * math.sin(2.0 * phase)
            pulse_raw = int(512 + 180 * wave + random.uniform(-8, 8))
            pulse_samples[i] = np.clip(pulse_raw, 0, 1023)

            # GSR response: tonic baseline drift + phasic skin conductance response (SCR)
            if is_lie:
                # SCR rise onset around t = 1.8s
                scr_curve = 0.0
                if t > 1.8:
                    scr_curve = gsr_surge * ((t - 1.8) / 2.2) * math.exp(-(t - 1.8) / 2.5)
                # SCR micro-bursts
                micro_bursts = sum(
                    15.0 * math.exp(-0.5 * ((t - peak_t) / 0.4) ** 2)
                    for peak_t in np.linspace(2.5, 8.5, num_scr_peaks)
                ) if num_scr_peaks > 0 else 0.0
                inst_gsr = base_gsr + scr_curve + micro_bursts
            else:
                # Minor baseline noise during truth
                inst_gsr = base_gsr + gsr_surge * 0.2 * math.sin(0.5 * t)

            inst_gsr += random.uniform(-3, 3)
            gsr_samples[i] = np.clip(int(inst_gsr), 0, 1023)

            raw_records.append({
                "timestamp_ms": round(t * 1000, 2),
                "subject_id": subject_id,
                "session_id": session_id,
                "baseline_bpm": round(base_bpm, 2),
                "baseline_gsr": round(base_gsr, 2),
                "question_id": q_id,
                "question_text": q_text,
                "question_type": q_type,
                "ground_truth_label": label,
                "pulse_raw": int(pulse_samples[i]),
                "gsr_raw": int(gsr_samples[i]),
                "instant_bpm": round(bpm_samples[i], 2)
            })

        # Extract features for this trial
        feats = extract_features_from_window(
            pulse_arr=pulse_samples,
            gsr_arr=gsr_samples,
            bpm_arr=bpm_samples,
            baseline_bpm=base_bpm,
            baseline_gsr=base_gsr,
            sample_rate_hz=sample_rate_hz
        )

        features_records.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "question_id": q_id,
            "question_text": q_text,
            "question_type": q_type,
            "baseline_bpm": round(base_bpm, 2),
            "baseline_gsr": round(base_gsr, 2),
            **feats,
            "label": label
        })

    return raw_records, features_records


def generate_benchmark_dataset(num_subjects: int = 15, questions_per_subject: int = 10):
    """Generates synthetic dataset across multiple subjects and writes to disk."""
    print(f"Generating physiological polygraph dataset across {num_subjects} subjects...")
    all_raw = []
    all_features = []

    for s_idx in range(1, num_subjects + 1):
        sub_id = f"subject_{s_idx:02d}"
        sess_id = f"sess_{s_idx:02d}"
        raw_rec, feat_rec = generate_subject_session(
            subject_id=sub_id,
            session_id=sess_id,
            num_questions=questions_per_subject
        )
        all_raw.extend(raw_rec)
        all_features.extend(feat_rec)

    # Save aggregated features
    features_df = pd.DataFrame(all_features)
    processed_path = PROCESSED_DIR / "polygraph_features.csv"
    features_df.to_csv(processed_path, index=False)

    # Save a sample raw session
    sample_raw_df = pd.DataFrame([r for r in all_raw if r["subject_id"] == "subject_01"])
    sample_raw_path = RAW_DIR / "sample_session_subject_01.csv"
    sample_raw_df.to_csv(sample_raw_path, index=False)

    print(f"Dataset Generation Complete!")
    print(f"  * Processed Features Dataset : {processed_path}")
    print(f"  * Total trials generated    : {len(features_df)}")
    print(f"  * Class distribution        : Truth={sum(features_df['label'] == 0)}, Lie={sum(features_df['label'] == 1)}")
    print(f"  * Sample Raw Session CSV    : {sample_raw_path}")
    return str(processed_path)


if __name__ == "__main__":
    generate_benchmark_dataset(num_subjects=18, questions_per_subject=10)
