#!/usr/bin/env python3
"""
Physiological Feature Extractor for Polygraph Lie Detection
============================================================
Transforms raw Pulse and GSR time-series windows into structured,
scientifically validated Autonomic Nervous System (ANS) features:
  - Cardiovascular / Pulse Metrics (BPM change, peak BPM, HRV: SDNN, RMSSD, pNN50)
  - Galvanic Skin Response / Electrodermal Activity (Tonic baseline shift, Phasic SCR peaks, Max slope, AUC)
  - Composite Sympathetic Arousal Index
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def extract_rr_intervals(pulse_signal: np.ndarray, sample_rate_hz: float = 100.0) -> List[float]:
    """Detects peak-to-peak intervals (RR intervals in milliseconds) from pulse waveform."""
    if len(pulse_signal) < 20:
        return []

    # Dynamic thresholding based on signal amplitude
    mean_val = np.mean(pulse_signal)
    std_val = np.std(pulse_signal)
    threshold = mean_val + 0.3 * std_val

    peaks = []
    min_distance = int(0.33 * sample_rate_hz)  # Max 180 BPM refractory window

    last_peak = -min_distance
    for i in range(1, len(pulse_signal) - 1):
        if pulse_signal[i] > threshold and pulse_signal[i] >= pulse_signal[i - 1] and pulse_signal[i] >= pulse_signal[i + 1]:
            if i - last_peak >= min_distance:
                peaks.append(i)
                last_peak = i

    rr_intervals_ms = []
    for j in range(1, len(peaks)):
        rr_ms = (peaks[j] - peaks[j - 1]) * (1000.0 / sample_rate_hz)
        if 330 <= rr_ms <= 1500:  # Physically valid human heartbeat range
            rr_intervals_ms.append(rr_ms)

    return rr_intervals_ms


def compute_hrv_metrics(rr_intervals_ms: List[float]) -> Dict[str, float]:
    """Calculates Heart Rate Variability (HRV) metrics: SDNN, RMSSD, pNN50."""
    if len(rr_intervals_ms) < 3:
        return {
            "hrv_sdnn": 0.0,
            "hrv_rmssd": 0.0,
            "hrv_pnn50": 0.0
        }

    rr_arr = np.array(rr_intervals_ms)
    sdnn = float(np.std(rr_arr))

    successive_diffs = np.diff(rr_arr)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100.0) if len(successive_diffs) > 0 else 0.0

    return {
        "hrv_sdnn": round(sdnn, 2),
        "hrv_rmssd": round(rmssd, 2),
        "hrv_pnn50": round(pnn50, 2)
    }


def detect_scr_peaks(gsr_signal: np.ndarray, sample_rate_hz: float = 100.0) -> int:
    """Estimates the count of Skin Conductance Responses (SCR peaks) during the window."""
    if len(gsr_signal) < 20:
        return 0

    # Moving average smooth
    window = max(3, int(sample_rate_hz * 0.2))  # 200ms smoothing window
    smooth = np.convolve(gsr_signal, np.ones(window) / window, mode="valid")

    diff = np.diff(smooth)
    peak_count = 0
    in_rise = False
    rise_magnitude = 0.0

    for d in diff:
        if d > 0.5:  # Significant upward slope
            in_rise = True
            rise_magnitude += d
        elif d < -0.2 and in_rise:
            if rise_magnitude > 5.0:  # Valid SCR threshold
                peak_count += 1
            in_rise = False
            rise_magnitude = 0.0

    return peak_count


def extract_features_from_window(
    pulse_arr: np.ndarray,
    gsr_arr: np.ndarray,
    bpm_arr: np.ndarray,
    baseline_bpm: float = 72.0,
    baseline_gsr: float = 400.0,
    sample_rate_hz: float = 100.0
) -> Dict[str, float]:
    """
    Extracts comprehensive physiological features from a question response window.
    """
    features = {}

    # --- 1. Cardiovascular / BPM Features ---
    valid_bpm = bpm_arr[(bpm_arr > 35) & (bpm_arr < 195)]
    if len(valid_bpm) == 0:
        valid_bpm = np.array([baseline_bpm])

    bpm_mean = float(np.mean(valid_bpm))
    bpm_max = float(np.max(valid_bpm))
    bpm_min = float(np.min(valid_bpm))
    bpm_std = float(np.std(valid_bpm))

    features["bpm_mean"] = round(bpm_mean, 2)
    features["bpm_max"] = round(bpm_max, 2)
    features["bpm_min"] = round(bpm_min, 2)
    features["bpm_std"] = round(bpm_std, 2)
    features["bpm_delta"] = round(bpm_mean - baseline_bpm, 2)
    features["bpm_max_delta"] = round(bpm_max - baseline_bpm, 2)

    # --- 2. HRV Metrics ---
    rr_intervals = extract_rr_intervals(pulse_arr, sample_rate_hz)
    hrv_dict = compute_hrv_metrics(rr_intervals)
    features.update(hrv_dict)

    # --- 3. Pulse Amplitude Features ---
    pulse_mean = float(np.mean(pulse_arr))
    pulse_std = float(np.std(pulse_arr))
    pulse_range = float(np.ptp(pulse_arr))
    features["pulse_mean"] = round(pulse_mean, 2)
    features["pulse_std"] = round(pulse_std, 2)
    features["pulse_amplitude"] = round(pulse_range, 2)

    # --- 4. Galvanic Skin Response (GSR) Features ---
    gsr_mean = float(np.mean(gsr_arr))
    gsr_max = float(np.max(gsr_arr))
    gsr_min = float(np.min(gsr_arr))
    gsr_std = float(np.std(gsr_arr))

    features["gsr_mean"] = round(gsr_mean, 2)
    features["gsr_max"] = round(gsr_max, 2)
    features["gsr_min"] = round(gsr_min, 2)
    features["gsr_std"] = round(gsr_std, 2)
    features["gsr_delta"] = round(gsr_mean - baseline_gsr, 2)
    features["gsr_max_delta"] = round(gsr_max - baseline_gsr, 2)
    features["gsr_ratio"] = round(gsr_mean / max(1.0, baseline_gsr), 3)
    features["gsr_max_ratio"] = round(gsr_max / max(1.0, baseline_gsr), 3)

    # Max rise slope (dGSR / dt)
    if len(gsr_arr) > 5:
        smoothed_gsr = np.convolve(gsr_arr, np.ones(5) / 5, mode="valid")
        slopes = np.diff(smoothed_gsr) * sample_rate_hz
        max_slope = float(np.max(slopes)) if len(slopes) > 0 else 0.0
    else:
        max_slope = 0.0
    features["gsr_max_slope"] = round(max(0.0, max_slope), 2)

    # Phasic SCR peaks count
    features["gsr_scr_peaks"] = detect_scr_peaks(gsr_arr, sample_rate_hz)

    # Area Under the Curve (AUC) relative to baseline
    diff_from_base = np.maximum(0.0, gsr_arr - baseline_gsr)
    features["gsr_auc"] = round(float(np.sum(diff_from_base) / sample_rate_hz), 2)

    # --- 5. Composite Stress Index ---
    # Polygraph stress score: normalized heart acceleration + normalized skin conductance surge
    norm_bpm_delta = max(0.0, features["bpm_delta"]) / 25.0
    norm_gsr_delta = max(0.0, features["gsr_delta"]) / 300.0
    features["stress_index"] = round(0.4 * norm_bpm_delta + 0.6 * norm_gsr_delta, 3)

    return features


def process_raw_session_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw session dataframe (containing continuous samples for multiple questions)
    and aggregates each question trial into a single feature row for ML training.
    """
    rows = []

    # Group by subject_id, session_id, question_id
    grouped = df.groupby(["subject_id", "session_id", "question_id"], sort=False)

    for (subject_id, session_id, q_id), group in grouped:
        baseline_bpm = float(group["baseline_bpm"].iloc[0]) if "baseline_bpm" in group.columns else 72.0
        baseline_gsr = float(group["baseline_gsr"].iloc[0]) if "baseline_gsr" in group.columns else 400.0
        label = int(group["ground_truth_label"].iloc[0]) if "ground_truth_label" in group.columns else 0
        q_type = group["question_type"].iloc[0] if "question_type" in group.columns else "unknown"
        q_text = group["question_text"].iloc[0] if "question_text" in group.columns else ""

        pulse_arr = group["pulse_raw"].to_numpy(dtype=float)
        gsr_arr = group["gsr_raw"].to_numpy(dtype=float)
        bpm_arr = group["instant_bpm"].to_numpy(dtype=float)

        feats = extract_features_from_window(
            pulse_arr=pulse_arr,
            gsr_arr=gsr_arr,
            bpm_arr=bpm_arr,
            baseline_bpm=baseline_bpm,
            baseline_gsr=baseline_gsr,
            sample_rate_hz=100.0
        )

        row_dict = {
            "subject_id": subject_id,
            "session_id": session_id,
            "question_id": q_id,
            "question_text": q_text,
            "question_type": q_type,
            "baseline_bpm": baseline_bpm,
            "baseline_gsr": baseline_gsr,
            **feats,
            "label": label  # 0 = Truth, 1 = Lie
        }
        rows.append(row_dict)

    return pd.DataFrame(rows)
