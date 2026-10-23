#!/usr/bin/env python3
"""
Combined Launcher for IoT Polygraph & Physiological Lie Detector
================================================================
Platform: Raspberry Pi 4 Model B / PC Cross-Platform Mode

Modes:
  1. Live Sensor Hardware Monitor (Raw ADC / VU Meter test)
  2. Record New Training Session (Interrogation protocol with ground-truth labels)
  3. Train / Retrain Edge ML Models (Random Forest, SVM, Gradient Boosting)
  4. Run Real-Time ML Lie Detector (Interrogation mode with live prediction & feedback)
  5. Generate Benchmark Synthetic Dataset (18 subjects with physiological responses)
  6. LIAR-PLUS NLP Claim Classifier Status (Text analysis direction)
  7. Exit
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARDWARE_SCRIPT = ROOT / "polygraph_pi4b.py"


def run_hardware_monitor():
    """Runs the direct hardware VU meter and buzzer monitor."""
    print("\n--- Starting Raw Sensor Hardware Monitor ---")
    if not HARDWARE_SCRIPT.exists():
        print(f"Error: Sensor script not found at {HARDWARE_SCRIPT}")
        return
    subprocess.run([sys.executable, str(HARDWARE_SCRIPT)], cwd=str(ROOT))


def run_data_collection():
    """Launches the labeled interrogation session recorder."""
    print("\n--- Starting Labeled Sensor Data Collector ---")
    from ml.data_collector import run_data_collection_session
    run_data_collection_session()


def run_model_training():
    """Runs feature extraction on raw sessions and trains candidate ML models."""
    print("\n--- Running Edge ML Training & Cross-Validation Pipeline ---")
    from ml.train_model import train_and_evaluate
    train_and_evaluate()


def run_live_polygraph_ml():
    """Runs the live ML Lie Detector interrogation engine."""
    print("\n--- Launching Real-Time ML Lie Detector ---")
    from ml.polygraph_ml import run_live_interrogation
    run_live_interrogation()


def run_benchmark_generator():
    """Generates synthetic physiological benchmark dataset."""
    print("\n--- Generating Benchmark Physiological Dataset ---")
    from ml.generate_synthetic_data import generate_benchmark_dataset
    generate_benchmark_dataset(num_subjects=18, questions_per_subject=10)


def show_nlp_status():
    """Displays information on the LIAR-PLUS text claim direction."""
    print("\n" + "=" * 65)
    print("        LIAR-PLUS TEXT CLAIM CLASSIFIER (NLP DIRECTION)")
    print("=" * 65)
    print("This mode is for the linguistic/text claim verification model.")
    print("Requirements before training the NLP neural network:")
    print("  1. LIAR-PLUS TSV files: train2.tsv, val2.tsv, test2.tsv")
    print("  2. GloVe 200d Word Embeddings: glove.6B.200d.txt")
    print("  3. GPU recommended for PyTorch ConvS2S / MultiHeadAttention training.")
    print("\nNote: The primary IoT focus is the Physiological Sensor ML pipeline")
    print("(Options 2, 3, and 4) which directly measures autonomic nervous arousal")
    print("from the pulse sensor and galvanic skin response electrodes.")
    print("=" * 65)


def main():
    while True:
        print("\n" + "=" * 65)
        print("   KINSKI IOT POLYGRAPH & LIE DETECTION SYSTEM (Raspberry Pi 4B)")
        print("=" * 65)
        print("  1. Live Sensor Hardware Monitor (Raw Pulse/GSR & VU needle sweep)")
        print("  2. Record New Training Session (Interrogation data logger with labels)")
        print("  3. Train / Retrain Edge ML Models (Random Forest, SVM, GBM)")
        print("  4. Run Real-Time ML Lie Detector (Interrogation mode + Live AI)")
        print("  5. Generate / Refresh Benchmark Training Dataset")
        print("  6. LIAR-PLUS Text Claim Classifier (NLP Requirements)")
        print("  7. Exit")
        print("=" * 65)

        choice = input("Select an option (1-7): ").strip()
        if choice == "1":
            run_hardware_monitor()
        elif choice == "2":
            run_data_collection()
        elif choice == "3":
            run_model_training()
        elif choice == "4":
            run_live_polygraph_ml()
        elif choice == "5":
            run_benchmark_generator()
        elif choice == "6":
            show_nlp_status()
        elif choice == "7":
            print("\nExiting. Thank you!")
            break
        else:
            print("\n[Invalid Selection] Please choose a number between 1 and 7.")


if __name__ == "__main__":
    main()
