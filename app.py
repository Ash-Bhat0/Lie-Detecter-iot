#!/usr/bin/env python3
"""Combined launcher for the Raspberry Pi 4B sensor prototype and LIAR-PLUS text model.

The two modes are intentionally kept separate:
1. Sensor mode measures pulse/GSR physiological signals.
2. Text mode is for the LIAR-PLUS claim-classification model.

These are not fused into a single lie score. The sensor measures arousal, while
the text model predicts one of the dataset's six claim labels. Neither proves
whether a person is lying.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SENSOR_SCRIPT = ROOT / "hardware" / "polygraph_pi4b.py"
MODEL_DIR = ROOT / "text_classifier_original"

def sensor_mode():
    if not SENSOR_SCRIPT.exists():
        print(f"Sensor script not found: {SENSOR_SCRIPT}")
        return
    print("\nStarting Raspberry Pi 4B sensor mode.")
    print("Make sure the MCP3008 and protected meter/buzzer driver circuits are wired correctly.")
    subprocess.run([sys.executable, str(SENSOR_SCRIPT)], check=False)

def classifier_mode():
    print("\nLIAR-PLUS text-classifier mode")
    print("This mode is not yet runnable from the uploaded files alone.")
    print("\nRequired before predictions can be made:")
    print("  1. LIAR-PLUS TSV data files (train2.tsv, val2.tsv, test2.tsv or equivalent)")
    print("  2. GloVe 200-dimensional embedding file (glove.6B.200d.txt)")
    print("  3. A trained model checkpoint compatible with the model architecture")
    print("\nThe uploaded source is research/training code, not a ready-to-use inference package.")
    print("Several source files also contain incomplete or inconsistent implementation details.")
    print("See text_classifier_original/README.md and the integration README.")
    print("\nDo not interpret a model label as proof of deception; it is a dataset-trained text classification.")

def main():
    while True:
        print("\n=== Kinski Polygraph + Claim Analyzer (Raspberry Pi 4 Model B) ===")
        print("1. Run pulse/GSR sensor mode")
        print("2. Claim text-classifier status / requirements")
        print("3. Exit")
        choice = input("Select 1-3: ").strip()
        if choice == "1":
            sensor_mode()
        elif choice == "2":
            classifier_mode()
        elif choice == "3":
            break
        else:
            print("Enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
