# Kinski Polygraph + Claim Analyzer — Raspberry Pi 4 Model B

This bundle combines the two uploaded project directions:

1. **Physiological sensor prototype** — pulse sensor + GSR electrode circuit, MCP3008 ADC, two analog VU meters, and buzzer, adapted from `NanoLieDetector.ino`.
2. **LIAR-PLUS text classifier** — the uploaded PyTorch research code for six-class claim classification (`true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`).

## Critical distinction: two modes, not one lie score

These components measure different things and should not be merged into a single score:

- The sensor mode measures pulse-related activity and skin conductance, which may change for many reasons including movement, temperature, excitement, stress, or fear.
- The text classifier predicts a dataset label for a written claim based on its trained model, text, justification, and credit-history features.

The LIAR-PLUS model does not read the physical sensors, and the physical sensor does not verify the truth of a claim. Combining their outputs into one “lie probability” would be scientifically unsupported. The launcher therefore keeps them as separate modes.

## Project structure

```text
Kinski_Polygraph_Claim_Analyzer_Pi4B/
├── app.py
├── hardware/
│   ├── polygraph_pi4b.py
│   └── NanoLieDetector_original.ino
└── text_classifier_original/
    ├── Attention.py
    ├── ConvS2S.py
    ├── datasets.py
    ├── Encoder.py
    ├── LiarLiar.py
    ├── LICENSE
    ├── main.py
    ├── README.md
    ├── test.py
    ├── trainer.py
    └── utils.py
```

## Run the combined launcher

On Raspberry Pi OS:

```bash
python3 app.py
```

Select:
- `1` for sensor mode
- `2` to see what is required to enable the text-classifier mode
- `3` to exit

## Sensor mode setup

The Raspberry Pi 4 Model B has no analog inputs, so the sensor script assumes an MCP3008 ADC.

Install packages and enable SPI:

```bash
sudo apt update
sudo apt install -y python3-spidev python3-gpiozero
sudo raspi-config
```

In `raspi-config`, select **Interface Options → SPI → Enable**, then reboot if prompted.

Typical SPI wiring:

| MCP3008 | Raspberry Pi 4B |
|---|---|
| VDD | 3.3 V, physical pin 1 |
| VREF | 3.3 V, physical pin 1 |
| AGND, DGND | GND, physical pin 6 |
| CLK | GPIO11 / SCLK, physical pin 23 |
| DOUT | GPIO9 / MISO, physical pin 21 |
| DIN | GPIO10 / MOSI, physical pin 19 |
| CS/SHDN | GPIO8 / CE0, physical pin 24 |
| CH0 | Analog pulse sensor output |
| CH1 | GSR measurement node |

Output driver inputs use BCM GPIO18 (physical pin 12) for the pulse meter, BCM GPIO19 (physical pin 35) for the GSR meter, and BCM GPIO23 (physical pin 16) for the buzzer.

**Electrical safety:** MCP3008 VDD/VREF should be 3.3 V when directly connected to Pi SPI. Do not apply 5 V to Pi GPIO. Never drive meter coils or a high-current buzzer directly from GPIO; use appropriately designed transistor/MOSFET driver circuits and suitable power. Verify the GSR circuit against the original circuit diagram before connecting electrodes to a person.

The Python pulse detector is an approximation of Arduino's interrupt-driven `PulseSensorPlayground` behavior and requires testing/calibration with the actual sensor.

## Enabling the text classifier

The uploaded text-classifier source is not yet a ready-to-run prediction service. The following files/data are not included in the uploaded set:

- `train2.tsv`, `val2.tsv`, `test2.tsv` or compatible LIAR-PLUS data
- `glove.6B.200d.txt`
- a trained PyTorch model checkpoint (`state_dict`)
- any extra pickled test fixtures referenced by `test.py`

Also, the source contains incomplete/inconsistent code that needs repair before training or inference. For example, some layer definitions refer to undefined values, imports use inconsistent capitalization, the trainer has undefined variables and GPU-only assumptions, and embedding creation uses an incompatible conversion. These need to be fixed and the model must be trained or supplied with a compatible checkpoint before claim predictions can be made.

The files are retained under `text_classifier_original/` for reference. They are not silently altered by this integration bundle.

## Suggested next integration stage

After the model code and assets are repaired/available, expose a function such as:

```python
predict_claim(statement: str, justification: str, credit_history: list[int]) -> dict
```

Then the app can offer separate screens for:
- live pulse/GSR readings
- claim text classification
- optional session report that shows the two outputs side-by-side without combining them into a deception score

For the current source, no such prediction function is claimed to work.

## Attribution and license

The original sensor project is attributed to Roni Bandini. The uploaded LIAR-PLUS repository includes an MIT license attributed to Tanzeel Ur Rahman; preserve the license when redistributing those source files. Check the original repositories for any additional asset or dataset terms.

## Scientific note

Neither physiological arousal nor a dataset-based claim label proves that a person is lying. Use this as an experimental electronics and text-classification project, not as an authoritative lie detector.
