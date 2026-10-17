Raspberry Pi 4 Model B

This is a Raspberry Pi 4 Model B adaptation of the original
`NanoLieDetector.ino` project. The source sketch has now been reviewed
and the Python port follows its main behavior:

-   pulse sensor input on the equivalent of Arduino A0
-   GSR input on the equivalent of Arduino A1
-   configured pulse threshold of 510
-   GSR compensation of 100
-   GSR reading updated every third detected beat
-   pulse meter briefly moves to PWM value 12 and returns to 2
-   GSR meter maps positive compensated readings to 0--20
-   buzzer pulses at 50 Hz for 10 ms when compensated GSR exceeds 800
-   startup needle sweep

**Important:** Python on Linux does not sample analog inputs with the
same deterministic timing as the Arduino's `PulseSensorPlayground`
interrupt-driven library. The beat detection in `polygraph_pi4b.py` is
therefore an approximation and must be calibrated with the actual
sensor. It is not a medically validated instrument and does not detect
truth or lies; it shows physiological responses that can have many
causes.

## Hardware

  ------------------------------------------------------------------------
                           Qty Component             Notes
  ---------------------------- --------------------- ---------------------
                             1 Raspberry Pi 4 Model  Main controller
                               B                     

                             1 Suitable regulated    Use a power supply
                               USB-C 5.1 V supply    appropriate for the
                                                     Pi 4

                             1 MCP3008 ADC           Needed because the Pi
                                                     4 has no analog input

                             1 Analog pulse sensor   Analog-output type
                                                     assumed by this code

                             1 GSR electrode pair    Aluminum foil
                                                     electrodes as in
                                                     original concept

                             1 300--330 kΩ resistor  Original notes
                                                     describe this in the
                                                     GSR circuit; verify
                                                     the exact circuit
                                                     diagram

                             2 Analog VU meter       Pulse and GSR
                               movements             displays

                             2 Transistor/MOSFET     Select for each
                               driver circuits       meter's coil
                                                     resistance/current

                             1 Buzzer                Verify
                                                     voltage/current;
                                                     driver may be
                                                     required

                           --- Breadboard, jumper    Prototyping
                               wires, resistors      

                           --- microSD card and      Operating system
                               Raspberry Pi OS       
  ------------------------------------------------------------------------

The original Arduino Nano is not needed.

## Wiring

Use **BCM GPIO numbering** in the Python code. Physical header pin
numbers are shown for the SPI connections.

### MCP3008 to Raspberry Pi 4

  MCP3008 pin/signal   Raspberry Pi 4 connection
  -------------------- --------------------------------------------
  VDD                  3.3 V, physical pin 1
  VREF                 3.3 V, physical pin 1
  AGND                 GND, physical pin 6
  DGND                 GND, physical pin 6
  CLK                  GPIO11 / SCLK, physical pin 23
  DOUT                 GPIO9 / MISO, physical pin 21
  DIN                  GPIO10 / MOSI, physical pin 19
  CS/SHDN              GPIO8 / CE0, physical pin 24
  CH0                  Pulse sensor analog output (A0 equivalent)
  CH1                  GSR measurement node (A1 equivalent)

Power the MCP3008 at 3.3 V when connected directly to Pi SPI. Do not
connect a 5 V logic signal to a Pi GPIO pin.

### Outputs

  Function                     BCM GPIO   Physical pin
  -------------------------- ---------- --------------
  Pulse meter driver input       GPIO18             12
  GSR meter driver input         GPIO19             35
  Buzzer driver input            GPIO23             16

**Do not connect meter coils directly to GPIO pins.** The GPIOs are
control signals only. Use suitable transistor/MOSFET drivers and an
appropriate meter supply, designed for the meter's rated current. Do not
power a buzzer from a GPIO if it exceeds GPIO current limits. Ensure the
driver inputs are compatible with 3.3 V logic.

### GSR circuit

The original sketch comments describe GSR electrode 1 connected to A1
with a 300 kΩ resistor to ground, and GSR electrode 2 connected to 5 V.
The README also mentions 330 kΩ as a suitable standard value. In this
adaptation, the measurement node goes to MCP3008 CH1.

Treat that as a description of the original project, not a complete
safety-certified circuit. Verify the original Hackster circuit diagram
before assembling it. Use a low-voltage, current-limited measurement
circuit and never connect a person to mains-powered circuitry.

## Install software

Install Raspberry Pi OS, boot the Pi, and run:

``` bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Enable SPI:

``` bash
sudo raspi-config
```

Select **Interface Options → SPI → Enable**, then reboot if prompted.

Install dependencies:

``` bash
sudo apt install -y python3-pip python3-venv python3-spidev python3-gpiozero
```

Verify SPI is available:

``` bash
ls /dev/spidev*
```

You should normally see `/dev/spidev0.0` after enabling SPI and using
CE0.

## Run the program

Copy `polygraph_pi4b.py` onto the Pi, then run:

``` bash
python3 polygraph_pi4b.py
```

Stop with `Ctrl+C`. The program sets the outputs off during shutdown.

If `spidev` is missing in a virtual environment, create one with system
packages enabled:

``` bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python polygraph_pi4b.py
```

## How the converted logic works

1.  At startup, both meter outputs sweep through values 0--19, then
    return to zero.
2.  The buzzer produces a brief startup pulse.
3.  The pulse analog signal is read from MCP3008 CH0.
4.  A rising crossing of the configured threshold (510) is treated as a
    beat, with a refractory period to reject immediate duplicate
    detections.
5.  The estimated BPM is calculated from the time between detected
    beats. Events at or above 180 BPM are ignored, matching the original
    sketch's `myBPM < 180` condition.
6.  The GSR input is read every third accepted beat.
7.  The compensation value 100 is subtracted from the GSR ADC reading.
8.  Positive GSR values are mapped to a meter value between 0 and 20;
    non-positive values set the GSR meter to zero.
9.  If the compensated GSR reading exceeds 800, the buzzer pulses for 10
    ms at 50 Hz.

## Calibration and limitations

-   `THRESHOLD = 510` comes from the original Arduino sketch. Keep it as
    a starting point, but calibrate it to your actual pulse sensor and
    signal range.
-   The Arduino `PulseSensorPlayground` library does more sophisticated
    sampling and beat detection than this Python approximation. If beats
    are missed or falsely detected, inspect sensor output and adjust the
    detection algorithm.
-   Some pulse sensors produce a signal that needs filtering or adaptive
    thresholding. A single fixed threshold may not work for every sensor
    or person.
-   `galvanicVal > 800` is retained from the original sketch. It may
    rarely trigger depending on the circuit and ADC values.
-   The code's GSR channel and meter values depend on the exact circuit.
    Higher readings are not universal proof of stress or deception.
-   The original sketch defines `potPin = 2` but does not use it in the
    shown logic; no potentiometer input is implemented in this port.
-   The original sketch uses `LED13` for a heartbeat indicator. The
    Raspberry Pi port does not use a separate LED by default.
-   The original code's analog meter outputs are PWM. On the Pi,
    `gpiozero` PWM is used as a control signal for external driver
    circuits.

## Troubleshooting

  -----------------------------------------------------------------------
  Issue                               What to check
  ----------------------------------- -----------------------------------
  `/dev/spidev0.0` not found          Enable SPI in `raspi-config`,
                                      reboot, check wiring

  ADC reads only 0 or 1023            Verify MCP3008 power, reference
                                      voltage, ground, channel and sensor
                                      output

  False/missed beats                  Sensor placement, threshold, noise,
                                      sample timing, and refractory
                                      period

  Meter does not move                 Driver circuit, meter current
                                      rating, GPIO assignment, power
                                      supply and ground

  Pi resets when meter moves          Meter coil current or electrical
                                      noise may be affecting the Pi;
                                      redesign/isolated supply for the
                                      driver

  Buzzer does not sound               Verify buzzer type,
                                      voltage/current, transistor driver,
                                      GPIO pin and ground

  `ModuleNotFoundError: spidev`       Install `python3-spidev` or use a
                                      venv created with
                                      `--system-site-packages`
  -----------------------------------------------------------------------

## References

-   Original repository: https://github.com/ronibandini/NanoLieDetector
-   Original Hackster tutorial:
    https://www.hackster.io/roni-bandini/arduino-polygraph-4d3849
-   Original project article:
    https://bandini.medium.com/lie-detector-with-arduino-45e44befa810
-   Raspberry Pi 4 Model B:
    https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
-   Raspberry Pi documentation:
    https://www.raspberrypi.com/documentation/
-   MCP3008: https://www.microchip.com/en-us/product/MCP3008
-   National Academies, *The Polygraph and Lie Detection*:
    https://nap.nationalacademies.org/catalog/10420/the-polygraph-and-lie-detection
-   American Psychological Association, polygraph overview:
    https://www.apa.org/topics/cognitive-neuroscience/polygraph

## License and attribution

The original README says the repository does not specify a software
license. Check the current repository license and obtain any required
permissions before redistributing original assets or derivative work.
The original project is attributed to Roni Bandini.
