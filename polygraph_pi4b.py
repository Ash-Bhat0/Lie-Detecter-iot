#!/usr/bin/env python3
"""
Kinski Polygraph — Raspberry Pi 4 Model B adaptation
Based on NanoLieDetector.ino by Roni Bandini (2020).

Hardware assumptions:
- Analog pulse sensor -> MCP3008 CH0
- GSR measurement node -> MCP3008 CH1
- MCP3008 SPI CE0, powered at 3.3 V
- Pulse VU-meter driver input -> BCM GPIO18
- GSR VU-meter driver input -> BCM GPIO19
- Buzzer driver input -> BCM GPIO23

This is a software port of the sketch's intended behavior, not a medically
validated instrument. Python sampling is not identical to the Arduino
PulseSensorPlayground interrupt-based library; calibrate the threshold.
"""

import time
import math
import random

# --- Hardware pins (BCM numbering) ---
PULSE_METER_GPIO = 18
GSR_METER_GPIO = 19
BUZZER_GPIO = 23

# --- MCP3008 channels ---
PULSE_CHANNEL = 0       # Arduino A0 equivalent
GSR_CHANNEL = 1          # Arduino A1 equivalent

# --- Values from the original sketch ---
THRESHOLD = 510
GALVANIC_COMPENSATION = 100
BPM_LIMIT = 180
SAMPLE_INTERVAL = 0.002  # target ~500 Hz sampling
REFRACTORY_PERIOD = 0.28 # reject implausibly close beat detections
GSR_READ_EVERY_N_BEATS = 3

# Arduino analogWrite() values are 0..255.
# gpiozero PWMOutputDevice.value is normalized 0.0..1.0.
PULSE_BEAT_PWM = 12
PULSE_REST_PWM = 2
METER_PWM_MAX = 255

# Mock device helper for non-Raspberry Pi environments
class MockPWMDevice:
    def __init__(self, pin, frequency=1000, initial_value=0):
        self.pin = pin
        self.value = initial_value
    def close(self):
        pass

# Attempt to load gpiozero
try:
    from gpiozero import PWMOutputDevice
    pulse_meter = PWMOutputDevice(PULSE_METER_GPIO, frequency=1000, initial_value=0)
    gsr_meter = PWMOutputDevice(GSR_METER_GPIO, frequency=1000, initial_value=0)
    buzzer = PWMOutputDevice(BUZZER_GPIO, frequency=50, initial_value=0)
    HAS_GPIO = True
except Exception:
    pulse_meter = MockPWMDevice(PULSE_METER_GPIO)
    gsr_meter = MockPWMDevice(GSR_METER_GPIO)
    buzzer = MockPWMDevice(BUZZER_GPIO)
    HAS_GPIO = False

# Attempt to load spidev
HAS_SPI = False
spi = None
try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)  # SPI bus 0, CE0
    spi.max_speed_hz = 1_000_000
    HAS_SPI = True
except Exception:
    HAS_SPI = False


# Simulation state for mock mode
sim_phase = 0.0
sim_bpm = 72.0


def read_adc(channel: int) -> int:
    """Read MCP3008 channel 0..7, returning 0..1023."""
    global sim_phase
    if HAS_SPI and spi is not None:
        if not 0 <= channel <= 7:
            raise ValueError("MCP3008 channel must be between 0 and 7")
        response = spi.xfer2([1, (8 + channel) << 4, 0])
        return ((response[1] & 3) << 8) | response[2]
    else:
        # Realistic signal simulation if running on PC without SPI hardware
        if channel == PULSE_CHANNEL:
            freq = sim_bpm / 60.0
            sim_phase += 2.0 * math.pi * freq * SAMPLE_INTERVAL
            wave = math.sin(sim_phase) + 0.35 * math.sin(2.0 * sim_phase)
            return int(np_clip(512 + 200 * wave + random.uniform(-10, 10), 0, 1023))
        elif channel == GSR_CHANNEL:
            return int(np_clip(450 + random.uniform(-15, 15), 0, 1023))
        return 0


def np_clip(val, low, high):
    return max(low, min(high, val))


def pwm_from_arduino(value: int) -> float:
    """Translate Arduino analogWrite 0..255 into gpiozero PWM 0..1."""
    return max(0.0, min(1.0, value / 255.0))


def beep_10ms() -> None:
    """Brief 50 Hz buzzer pulse matching tone(..., 50) for 10 ms."""
    buzzer.value = 0.5
    time.sleep(0.010)
    buzzer.value = 0


def main() -> None:
    beat_count = 0
    galvanic_val = 0
    galvanic_vu = 0
    last_beat_time = 0.0
    was_above_threshold = False
    pulse_samples = []
    pulse_baseline = 0.0
    baseline_initialized = False

    print("Kinski Polygraph started — Raspberry Pi 4 Model B")
    if not (HAS_SPI and HAS_GPIO):
        print("[Notice] Running in simulation/test mode (spidev/gpiozero hardware not detected).")
    print(f"Pulse threshold: {THRESHOLD}")
    print("Press Ctrl+C to stop.")
    print("Note: this is an experimental physiological-response device, not a lie detector.")

    # Startup needle sweep, similar to original sketch's 0..19 sweep.
    for value in range(20):
        pulse_meter.value = pwm_from_arduino(value)
        gsr_meter.value = pwm_from_arduino(value)
        time.sleep(0.050)
    pulse_meter.value = 0
    gsr_meter.value = 0
    beep_10ms()

    try:
        while True:
            now = time.monotonic()
            pulse_raw = read_adc(PULSE_CHANNEL)

            # A small moving baseline reduces slow drift. The original Arduino
            # library uses a more advanced beat detector; this is a practical
            # Python approximation and needs calibration on the actual sensor.
            if not baseline_initialized:
                pulse_baseline = float(pulse_raw)
                baseline_initialized = True
            else:
                pulse_baseline += 0.01 * (pulse_raw - pulse_baseline)

            pulse_signal = pulse_raw
            above = pulse_signal >= THRESHOLD

            # Detect a rising threshold crossing with a refractory interval.
            # This preserves the configured threshold concept from the sketch.
            beat_detected = (
                above
                and not was_above_threshold
                and (now - last_beat_time) >= REFRACTORY_PERIOD
            )
            was_above_threshold = above

            if beat_detected:
                interval = now - last_beat_time if last_beat_time else 0
                last_beat_time = now
                bpm = (60.0 / interval) if interval > 0 else 0.0

                if bpm < BPM_LIMIT:
                    beat_count += 1
                    pulse_meter.value = pwm_from_arduino(PULSE_BEAT_PWM)

                    # Original sketch updates GSR once every three beats.
                    if beat_count >= GSR_READ_EVERY_N_BEATS:
                        galvanic_val = read_adc(GSR_CHANNEL) - GALVANIC_COMPENSATION
                        beat_count = 0

                    # Original sketch maps positive GSR values to 0..20.
                    if galvanic_val > 0:
                        galvanic_vu = int((galvanic_val * 20) / 1024)
                        galvanic_vu = max(0, min(20, galvanic_vu))
                    else:
                        galvanic_vu = 0

                    gsr_meter.value = pwm_from_arduino(galvanic_vu)

                    print(
                        f"Beat detected | BPM: {bpm:5.1f} | "
                        f"Galvanic: {galvanic_val:4d} | "
                        f"GSR meter: {galvanic_vu:2d}"
                    )

                    # Original condition: galvanicVal > 800.
                    if galvanic_val > 800:
                        beep_10ms()

            # Original loop returns pulse needle to analogWrite(..., 2).
            pulse_meter.value = pwm_from_arduino(
                PULSE_BEAT_PWM if beat_detected else PULSE_REST_PWM
            )
            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pulse_meter.value = 0
        gsr_meter.value = 0
        buzzer.value = 0
        if HAS_SPI and spi is not None:
            spi.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        pulse_meter.close()
        gsr_meter.close()
        buzzer.close()
