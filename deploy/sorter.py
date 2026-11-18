"""Real-time tomato sorter for Raspberry Pi 4 — original thesis program (Lampiran 2),
restored from the PDF listing.

Changes from the printed listing:
  * indentation restored (the PDF layout broke several blocks)
  * --simulate: runs the same main loop with fake hardware, so it works on any computer
  * --items N: stop after N tomatoes (default: run until Ctrl+C, as in the thesis)
  * the model file may be either a plain scikit-learn model (thesis) or the bundle saved by src/tomato

On the Pi:     sudo systemctl start pigpiod && python deploy/sorter.py --model model_svm_pso.pkl
Anywhere:      python deploy/sorter.py --model models/svm_pso.joblib --simulate --items 10
"""

import argparse
import csv
import os
import random
import signal
import sys
import time
from datetime import datetime

import joblib
import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="model_svm_pso.pkl")
parser.add_argument("--simulate", action="store_true", help="fake sensors and actuators; no Raspberry Pi needed")
parser.add_argument("--items", type=int, default=0, help="stop after N tomatoes (0 = run until Ctrl+C)")
args = parser.parse_args()

if not args.simulate:
    import adafruit_tcs34725
    import board
    import busio
    import pigpio
    from hx711 import HX711
    from RPLCD.i2c import CharLCD

# KONFIGURASI
L298N_ENA = 12
L298N_IN1 = 13
L298N_IN2 = 16
SERVO1_PIN = 18
SERVO2_PIN = 19
HX711_DT = 20
HX711_SCK = 21
IR_SENSOR_PIN = 27

CONVEYOR_PWM_FREQ = 1000
KECEPATAN_NORMAL = 180
KECEPATAN_LAMBAT = 60

HX711_REFERENCE_UNIT = 1.0
HX711_TARE_OFFSET = 0
BERAT_MIN = 30.0
BERAT_MAX = 72.0

TCS_INTEGRATION_TIME = 50
TCS_GAIN = 4
C_MIN_THRESHOLD = 500
N_READINGS = 5

LCD_I2C_ADDRESS = 0x27
LCD_COLS = 16
LCD_ROWS = 2

CLASS_LABELS = {0: "Tidak Matang", 1: "Setengah Matang", 2: "Matang"}
MODEL_PATH = args.model

IR_DEBOUNCE_S = 0.05
DELAY_KE_TCS_S = 0.0 if args.simulate else 3.00
SERVO_HOLD_TIME = 0.0 if args.simulate else 0.80
COOLDOWN_AFTER_DETECT_S = 0.0 if args.simulate else 1.00


# KELAS
class LoadCell:
    def __init__(self):
        self.hx = HX711(dout_pin=HX711_DT, pd_sck_pin=HX711_SCK)
        self.hx.reset()

    def baca_gram(self, readings=5):
        raw = self.hx.get_raw_data_mean(readings)
        if raw is False or raw is None:
            return None
        return (raw - HX711_TARE_OFFSET) / HX711_REFERENCE_UNIT

    def cleanup(self):
        try:
            self.hx.power_down()
        except Exception:
            pass


class SensorIR:
    def __init__(self, pi):
        self.pi = pi
        self.pin = IR_SENSOR_PIN
        self.pi.set_mode(self.pin, pigpio.INPUT)
        self.pi.set_pull_up_down(self.pin, pigpio.PUD_UP)

    def ada_objek(self):
        return self.pi.read(self.pin) == 0  # LOW = ada objek


class SensorWarna:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_tcs34725.TCS34725(i2c)
        self.sensor.integration_time = TCS_INTEGRATION_TIME
        self.sensor.gain = TCS_GAIN

    def baca_raw(self):
        return self.sensor.color_raw

    def baca_normalisasi(self):
        rs = gs = bs = cs = 0
        for _ in range(N_READINGS):
            r, g, b, c = self.baca_raw()
            rs += r
            gs += g
            bs += b
            cs += c
            time.sleep(0.02)
        r, g, b, c = rs / N_READINGS, gs / N_READINGS, bs / N_READINGS, cs / N_READINGS
        if c < C_MIN_THRESHOLD or c == 0:
            return {"valid": False, "R": r, "G": g, "B": b, "C": c, "R_norm": 0.0, "G_norm": 0.0, "B_norm": 0.0}
        return {"valid": True, "R": r, "G": g, "B": b, "C": c, "R_norm": r / c, "G_norm": g / c, "B_norm": b / c}


class Conveyor:
    def __init__(self, pi):
        self.pi = pi
        self.pi.set_mode(L298N_IN1, pigpio.OUTPUT)
        self.pi.set_mode(L298N_IN2, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(L298N_ENA, CONVEYOR_PWM_FREQ)
        self.pi.set_PWM_range(L298N_ENA, 255)
        self.berhenti()

    def _duty(self, duty):
        self.pi.set_PWM_dutycycle(L298N_ENA, max(0, min(255, int(duty))))

    def maju(self, kecepatan=KECEPATAN_NORMAL):
        self.pi.write(L298N_IN1, 1)
        self.pi.write(L298N_IN2, 0)
        self._duty(kecepatan)

    def lambatkan(self, kecepatan=KECEPATAN_LAMBAT):
        self.pi.write(L298N_IN1, 1)
        self.pi.write(L298N_IN2, 0)
        self._duty(kecepatan)

    def berhenti(self):
        self._duty(0)
        self.pi.write(L298N_IN1, 0)
        self.pi.write(L298N_IN2, 0)


class Servo:
    def __init__(self, pin, pi):
        self.pin = pin
        self.pi = pi

    def set_angle(self, angle):
        angle = max(0, min(180, angle))
        pw = int(500 + (angle / 180.0) * 2000)
        self.pi.set_servo_pulsewidth(self.pin, pw)

    def rest(self):
        self.set_angle(90)

    def detach(self):
        self.pi.set_servo_pulsewidth(self.pin, 0)


def sortasi(servo1, servo2, kelas_id):
    if kelas_id == 2:  # Matang
        servo1.set_angle(145)
        servo2.set_angle(90)
    elif kelas_id == 1:  # Setengah Matang
        servo1.set_angle(90)
        servo2.set_angle(145)
    else:  # Tidak Matang
        servo1.set_angle(90)
        servo2.set_angle(90)


class LCD:
    def __init__(self):
        self.lcd = CharLCD(i2c_expander="PCF8574", address=LCD_I2C_ADDRESS, cols=LCD_COLS, rows=LCD_ROWS)
        self.lcd.clear()

    def tampil(self, baris1="", baris2=""):
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(baris1[:LCD_COLS])
        if LCD_ROWS > 1:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(baris2[:LCD_COLS])

    def cleanup(self):
        try:
            self.lcd.clear()
        except Exception:
            pass


class Klasifikator:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model '{MODEL_PATH}' tidak ditemukan.")
        loaded = joblib.load(MODEL_PATH)
        self.model = loaded["model"] if isinstance(loaded, dict) else loaded

    def prediksi(self, r_norm, g_norm, b_norm):
        x = np.array([[r_norm, g_norm, b_norm]])
        kelas = int(self.model.predict(x)[0])
        return kelas, CLASS_LABELS.get(kelas, "Unknown")


# ---- Simulation: stand-ins for the hardware classes (same methods) -------------
class _FakePi:
    connected = True
    INPUT = OUTPUT = PUD_UP = 0

    def __getattr__(self, name):  # set_mode, write, set_PWM_*, set_servo_pulsewidth, stop ...
        return lambda *a, **k: None

    def read(self, pin):
        return 0  # IR always sees the next tomato


class _FakeLoadCell:
    def baca_gram(self, readings=5):
        return random.uniform(25, 80)  # some readings fall outside 30-72 g and are skipped

    def cleanup(self):
        pass


class _FakeSensorWarna(SensorWarna):
    """Readings drawn around the per-class means from the thesis (Table 4.2)."""

    MEANS = {2: (0.6588, 0.2339, 0.1503), 1: (0.5516, 0.3070, 0.1384), 0: (0.4699, 0.3592, 0.1706)}

    def __init__(self):
        self.kelas_asli = 0

    def baca_raw(self):
        c = random.uniform(600, 1500)
        r, g, b = (max(0.0, random.gauss(m, 0.03)) * c for m in self.MEANS[self.kelas_asli])
        return r, g, b, c

    def baca_normalisasi(self):
        self.kelas_asli = random.choice([0, 1, 2])
        return super().baca_normalisasi()


class _FakeLCD:
    def tampil(self, baris1="", baris2=""):
        print(f"  [LCD] {baris1[:LCD_COLS]:<16} | {baris2[:LCD_COLS]}")

    def cleanup(self):
        pass


if args.simulate:
    pigpio = _FakePi()
    pigpio.pi = lambda: _FakePi()
    LoadCell, SensorWarna, LCD = _FakeLoadCell, _FakeSensorWarna, _FakeLCD
    random.seed(0)

# MAIN
running = True
counter = {0: 0, 1: 0, 2: 0}
log_filename = f"log_pengujian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def handle_stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


def init_log():
    with open(log_filename, "w", newline="") as f:
        csv.writer(f).writerow(["No", "Waktu", "Berat_g", "R", "G", "B", "C", "R_norm", "G_norm", "B_norm", "Kelas_ID", "Kelas_Label"])


def tulis_log(no, berat, d, kelas_id, label):
    with open(log_filename, "a", newline="") as f:
        csv.writer(f).writerow([
            no, datetime.now().strftime("%H:%M:%S"), f"{berat:.2f}",
            f"{d['R']:.0f}", f"{d['G']:.0f}", f"{d['B']:.0f}", f"{d['C']:.0f}",
            f"{d['R_norm']:.4f}", f"{d['G_norm']:.4f}", f"{d['B_norm']:.4f}",
            kelas_id, label,
        ])


def main():
    global running

    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: jalankan pigpiod dulu (sudo systemctl start pigpiod)")
        sys.exit(1)

    try:
        lcd = LCD()
        lcd.tampil("Memuat sistem", "Mohon tunggu...")
        klasifikator = Klasifikator()
        load_cell = LoadCell()
        sensor_ir = SensorIR(pi)
        sensor_warna = SensorWarna()
        conveyor = Conveyor(pi)
        servo1 = Servo(SERVO1_PIN, pi)
        servo2 = Servo(SERVO2_PIN, pi)
        servo1.rest()
        servo2.rest()
        init_log()
        lcd.tampil("Sistem siap", "Conveyor jalan")
        time.sleep(0 if args.simulate else 1)
    except Exception as e:
        print(f"Inisialisasi gagal: {e}")
        sys.exit(1)
    no = 0
    conveyor.maju()

    try:
        while running:
            # LOAD CELL
            berat = load_cell.baca_gram(readings=5)
            if berat is None or not (BERAT_MIN <= berat <= BERAT_MAX):
                time.sleep(0.05)
                continue

            # SENSOR IR
            if not sensor_ir.ada_objek():
                time.sleep(0.02)
                continue
            time.sleep(IR_DEBOUNCE_S)
            if not sensor_ir.ada_objek():
                continue

            no += 1
            print(f"\n--- Tomat #{no} | Berat: {berat:.1f} g ---")

            conveyor.lambatkan()
            time.sleep(DELAY_KE_TCS_S)
            data = sensor_warna.baca_normalisasi()
            if not data["valid"]:
                print(" Pembacaan warna tidak valid. Skip.")
                lcd.tampil(f"#{no}: error", "Baca warna gagal")
                conveyor.maju()
                time.sleep(COOLDOWN_AFTER_DETECT_S)
                continue
            print(f"     R_norm={data['R_norm']:.4f} G_norm={data['G_norm']:.4f} B_norm={data['B_norm']:.4f}")

            # KLASIFIKASI SVM-PSO
            kelas_id, label = klasifikator.prediksi(data["R_norm"], data["G_norm"], data["B_norm"])
            counter[kelas_id] += 1
            print(f" -> KELAS: {label}" + (f"   (simulasi, kelas asli: {CLASS_LABELS[sensor_warna.kelas_asli]})" if args.simulate else ""))
            lcd.tampil(f"#{no}: {label[:12]}", f"M:{counter[2]} S:{counter[1]} T:{counter[0]}")

            sortasi(servo1, servo2, kelas_id)
            conveyor.maju()
            time.sleep(SERVO_HOLD_TIME)
            servo1.rest()
            servo2.rest()

            tulis_log(no, berat, data, kelas_id, label)
            time.sleep(COOLDOWN_AFTER_DETECT_S)
            if args.items and no >= args.items:
                running = False

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    finally:
        try:
            conveyor.berhenti()
        except Exception:
            pass
        try:
            servo1.detach()
            servo2.detach()
        except Exception:
            pass
        try:
            load_cell.cleanup()
        except Exception:
            pass
        try:
            lcd.tampil("Sistem berhenti", f"Total: {sum(counter.values())}")
        except Exception:
            pass
        time.sleep(0 if args.simulate else 1)
        try:
            lcd.cleanup()
        except Exception:
            pass
        try:
            pi.stop()
        except Exception:
            pass
        print(f"\nTotal: {no} tomat | Matang: {counter[2]} | Setengah: {counter[1]} | Tidak Matang: {counter[0]}")
        print(f"Log: {log_filename}")


if __name__ == "__main__":
    main()
