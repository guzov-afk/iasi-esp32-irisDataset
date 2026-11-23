"""Evaluate the physical ESP32 on the exported 30-sample holdout split.

Run only after flashing ESP32 and closing PlatformIO Serial Monitor.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path
import serial

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--port', required=True, help='e.g. COM5 or /dev/ttyUSB0')
p.add_argument('--baud', type=int, default=115200)
a = p.parse_args()

with (ROOT / 'results' / 'iris_test.csv').open(newline='', encoding='utf8') as f:
    cases = list(csv.DictReader(f))
cm = [[0] * 3 for _ in range(3)]
latencies = []
correct = 0
with serial.Serial(a.port, a.baud, timeout=4.0) as ser:
    time.sleep(2.5)  # Opening USB/UART can reset an ESP32
    ser.reset_input_buffer()
    for idx, row in enumerate(cases, 1):
        labels = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        message = ','.join(row[key] for key in labels) + '\n'
        ser.write(message.encode('ascii'))
        deadline = time.monotonic() + 5
        result = None
        while time.monotonic() < deadline:
            line = ser.readline().decode('utf8', errors='replace').strip()
            if not line.startswith('RESULT,'):
                continue
            chunks = line.split(',')
            if len(chunks) == 6:
                try:
                    result = (int(chunks[1]), float(chunks[5]))
                except ValueError:
                    continue
                break
        if result is None:
            raise RuntimeError(f'Missing RESULT for sample {idx}: {message.strip()}')
        predicted, latency_us = result
        actual = int(row['label'])
        if predicted not in range(3):
            raise RuntimeError(f'Unexpected class: {predicted}')
        cm[actual][predicted] += 1
        latencies.append(latency_us)
        correct += predicted == actual
        print(f'{idx:02d}: expected={actual} predicted={predicted} latency={latency_us:g}us')

print(f'ESP32 HOLDOUT ACCURACY = {correct}/{len(cases)} = {correct/len(cases):.3f}')
print(f'Average inference latency: {statistics.mean(latencies):.1f} us')
print('Confusion matrix (rows=true, columns=predicted):')
for row in cm:
    print(' '.join(f'{v:3d}' for v in row))
