# IASI – Iris Classification on a Physical ESP32

A master's laboratory handout and **runnable local MLP inference baseline** for an ESP32-WROOM-32 (or another appropriately configured ESP32 board).

## Included

- `IASI_Iris_ESP32_Lab.tex`: English LaTeX handout in the same style as the supplied laboratory model.
- `train_iris.py`: trains a scikit-learn MLP, splits train/test without scaler leakage, and exports exact float32 weights/biases and normalization.
- `src/main.cpp`: Arduino-framework forward-pass implementation, with USB Serial input and latency output.
- `platformio.ini`: ESP32 Dev Module configuration (modify `board` for your exact physical board).
- `evaluate_device.py`: sends 30 unseen examples to a flashed physical ESP32, reports accuracy/confusion matrix/latency.
- `include/iris_weights.h`: exported weights from the default training run.
- `results/iris_test.csv`, `results/metrics.json`: generated reproducible holdout samples and host metrics.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_iris.py
```

Open folder in VS Code + PlatformIO, select the correct board, connect USB:

```bash
pio device list
pio run
pio run -t upload --upload-port COM5        # change port as needed
pio device monitor -p COM5 -b 115200         # send LF-terminated lines
```

Sample serial input: `5.1,3.5,1.4,0.2`. The ESP32 prints:

`RESULT,<class_id>,<prob_setosa>,<prob_versicolor>,<prob_virginica>,<latency_us>`

Close the monitor before automated testing (the USB port cannot be shared):

```bash
python evaluate_device.py --port COM5
```

## Changing architecture

```bash
python train_iris.py --h1 8 --h2 4
python train_iris.py --h1 32 --h2 16 --lr 0.001
```

After retraining, **rebuild and re-flash** the ESP32, because weights are embedded in firmware.

## Validation and limitations

The default model was trained on this host and its float32 forward pass checked against scikit-learn on the holdout samples. At baseline the 4–16–8–3 network achieved **29/30 (96.7%)** test accuracy on PC, with **972 bytes** of float32 weight/bias payload (not the total executable size). Your results may differ when changing parameters. An ESP32 **physical flash/build test was not possible** in this environment; confirm the PlatformIO board ID, USB drivers, serial port, and runtime results with actual hardware before teaching.

**Important distinction:** the baseline uses direct C++ inference (not a `.tflite` interpreter). This ensures the first lab does not depend on an older, incompatible Arduino TensorFlow library. The LaTeX document contains an *optional* advanced TensorFlow Lite Micro exercise referencing the official Espressif project. For comparison, the community Iris ESP32 example is at https://github.com/ArianKheir/TinyML-on-ESP32-Iris-Dataset-Classification; its dependencies and UART port need attention for modern physical boards.

## Compile the LaTeX handout

```bash
pdflatex -interaction=nonstopmode IASI_Iris_ESP32_Lab.tex
pdflatex -interaction=nonstopmode IASI_Iris_ESP32_Lab.tex
```
