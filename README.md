# SC4172_IOT_Tiny_ML_Project

This project demonstrates the deployment of a **deep learning model** on a **low-power microcontroller** (Adafruit Feather Sense) for **real-time classification** of classic slalom inline skating movements.  
It uses **9-axis IMU (Inertial Measurement Unit)** data to classify moves into four categories:  
**Jumping**, **Spinning**, **Weaving**, and **Null**.

The system captures and processes sensor data, runs inference using a **custom-trained TensorFlow Lite Micro (TFLM)** model, and transmits the results via **Bluetooth Low Energy (BLE)**.

---

## Key Features

- **Real-Time Classification:** Performs on-device inference to classify skating moves as they happen.  
- **Low-Power Hardware:** Runs on the Adafruit Feather Sense (nRF52840), optimized for power efficiency.  
- **9-Axis Sensor Fusion:** Uses data from accelerometer, gyroscope, and magnetometer.  
- **Embedded ML:** Implements a custom CNN (Convolutional Neural Network) deployed with TensorFlow Lite Micro.  
- **Wireless Feedback:** Transmits classification results via BLE for real-time feedback.  
- **Robust Training:** Trained with data augmentation (axis permutations) to ensure robustness to sensor orientation.

---

## Hardware Requirements

| Component | Description |
|------------|--------------|
| **Microcontroller** | Adafruit Feather Sense (nRF52840) |
| **Accelerometer & Gyroscope** | LSM6DS33 (6-axis) |
| **Magnetometer** | LIS3MDL (3-axis) |

---

## Brief Setup Instructions

Before running the firmware on your Adafruit Feather Sense, ensure that your Arduino IDE and libraries are properly configured. Remeber to install the Tensorflow Lite Library, Sensor Hardware Library, Bluetooth Library.

Download and install the latest **Arduino IDE** from the [official Arduino website](https://www.arduino.cc/en/software).


1. Open **Arduino IDE → Preferences**.  
2. In the *“Additional Boards Manager URLs”* field, add:  

## How It Works

The project is divided into two main parts:

### 1. On-Device Firmware (Inference)

The C++ firmware on the Adafruit Feather Sense performs the following loop:

1. **Data Collection:** Samples accelerometer (x, y, z), gyroscope (x, y, z), and magnetometer (x, y, z) at 30 Hz.  
2. **Buffering:** Stores samples in a circular buffer.  
3. **Sliding Window:** After 16 new samples (~528 ms), extracts a 45-sample (1.5 s) window.  
4. **Preprocessing:** Normalizes data using means/scales from the training process.  
5. **Inference:** Feeds the normalized data into the TFLite model (~11.7 ms inference time).  
6. **Transmission:** Sends classification probabilities (Jump, Spin, Weave, Null) via BLE.

**Total Cycle Time:** ~28–30 ms (Sensor Acquisition + Inference + BLE)  
**Within Real-Time Window:** 528 ms sliding window interval.

---

### 2. Model Training Pipeline (Python)

1. **Data Loading:** Loads labeled IMU segments (e.g., `imu_segments.pkl`).  
2. **Augmentation:** Applies all six axis permutations to improve robustness to sensor placement.  
3. **Data Splitting:** Stratified into Train (65%), Validation (20%), and Test (15%).  
4. **Normalization:** `StandardScaler` fit on training data; means/scales saved for firmware.  
5. **Model Training:** Custom CNN trained with balanced class weights.  
6. **Conversion:** Keras `.h5` → TensorFlow Lite `.tflite` (with quantization).  
7. **Export:** `.tflite` converted into a C header (`imu_model.h`) using `xxd` for embedding in firmware.

---

## Model Architecture

| Layer | Details |
|-------|----------|
| **Input** | Shape: (1, 1, 45, 9) → 45 timesteps × 9 features |
| **Conv2D** | 8 filters, kernel 1×5, ReLU |
| **Conv2D** | 16 filters, kernel 1×5, ReLU |
| **GlobalAveragePooling2D** | — |
| **Dense** | 4 units (Softmax) – Outputs Jump, Spin, Weave, Null |

**Regularization:** Dropout (20–25%) used during training.  
**Compatibility:** Designed for TFLM – avoids unsupported 1D ops by reshaping to Conv2D.

---

## 📊 Performance

| Class | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| Jump | 0.95 | 0.92 | 0.94 |
| Null | 0.74 | 0.79 | 0.76 |
| Spin | 0.85 | 0.90 | 0.88 |
| Weave | 0.89 | 0.76 | 0.82 |
| **Weighted Avg** | **0.85** | **0.84** | **0.84** |

**Test Accuracy:** 84.21%  
**Model Size:** ≈ 7 KB (quantized, unpruned)

---

## Optimization Attempted

Magnitude pruning (up to 80% sparsity) was tested using **TFMOT** but:

- Accuracy dropped to **79.43%**
- File size unchanged (7,088 B → 7,072 B)
- Platform lacks **sparse kernel support**, so zeroed weights are still stored.

**Conclusion:** The **unpruned, quantized model** is optimal for this microcontroller.

---

## Summary

| Metric | Value |
|--------|--------|
| Sampling Rate | 30 Hz |
| Window Size | 45 samples (1.5 s) |
| Inference Time | ~11.7 ms |
| Total Cycle Time | ~28–30 ms |
| Model Size | ~7 KB |
| Accuracy | 84.21% |

