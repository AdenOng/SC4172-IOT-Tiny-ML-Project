#include "Arduino.h"
#include "Adafruit_I2CDevice.h"
#include "Adafruit_SPIDevice.h"
#include "Adafruit_Sensor.h"
#include <bluefruit.h>  // Use Adafruit Bluefruit BLE library
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_LSM6DS33.h>

// Optionally include BLEUart header if not included by bluefruit.h
// #include <BLEUart.h>

#define USER_BUTTON_PIN 7

// Sensor instances
Adafruit_LSM6DS33 lsm6ds33;
Adafruit_LIS3MDL lis3mdl;

// Create a global BLE UART instance
BLEUart bleuart;

// Structure to hold one IMU sample (temperature removed)
struct IMUSample {
  float ax, ay, az;  // Accelerometer (m/s^2)
  float gx, gy, gz;  // Gyroscope (rad/s)
  float mx, my, mz;  // Magnetometer (uTesla)
  uint32_t timestamp;
};

const int NUM_SAMPLES = 900; // 30s * 30Hz sampling rate
IMUSample sampleBuffer[NUM_SAMPLES];
int sampleIndex = 0;
bool recording = false;
unsigned long recordStart = 0;
bool transmissionStarted = false;

// Function to detect trigger event (Replace with your own implementation)
bool triggerDetected() {
  static bool buttonPreviouslyPressed = false;
  bool buttonState = digitalRead(USER_BUTTON_PIN); // HIGH when not pressed, LOW when pressed

  // Detect falling edge: button goes from HIGH to LOW
  if (!buttonPreviouslyPressed && (buttonState == LOW)) {
    delay(50); // debounce
    if (digitalRead(USER_BUTTON_PIN) == LOW) {
      buttonPreviouslyPressed = true;
      return true;
    }
  }
  if (buttonState == HIGH) {
    buttonPreviouslyPressed = false;
  }
  return false;
}

// Helper function to send data in chunks (max 20 bytes per chunk)
void bleSendDataChunked(const char* data) {
  int len = strlen(data);
  const int chunkSize = 20;
  for (int i = 0; i < len; i += chunkSize) {
    int bytesToSend = (len - i) < chunkSize ? (len - i) : chunkSize;
    char chunk[chunkSize + 1]; // +1 for null terminator
    memcpy(chunk, data + i, bytesToSend);
    chunk[bytesToSend] = '\0';
    bleuart.print(chunk);
    delay(10); // slight delay to avoid congestion
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(USER_BUTTON_PIN, INPUT_PULLUP);
  while (!Serial);

  // Initialize LSM6DS33 (Accel, Gyro)
  if (!lsm6ds33.begin_I2C()) {
    Serial.println("Failed to initialize LSM6DS33!");
    while (1) { delay(10); }
  }
  // Optionally set accelerometer and gyro ranges/data rates here:
  // lsm6ds33.setAccelRange(LSM6DS_ACCEL_RANGE_2_G);
  // lsm6ds33.setGyroRange(LSM6DS_GYRO_RANGE_250_DPS);
  // lsm6ds33.setAccelDataRate(LSM6DS_RATE_104_HZ);
  // lsm6ds33.setGyroDataRate(LSM6DS_RATE_104_HZ);

  // Initialize LIS3MDL (Magnetometer)
  if (!lis3mdl.begin_I2C()) {
    Serial.println("Failed to initialize LIS3MDL!");
    while (1) { delay(10); }
  }
  // Set magnetometer configuration
  lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);

  // Initialize BLE
  Bluefruit.begin();
  Bluefruit.setName("IMURecorder");

  // Optionally, if supported by your Bluefruit library version, set a higher ATT MTU 
  // to allow for larger data packets (e.g., 247 bytes).
  // Bluefruit.setAttMtu(247);

  // Initialize BLE UART service
  bleuart.begin();

  // Setup BLE Advertising to include the BLE UART service
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);  // BLEUart instance automatically provides its service.
  Bluefruit.Advertising.start();

  Serial.println("BLE active, waiting for connection...");
}

void loop() {
  // Wait for trigger to start recording
  if (!recording && triggerDetected()) {
    recording = true;
    sampleIndex = 0;
    recordStart = millis();
    Serial.println("Recording started...");
  }

  // Recording phase: sample data for 30 seconds at ~30Hz
  if (recording && (millis() - recordStart < 30000)) {
    // Get LSM6DS33 sensor events (accel and gyro); ignore temperature by using a dummy variable.
    sensors_event_t accel, gyro, dummy;
    lsm6ds33.getEvent(&accel, &gyro, &dummy);
    
    // Get magnetometer event from LIS3MDL
    sensors_event_t magEvent;
    lis3mdl.getEvent(&magEvent);
    
    // Store sensor data in the sample buffer
    sampleBuffer[sampleIndex].ax = accel.acceleration.x;
    sampleBuffer[sampleIndex].ay = accel.acceleration.y;
    sampleBuffer[sampleIndex].az = accel.acceleration.z;
    sampleBuffer[sampleIndex].gx = gyro.gyro.x;
    sampleBuffer[sampleIndex].gy = gyro.gyro.y;
    sampleBuffer[sampleIndex].gz = gyro.gyro.z;
    sampleBuffer[sampleIndex].mx = magEvent.magnetic.x;
    sampleBuffer[sampleIndex].my = magEvent.magnetic.y;
    sampleBuffer[sampleIndex].mz = magEvent.magnetic.z;
    sampleBuffer[sampleIndex].timestamp = millis();
    sampleIndex++;

    delay(33); // Approximately 30Hz sampling rate (1000ms/30 ≈ 33ms)
  }
  else if (recording && (millis() - recordStart >= 30000)) {
    recording = false;
    transmissionStarted = true;
    Serial.println("Recording complete. Starting transmission...");
  }

  // Transmission phase: send each sample over BLE UART in ASCII format using 20-byte chunks
  if (transmissionStarted) {
    char asciiBuffer[128]; // Buffer to hold the formatted ASCII string
    for (int i = 0; i < sampleIndex; i++) {
      // Format the sample data into a readable ASCII string (comma-separated values)
      snprintf(asciiBuffer, sizeof(asciiBuffer),
        "Sample %d: ax=%.2f, ay=%.2f, az=%.2f; gx=%.2f, gy=%.2f, gz=%.2f; mx=%.2f, my=%.2f, mz=%.2f; t=%lu\n",
        i,
        sampleBuffer[i].ax, sampleBuffer[i].ay, sampleBuffer[i].az,
        sampleBuffer[i].gx, sampleBuffer[i].gy, sampleBuffer[i].gz,
        sampleBuffer[i].mx, sampleBuffer[i].my, sampleBuffer[i].mz,
        sampleBuffer[i].timestamp
      );
      
      // Print the full ASCII sample to the Serial Monitor
      Serial.print(asciiBuffer);
      // Transmit the ASCII sample over BLE UART in chunks
      bleSendDataChunked(asciiBuffer);
      delay(50); // Delay to avoid saturating the BLE connection
    }
    transmissionStarted = false;
    Serial.println("Transmission complete.");
  }
}