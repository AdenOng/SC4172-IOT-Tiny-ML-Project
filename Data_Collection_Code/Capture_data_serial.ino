#include "Arduino.h"
#include "Adafruit_I2CDevice.h"
#include "Adafruit_SPIDevice.h"
#include "Adafruit_Sensor.h"
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_LSM6DS33.h>

#define USER_BUTTON_PIN 7
#define RED_LED_PIN 13   // Red LED (status LED)
#define BLUE_LED_PIN 4   // Blue LED (connectivity/transmission LED)

// Sensor instances
Adafruit_LSM6DS33 lsm6ds33;
Adafruit_LIS3MDL lis3mdl;

// Structure to hold one IMU sample
struct IMUSample {
  float ax, ay, az;  // Accelerometer (m/s^2)
  float gx, gy, gz;  // Gyroscope (rad/s)
  float mx, my, mz;  // Magnetometer (uTesla)
  uint32_t timestamp;
};

const int NUM_SAMPLES = 900;  // Buffer size for 30 sec at ~30Hz
IMUSample sampleBuffer[NUM_SAMPLES];
int sampleIndex = 0;
bool recording = false;
unsigned long recordStart = 0;
bool transmissionStarted = false;

// Function to detect a trigger event on the user button (active LOW)
bool triggerDetected() {
  static bool buttonPreviouslyPressed = false;
  bool buttonState = digitalRead(USER_BUTTON_PIN);

  // Detect falling edge: when the button goes from HIGH to LOW
  if (!buttonPreviouslyPressed && (buttonState == LOW)) {
    delay(50);  // debounce
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

void setup() {
  Serial.begin(115200);
  
  // Setup button and LED pins
  pinMode(USER_BUTTON_PIN, INPUT_PULLUP);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  
  // Start with LEDs off
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);

  // Initialize LSM6DS33 (Accelerometer & Gyro)
  if (!lsm6ds33.begin_I2C()) {
    Serial.println("Failed to initialize LSM6DS33!");
    while (1) { delay(10); }
  }

  // Initialize LIS3MDL (Magnetometer)
  if (!lis3mdl.begin_I2C()) {
    Serial.println("Failed to initialize LIS3MDL!");
    while (1) { delay(10); }
  }
  
  // Configure Magnetometer settings
  lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);

  Serial.println("System ready. Press the user button to start recording...");
}

void loop() {
  // Start recording when the user presses the button
  if (!recording && triggerDetected()) {
    recording = true;
    sampleIndex = 0;
    recordStart = millis();
    digitalWrite(RED_LED_PIN, HIGH);  // Activate red LED during recording
    Serial.println("Recording started...");
  }

  // Recording phase: capture sensor data for 30 seconds at ~30Hz
  if (recording && (millis() - recordStart < 30000)) {
    sensors_event_t accel, gyro, dummy;
    lsm6ds33.getEvent(&accel, &gyro, &dummy);
    sensors_event_t magEvent;
    lis3mdl.getEvent(&magEvent);
    // Store sensor data in the buffer
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
    delay(33);  // Approximately 30Hz sampling rate (1000ms/30 ≈ 33ms)
  }
  else if (recording && (millis() - recordStart >= 30000)) {
    // Stop recording after 30 seconds
    recording = false;
    transmissionStarted = true;
    digitalWrite(RED_LED_PIN, LOW);  // Turn off red LED once recording is done
    Serial.println("Recording complete.");
  }

  // Transmission phase: Wait for a serial connection before sending data
  if (transmissionStarted) {
    // Wait until a Serial connection is detected (e.g., opening Serial Monitor)
    
    digitalWrite(BLUE_LED_PIN, HIGH);  // Turn on blue LED during transmission
    // Send a header indicating the CSV data format
    Serial.println("Data format: ax,ay,az,gx,gy,gz,mx,my,mz,timestamp");
    // Transmit each sample as a comma-separated line
    char asciiBuffer[128];
    for (int i = 0; i < sampleIndex; i++) {
      snprintf(
        asciiBuffer,
        sizeof(asciiBuffer),
        "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lu",
        sampleBuffer[i].ax,
        sampleBuffer[i].ay,
        sampleBuffer[i].az,
        sampleBuffer[i].gx,
        sampleBuffer[i].gy,
        sampleBuffer[i].gz,
        sampleBuffer[i].mx,
        sampleBuffer[i].my,
        sampleBuffer[i].mz,
        sampleBuffer[i].timestamp
      );
      Serial.println(asciiBuffer);
      delay(50);  // Delay to allow transmission to keep up
    }
    
    digitalWrite(BLUE_LED_PIN, LOW);  // Turn off blue LED after transmission
    transmissionStarted = false;
    Serial.println("Transmission complete.");
  }
}
