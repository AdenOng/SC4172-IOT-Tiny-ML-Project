#include <Arduino.h>
#include "Adafruit_Sensor.h"
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_LSM6DS33.h>

// TensorFlow Lite Micro includes
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/compatibility.h>

// BLE includes
#include <bluefruit.h>

// <<<--- MAKE SURE THIS IS THE CORRECT, NEWLY GENERATED MODEL FILE --- >>>
#include "imu_model.h" // Should contain g_imu_model_data and g_imu_model_data_len

// --- Pin Definitions ---
#define RED_LED_PIN 13     // Typically pinn 13 on Feather Sense
#define BLUE_LED_PIN 4     // Typically pin 4 for Blue LED on Feather Sense
#define USER_BUTTON_PIN 7  // Feather Sense User Switch

// --- TFLM Constants ---
#define NUM_CLASSES 4
#define NUM_TIMESTEPS 45   // Input window size for the model
#define NUM_FEATURES 9

// --- Timing Constants ---
#define SAMPLE_INTERVAL_MS 33 // Target sample interval (~30 Hz)
#define SLIDE_INTERVAL_MS 500 // How often to run inference (0.5 seconds)
const int SLIDE_STEP = (int)ceil((float)SLIDE_INTERVAL_MS / SAMPLE_INTERVAL_MS); // Should be ~16

// --- Feature Normalization Constants ---
// <<<--- PASTE LATEST VALUES FROM YOUR PYTHON SCRIPT OUTPUT --- >>>
const float FEATURE_MEANS[NUM_FEATURES] = {
    1.20960010e-01,  7.94896026e-01, -5.09449145e+00, // ax, ay, az means
    7.74947705e-04,  1.49835118e-01, -4.48857635e-01, // gx, gy, gz means
   -3.34466937e+00, -4.57668365e+00, -2.40626536e+01  // mx, my, mz means (EXAMPLE - REPLACE)
};
const float FEATURE_SCALES[NUM_FEATURES] = {
     7.87378730e+00,  1.02205258e+01,  7.44162342e+00, // ax, ay, az scales (std dev) (EXAMPLE - REPLACE)
     2.05283428e+00,  3.50854654e+00,  2.14226314e+00, // gx, gy, gz scales (std dev) (EXAMPLE - REPLACE)
     3.05463829e+01,  3.34312034e+01,  2.82488909e+01  // mx, my, mz scales (std dev) (EXAMPLE - REPLACE)
};

// Label mapping
const char* LABELS[NUM_CLASSES] = {"jump", "null", "spin", "weave"}; // <<<--- VERIFY/REPLACE --- >>>

// --- Sensor Instances ---
Adafruit_LSM6DS33 lsm6ds33;
Adafruit_LIS3MDL lis3mdl;

// --- BLE Instance ---
BLEUart bleuart; // BLE UART service

// --- TensorFlow Lite Globals ---
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 20 * 1024;
  #ifndef TF_LITE_MICRO_ALIGN
  #define TF_LITE_MICRO_ALIGN(x) __attribute__((aligned(x)))
  #endif
  TF_LITE_MICRO_ALIGN(8) uint8_t tensor_arena[kTensorArenaSize];

  // Circular buffer for sensor data
  float inference_buffer[NUM_TIMESTEPS][NUM_FEATURES];
  int buffer_write_index = 0;
  bool buffer_is_full = false;
} // namespace

// --- State and Latency Variables ---
bool ble_connected = false;
bool system_ready = false;
bool inference_running = false;
unsigned long last_sample_time_ms = 0;
int samples_since_last_inference = 0;
// --- Latency Measurement Variables ---
unsigned long last_sensor_read_us = 0;
unsigned long last_invoke_us = 0;
unsigned long last_ble_send_us = 0;
// --- Sliding Window Timing ---
unsigned long last_inference_start_ms = 0; // Timestamp of the previous inference start
unsigned long last_actual_slide_ms = 0;   // Measured duration between inferences
// ------------------------------------

// --- LED State Definitions ---
enum LedState { OFF, SETUP, WAITING_BLE, WAITING_BUTTON, SAMPLING, INFERRING, ERROR };
LedState current_led_state = OFF;
unsigned long last_blink_time = 0;
bool blink_state = false;

// --- Helper Functions ---

// Set LED state machine (Same as before)
void setLedState(LedState new_state) {
  if (new_state == current_led_state) return; current_led_state = new_state;
  last_blink_time = millis(); blink_state = true;
  digitalWrite(RED_LED_PIN, LOW); digitalWrite(BLUE_LED_PIN, LOW);
  switch (current_led_state) {
    case SETUP: digitalWrite(RED_LED_PIN, HIGH); break;
    case WAITING_BUTTON: case INFERRING: digitalWrite(BLUE_LED_PIN, HIGH); break;
    default: break; // Blinking/Off handled in updateLed
  }
}

// Update blinking LEDs (Same as before)
void updateLed() {
  unsigned long current_time = millis(); unsigned long blink_interval = 500;
  switch (current_led_state) {
    case WAITING_BLE: blink_interval = 750; break; // Slow blink blue
    case ERROR: blink_interval = 150; break;       // Fast blink red
    default: return; // Only handle blinking states
  }
  if (current_time - last_blink_time > blink_interval) {
    last_blink_time = current_time; blink_state = !blink_state;
    digitalWrite(current_led_state == WAITING_BLE ? BLUE_LED_PIN : RED_LED_PIN, blink_state ? HIGH : LOW);
  }
}

// Reads latest sensor data, puts it into buffer, AND MEASURES DURATION (Same as before)
bool ReadAndStoreSensorData() {
    sensors_event_t accel, gyro, mag;
    unsigned long sensor_read_start = micros();
    bool lsm_ok = lsm6ds33.getEvent(&accel, &gyro, nullptr); bool lis_ok = lis3mdl.getEvent(&mag);
    unsigned long sensor_read_end = micros(); last_sensor_read_us = sensor_read_end - sensor_read_start;
    if (!lsm_ok) { if (error_reporter) error_reporter->Report("ERR: LSM6DS33 Read Fail"); return false; }
    if (!lis_ok) { if (error_reporter) error_reporter->Report("ERR: LIS3MDL Read Fail"); return false; }
    inference_buffer[buffer_write_index][0] = accel.acceleration.x; inference_buffer[buffer_write_index][1] = accel.acceleration.y; inference_buffer[buffer_write_index][2] = accel.acceleration.z;
    inference_buffer[buffer_write_index][3] = gyro.gyro.x; inference_buffer[buffer_write_index][4] = gyro.gyro.y; inference_buffer[buffer_write_index][5] = gyro.gyro.z;
    inference_buffer[buffer_write_index][6] = mag.magnetic.x; inference_buffer[buffer_write_index][7] = mag.magnetic.y; inference_buffer[buffer_write_index][8] = mag.magnetic.z;
    buffer_write_index = (buffer_write_index + 1) % NUM_TIMESTEPS;
    if (buffer_write_index == 0 && !buffer_is_full) { buffer_is_full = true; }
    return true;
}

// Normalizes data from the circular buffer and loads it into the input tensor (Same as before)
void LoadCircularBufferToInputTensor() {
    if (input == nullptr || input->dims == nullptr || input->data.f == nullptr) { if (error_reporter) error_reporter->Report("ERR: Input tensor invalid"); return; }
    if (input->type != kTfLiteFloat32) { if (error_reporter) error_reporter->Report("ERR: Input tensor not Float32"); return; }
    float* input_data_ptr = input->data.f; size_t input_elements = 1;
    for (int i = 0; i < input->dims->size; ++i) { input_elements *= input->dims->data[i]; }
    if (input_elements != (size_t)(NUM_TIMESTEPS * NUM_FEATURES)) { if (error_reporter) error_reporter->Report("ERR: Input element count mismatch!"); return; }
    int buffer_read_start_index = buffer_is_full ? buffer_write_index : 0;
    for (int t = 0; t < NUM_TIMESTEPS; ++t) {
        int buffer_idx = (buffer_read_start_index + t) % NUM_TIMESTEPS;
        for (int f = 0; f < NUM_FEATURES; ++f) {
            int tensor_idx = t * NUM_FEATURES + f;
            input_data_ptr[tensor_idx] = (inference_buffer[buffer_idx][f] - FEATURE_MEANS[f]) / FEATURE_SCALES[f];
        }
    }
}

// --- BLE Communication ---

// Send data in BLE UART chunks (Same as before)
void bleSendDataChunked(const char* data) {
  int len = strlen(data); const int chunkSize = 20;
  for (int i = 0; i < len; i += chunkSize) {
    int bytesToSend = min(chunkSize, len - i); char chunk[chunkSize + 1];
    memcpy(chunk, data + i, bytesToSend); chunk[bytesToSend] = '\0';
    if (ble_connected && bleuart.notifyEnabled()) {
       bleuart.write(chunk, bytesToSend); delay(15); // Small delay
    } else { break; }
  }
}

// Format, send the prediction result over BLE, AND MEASURE DURATION (Same as before)
void sendPredictionBLE(const char* label, float probability) {
   if (!ble_connected || !bleuart.notifyEnabled()) return;
   unsigned long ble_send_start = micros();
   char bleBuffer[64];
   snprintf(bleBuffer, sizeof(bleBuffer), "P:%s,C:%.2f\n", label, probability);
   bleSendDataChunked(bleBuffer);
   unsigned long ble_send_end = micros();
   last_ble_send_us = ble_send_end - ble_send_start;
}

// BLE Connection Callback (Same as before)
void connect_callback(uint16_t conn_handle) {
  (void) conn_handle; ble_connected = true; system_ready = false;
  Serial.println("\nBLE Central Connected");
}

// BLE Disconnect Callback (Same as before)
void disconnect_callback(uint16_t conn_handle, uint8_t reason) {
  (void) conn_handle; (void) reason; ble_connected = false; system_ready = false;
  if (inference_running) { Serial.println("BLE Disconnected, Stopping Inference."); inference_running = false; }
  else { Serial.println("BLE Central Disconnected"); }
  setLedState(WAITING_BLE);
}

// Button Detection (Same as before)
bool checkButtonToggle() {
  static bool buttonPreviouslyPressed = false; bool stateShouldChange = false;
  bool buttonState = digitalRead(USER_BUTTON_PIN);
  if (!buttonPreviouslyPressed && (buttonState == LOW)) {
    delay(50); if (digitalRead(USER_BUTTON_PIN) == LOW) { buttonPreviouslyPressed = true; stateShouldChange = true; }
  }
  if (buttonPreviouslyPressed && (buttonState == HIGH)) { buttonPreviouslyPressed = false; }
  return stateShouldChange;
}

// --- Setup Function --- (Mostly same)
void setup() {
  Serial.begin(115200);
  pinMode(RED_LED_PIN, OUTPUT); pinMode(BLUE_LED_PIN, OUTPUT); pinMode(USER_BUTTON_PIN, INPUT_PULLUP);
  setLedState(SETUP); // Red solid during setup
  unsigned long start_time = millis(); while (!Serial && (millis() - start_time < 3000)) { ; }

  Serial.println("-------------------------------------------");
  Serial.println("Adafruit Feather Sense - IMU Classifier");
  Serial.println(" Continuous Inference + BLE Output + Latency");
  Serial.println("-------------------------------------------");
  Serial.printf("Model Window: %d steps (~%.1f s)\n", NUM_TIMESTEPS, (float)NUM_TIMESTEPS * SAMPLE_INTERVAL_MS / 1000.0);
  Serial.printf("Sample Interval: %d ms (~%d Hz)\n", SAMPLE_INTERVAL_MS, 1000/SAMPLE_INTERVAL_MS);
  Serial.printf("Slide Step: %d samples (~%.1f s target)\n", SLIDE_STEP, (float)SLIDE_STEP * SAMPLE_INTERVAL_MS / 1000.0); // Note: target slide time

  // Sensor Init (Same as before)
  Serial.print("Initializing LSM6DS33..."); if (!lsm6ds33.begin_I2C()) { Serial.println(" FAILED!"); setLedState(ERROR); while (1); } Serial.println(" OK.");
  Serial.print("Initializing LIS3MDL..."); if (!lis3mdl.begin_I2C()) { Serial.println(" FAILED!"); setLedState(ERROR); while (1); } Serial.println(" OK.");
  lsm6ds33.setAccelRange(LSM6DS_ACCEL_RANGE_4_G); lsm6ds33.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
  lsm6ds33.setAccelDataRate(LSM6DS_RATE_104_HZ); lsm6ds33.setGyroDataRate(LSM6DS_RATE_104_HZ);
  lis3mdl.setPerformanceMode(LIS3MDL_HIGHMODE); lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_80_HZ); lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  Serial.println("Sensors configured.");

  // TFLM Init (Same as before, check model data variable name)
  Serial.println("Setting up TensorFlow Lite Micro...");
  static tflite::MicroErrorReporter micro_error_reporter; error_reporter = &micro_error_reporter;
  error_reporter->Report("Loading model...");
  #ifndef TFLITE_SCHEMA_VERSION
  #define TFLITE_SCHEMA_VERSION 3
  #endif
  // --- ENSURE this variable name matches your .h file ---
  model = tflite::GetModel(imu_model_explicit2d_tflite); // Example name
  // ------------------------------------------------------
  if (model->version() != TFLITE_SCHEMA_VERSION) { error_reporter->Report("ERR: Model schema %d != Supported %d.", model->version(), TFLITE_SCHEMA_VERSION); setLedState(ERROR); while(1); }
  error_reporter->Report("Model Schema OK (Version %d).", model->version());
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddConv2D(); micro_op_resolver.AddAveragePool2D(); micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected(); micro_op_resolver.AddSoftmax();
  error_reporter->Report("Op Resolver created.");
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter; error_reporter->Report("Interpreter created.");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) { error_reporter->Report("ERR: AllocateTensors() FAILED! Status: %d", allocate_status); error_reporter->Report("Arena size: %d bytes. Used: %d.", kTensorArenaSize, interpreter->arena_used_bytes()); setLedState(ERROR); while (1); }
  error_reporter->Report("Tensor Arena allocated OK (%d bytes used / %d total).", interpreter->arena_used_bytes(), kTensorArenaSize);
  input = interpreter->input(0); output = interpreter->output(0);
  if (input == nullptr || input->dims == nullptr || output == nullptr) { error_reporter->Report("ERR: Failed to get valid input/output tensor."); setLedState(ERROR); while(1); }
  error_reporter->Report("Input/Output tensor pointers obtained.");

  // BLE Init (Same as before)
  Serial.println("Initializing BLE...");
  Bluefruit.begin(); Bluefruit.setName("FeatherSenseIMU");
  Bluefruit.Periph.setConnectCallback(connect_callback); Bluefruit.Periph.setDisconnectCallback(disconnect_callback);
  bleuart.begin();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE); Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart); Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true); Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.setFastTimeout(30); Bluefruit.Advertising.start(0);
  Serial.println("BLE Advertising started.");

  Serial.println("--- Setup Complete ---");
  setLedState(WAITING_BLE); // Blue slow blink - waiting for connection
  Serial.println("\nWaiting for BLE connection...");
}

// --- Arduino Loop ---
void loop() {
    updateLed(); // Handle LED blinking

    // --- Handle BLE Connection & System Ready States --- (Same as before)
    if (!ble_connected) { delay(50); return; }
    if (ble_connected && !system_ready) {
        system_ready = true; setLedState(WAITING_BUTTON);
        Serial.println("BLE Connected. Press User Button to Start/Stop Inference.");
    }

    // --- Handle User Button Press --- (Same as before)
    if (system_ready && checkButtonToggle()) {
        inference_running = !inference_running;
        if (inference_running) {
            Serial.println("--> Continuous Inference STARTED by user");
            buffer_write_index = 0; buffer_is_full = false; samples_since_last_inference = 0;
            last_sample_time_ms = millis();
            last_inference_start_ms = 0; // Reset slide timer when starting
            last_actual_slide_ms = 0;
            setLedState(SAMPLING);
        } else {
            Serial.println("--> Continuous Inference STOPPED by user");
            setLedState(WAITING_BUTTON);
        }
    }

    // --- Main Inference Loop ---
    if (!inference_running) { delay(10); return; }

    // --- Continuous Sampling ---
    unsigned long current_time_ms = millis();
    if (current_time_ms - last_sample_time_ms >= SAMPLE_INTERVAL_MS) {
        last_sample_time_ms = current_time_ms;
        bool read_ok = ReadAndStoreSensorData(); // Updates last_sensor_read_us

        if (read_ok) {
             samples_since_last_inference++;
             // --- Check if it's time to run inference ---
             if (buffer_is_full && samples_since_last_inference >= SLIDE_STEP) {
                samples_since_last_inference = 0; // Reset slide counter

                // --- Measure Actual Slide Time ---
                unsigned long current_inference_start_ms = millis();
                if (last_inference_start_ms != 0) { // Avoid calculation on the very first run
                    last_actual_slide_ms = current_inference_start_ms - last_inference_start_ms;
                }
                last_inference_start_ms = current_inference_start_ms; // Update for the next cycle
                // ---------------------------------

                setLedState(INFERRING);

                // Load data and run inference (Measure Invoke time)
                LoadCircularBufferToInputTensor();
                unsigned long invoke_start = micros();
                TfLiteStatus invoke_status = interpreter->Invoke();
                unsigned long invoke_end = micros();
                last_invoke_us = invoke_end - invoke_start; // Store Invoke Duration

                if (invoke_status != kTfLiteOk) {
                    error_reporter->Report("ERR: Invoke() failed! Status: %d", invoke_status);
                    setLedState(ERROR); inference_running = false;
                } else {
                    // Process output
                    float max_prob = -1.0f; int predicted_class_index = -1;
                    float* output_data_ptr = output->data.f;
                    if (output_data_ptr != nullptr && output->dims && output->dims->size > 0 && output->dims->data[output->dims->size - 1] == NUM_CLASSES) {
                        for (int i = 0; i < NUM_CLASSES; ++i) { if (output_data_ptr[i] > max_prob) { max_prob = output_data_ptr[i]; predicted_class_index = i; }}
                        if (predicted_class_index != -1) {
                             const char* predicted_label = LABELS[predicted_class_index];

                             // Send prediction over BLE (Updates last_ble_send_us)
                             sendPredictionBLE(predicted_label, max_prob);

                             // <<< MODIFIED: Print prediction AND ALL latencies to Serial >>>
                             Serial.printf("[%lu] Pred: %s (%.2f) | Slide: %lu ms | Lat(us) RD: %lu, INV: %lu, BLE: %lu\n",
                                           millis(),
                                           predicted_label,
                                           max_prob,
                                           last_actual_slide_ms,  // Print actual time between inferences
                                           last_sensor_read_us,   // Print last measured sensor read time
                                           last_invoke_us,        // Print measured invoke time
                                           last_ble_send_us);     // Print last measured BLE send time

                        } else { Serial.printf("[%lu] Prediction failed index check\n", millis()); }
                    } else { error_reporter->Report("ERR: Output tensor invalid!"); setLedState(ERROR); inference_running = false; }
                    // Set LED back to sampling state if no error occurred
                    if(inference_running) setLedState(SAMPLING); // Blue LED off
                    else setLedState(WAITING_BUTTON); // If stopped by error, go back to waiting
                } // End successful invoke processing
             } // End inference trigger check
        } else { delay(5); } // End successful read check
    } // End sample interval check

    if(millis() == current_time_ms && inference_running) { delayMicroseconds(50); } // Tiny delay if loop is too fast
} // End loop()