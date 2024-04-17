#include <Wire.h> // Include the Wire library for I2C devices
#include <BluetoothSerial.h> // Include the Bluetooth Serial library for ESP32
#include <Adafruit_TCS34725.h> // Include the Adafruit TCS34725 library
#include <Preferences.h> // Include the Preferences library for non-volatile storage

// Define pin numbers
#define BUTTON_PIN 12
#define MOTOR_FWD_PIN 27 //26
#define MOTOR_BCK_PIN 26 //27
#define RGB_LED 25
#define SWITCH_OPEN_PIN 14
#define SWITCH_CLOSE_PIN 13

#define PWM_CHANNEL 0
#define PWM_FREQUENCY 5000
#define PWM_RESOLUTION 8


// Create a Bluetooth Serial object
BluetoothSerial SerialBT;
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_240MS, TCS34725_GAIN_1X);
Preferences preferences;

// Define motor state
enum MotorState {STOPPED, FORWARD, BACKWARD};
MotorState motorState = STOPPED;

volatile bool buttonPressed = false; // volatile as it is used in interrupt routine
unsigned long lastInterruptTime = 0; // debounce: last time the interrupt was triggered

void IRAM_ATTR isr() { // Interrupt service routine
  // Debounce: If interrupts come faster than 200ms, assume it's a bounce and ignore
  Serial.println("ISR");
  if (millis() - lastInterruptTime > 200) {
    Serial.println(digitalRead(BUTTON_PIN));
    buttonPressed = true;
  }
  lastInterruptTime = millis();
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Initialize Bluetooth communication
  SerialBT.begin("ESP32");

  // Initialize button and switches
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(SWITCH_OPEN_PIN, INPUT_PULLUP);
  pinMode(SWITCH_CLOSE_PIN, INPUT_PULLUP);
  pinMode(RGB_LED, OUTPUT);

  if (tcs.begin()) {
    Serial.println("Found sensor");
  } else {
    Serial.println("No TCS34725 found ... check your connections");
    while (1); // halt!
  }

  ledcSetup(PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_FWD_PIN, PWM_CHANNEL);


  // Attach interrupt to button pin
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), isr, RISING);

  // Initialize motor control pins
  //pinMode(MOTOR_FWD_PIN, OUTPUT);
  pinMode(MOTOR_BCK_PIN, OUTPUT);

  // Ensure motor is stopped at startup
  digitalWrite(MOTOR_FWD_PIN, LOW);
  digitalWrite(MOTOR_BCK_PIN, LOW);
}

void loop() {
  // Check for Bluetooth commands
  if (SerialBT.available()) {
    String command = SerialBT.readString();
      if (command == "start") {
        uint16_t r, g, b, c;
        tcs.getRawData(&r, &g, &b, &c);
        preferences.begin("calibration", true); // Open in read-only mode
        uint16_t r_min = preferences.getUInt("r_min", 0);
        uint16_t r_max = preferences.getUInt("r_max", 65535);
        uint16_t g_min = preferences.getUInt("g_min", 0);
        uint16_t g_max = preferences.getUInt("g_max", 65535);
        uint16_t b_min = preferences.getUInt("b_min", 0);
        uint16_t b_max = preferences.getUInt("b_max", 65535);
        preferences.end();
         // Start the motor moving forward
        startMotor(FORWARD);

        // Keep reading sensor, normalizing and transmitting the readings until the end switch is hit
        digitalWrite(RGB_LED, HIGH);
        Serial.println("Starting");
        while (digitalRead(SWITCH_CLOSE_PIN) == HIGH) {
          uint16_t r, g, b, c;
          tcs.getRawData(&r, &g, &b, &c);
          float r_norm = normalize(r, r_min, r_max);
          float g_norm = normalize(g, g_min, g_max);
          float b_norm = normalize(b, b_min, b_max);
          SerialBT.println(String(r_norm) + "," +String(g_norm) + "," + String(b_norm));
        }
        delay(100);
        SerialBT.println("Terminate");
        stopMotor();
        digitalWrite(RGB_LED, LOW);
        Serial.println("Done");
      }
      else if (command == "calibrate") {
      calibrate();
      }
  }

  // Check button state
  if (buttonPressed) {
    buttonPressed = false; // reset the flag
    Serial.println("Button Pressed");
    switch (motorState) {
      case STOPPED:
        if (digitalRead(SWITCH_OPEN_PIN) == LOW) {
          Serial.println("forward");
          startMotor(FORWARD);
        } else if(digitalRead(SWITCH_CLOSE_PIN) == LOW){
          Serial.println("backward");
          startMotor(BACKWARD);
        }else{
          Serial.println("backward");
          startMotor(FORWARD);
        }
        break;
      case FORWARD:
        Serial.println("backward");
        startMotor(BACKWARD);
        break;
      case BACKWARD:
        Serial.println("forwawrd");
        startMotor(FORWARD);
        break;
    }
  }

  // Check end switch states
  if ((motorState == FORWARD && digitalRead(SWITCH_CLOSE_PIN) == LOW) ||
      (motorState == BACKWARD && digitalRead(SWITCH_OPEN_PIN) == LOW)) {
    stopMotor();
  }
}

void startMotor(MotorState direction) {
  // Ensure motor is stopped
  stopMotor();

  // Start motor in the specified direction
  if (direction == FORWARD) {
    Serial.println("Rotating forward");
    ledcWrite(PWM_CHANNEL, 200); // Control MOTOR_FWD_PIN speed via PWM
    digitalWrite(MOTOR_BCK_PIN, LOW);
  } else {
    Serial.println("Rotating backward");
    digitalWrite(MOTOR_FWD_PIN, LOW); // Turn off MOTOR_FWD_PIN
    digitalWrite(MOTOR_BCK_PIN, HIGH); // MOTOR_BCK_PIN always at full speed
  }

  // Update motor state
  motorState = direction;
  }


void stopMotor() {
  // Stop motor
  Serial.println("Motor Stopped");
  ledcWrite(PWM_CHANNEL, 0); // Stop MOTOR_FWD_PIN
  digitalWrite(MOTOR_BCK_PIN, LOW);

  // Update motor state
  motorState = STOPPED;
}

float normalize(uint16_t value, uint16_t min, uint16_t max) {
  return (float)(value - min) / (max - min);
}

void calibrate() {
  // Initialize min and max values to extreme opposites
  uint16_t r_min = 65535, r_max = 0;
  uint16_t g_min = 65535, g_max = 0;
  uint16_t b_min = 65535, b_max = 0;

  startMotor(FORWARD);
  digitalWrite(RGB_LED, HIGH);
  

  while (digitalRead(SWITCH_CLOSE_PIN) == HIGH) {
    uint16_t r, g, b, c;
    tcs.getRawData(&r, &g, &b, &c);
    r_min = min(r_min, r);
    r_max = max(r_max, r);
    g_min = min(g_min, g);
    g_max = max(g_max, g);
    b_min = min(b_min, b);
    b_max = max(b_max, b);
  }

  Serial.println("Calibrated...");
  Serial.print("Red min : ");Serial.print(r_min);Serial.print(", ");Serial.print("Red max : ");Serial.println(r_max);
  Serial.print("Green min : ");Serial.print(g_min);Serial.print(", ");Serial.print("Green max : ");Serial.println(g_max);
  Serial.print("Blue min : ");Serial.print(b_min);Serial.print(", ");Serial.print("Blue max : ");Serial.println(b_max);
  digitalWrite(RGB_LED, LOW);
  // Stop the motor
  stopMotor();

  // Store min and max values to non-volatile memory
  preferences.begin("calibration", false);
  preferences.putUInt("r_min", r_min);
  preferences.putUInt("r_max", r_max);
  preferences.putUInt("g_min", g_min);
  preferences.putUInt("g_max", g_max);
  preferences.putUInt("b_min", b_min);
  preferences.putUInt("b_max", b_max);
  preferences.end();
}
