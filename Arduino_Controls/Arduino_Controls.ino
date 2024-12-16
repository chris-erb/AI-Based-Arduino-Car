#include <Arduino.h>

#define PIN_Motor_PWMA 5
#define PIN_Motor_PWMB 6
#define PIN_Motor_BIN_1 8
#define PIN_Motor_AIN_1 7
#define PIN_Motor_STBY 3

bool isAutonomousMode = false; // Flag to track autonomous mode status

void setup() {
    Serial.begin(115200); // Start serial communication
    pinMode(PIN_Motor_PWMA, OUTPUT);
    pinMode(PIN_Motor_PWMB, OUTPUT);
    pinMode(PIN_Motor_AIN_1, OUTPUT);
    pinMode(PIN_Motor_BIN_1, OUTPUT);
    pinMode(PIN_Motor_STBY, OUTPUT);
    digitalWrite(PIN_Motor_STBY, LOW); // Initially keep motors off
}

void loop() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n'); // Read until newline
        command.trim(); // Trim whitespace
        
        if (command == "FORWARD") {
            moveForward();
        } else if (command == "BACKWARD") {
            moveBackward();
        } else if (command == "LEFT") {
            turnLeft();
        } else if (command == "RIGHT") {
            turnRight();
        } else if (command == "STOP") {
            stopMotors();
            isAutonomousMode = false;
        } 
    }
}

void moveForward() {
    digitalWrite(PIN_Motor_STBY, HIGH);
    digitalWrite(PIN_Motor_AIN_1, HIGH);
    analogWrite(PIN_Motor_PWMA, 65); // 255 is full speed
    digitalWrite(PIN_Motor_BIN_1, HIGH);
    analogWrite(PIN_Motor_PWMB, 65);
}

void moveBackward() {
    digitalWrite(PIN_Motor_STBY, HIGH);
    digitalWrite(PIN_Motor_AIN_1, LOW);
    analogWrite(PIN_Motor_PWMA, 65);
    digitalWrite(PIN_Motor_BIN_1, LOW);
    analogWrite(PIN_Motor_PWMB, 65); 
}

void turnLeft() {
    digitalWrite(PIN_Motor_STBY, HIGH);
    digitalWrite(PIN_Motor_AIN_1, LOW);
    analogWrite(PIN_Motor_PWMA, 0);
    digitalWrite(PIN_Motor_BIN_1, HIGH);
    analogWrite(PIN_Motor_PWMB, 65); 
}

void turnRight() {
    digitalWrite(PIN_Motor_STBY, HIGH);
    digitalWrite(PIN_Motor_AIN_1, HIGH);
    analogWrite(PIN_Motor_PWMA, 65); 
    digitalWrite(PIN_Motor_BIN_1, LOW);
    analogWrite(PIN_Motor_PWMB, 0);
}

void stopMotors() {
    digitalWrite(PIN_Motor_STBY, LOW); // Disable motors
}