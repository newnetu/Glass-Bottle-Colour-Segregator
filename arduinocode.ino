#include <Servo.h>
int clearLed = 11;
int brownLed = 12; // pin number of the brown LED
int greenLed = 10; // pin number of the green LED
int buttonPin = 2; // pin number of the button
Servo myservo;

void setup() {
  pinMode(brownLed, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  myservo.attach(3);
  Serial.begin(9600);
}

void loop() {
 
    int signal = Serial.read();
    if (signal == '1') {
      // Brown bottle detected, flash the brown LED
      myservo.write(90);
      digitalWrite(brownLed, HIGH);
      delay(1000);
      digitalWrite(brownLed, LOW);
      myservo.write(0);
    } else if (signal == '2') {
      // Green bottle detected, flash the green LED
      digitalWrite(greenLed, HIGH);
      myservo.write(135);
      delay(2000);
      myservo.write(0);
      digitalWrite(greenLed, LOW);
    } else if(signal == '3'){
      // Clear bottle detected, activate the actuator
      myservo.write(45);
      digitalWrite(clearLed, HIGH);
      delay(1000);
      digitalWrite(clearLed, LOW);
      myservo.write(0);
    }
}
