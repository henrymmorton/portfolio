#include "sensors.h"

bool hall_trpd;
float rev_count = 0;
bool new_dat = false;
float t1;
float t2;
float dt;

void Tachometer::init_tachometer() {
  pinMode(hall_pin, INPUT); // initialize the hall pin as a input:
}

void Tachometer::simple_read() {
  Serial.println(digitalRead(hall_pin));
  delay(500);
}

void Tachometer::update_tachometer() {
  if (new_dat) {
    rev_count = 0;
    new_dat = false;
    t1 = micros();
  }
  if (digitalRead(hall_pin) == 0) {
    if (hall_trpd == false) {
      hall_trpd = true;
      rev_count += 1;
    }
  } else {
    hall_trpd = false;
  }
  if (rev_count>=rev_thrsh) {
    dt = micros() - t1;
    rpm = (rev_count / (dt / 1000000)) * 60;
    new_dat = true;
  }
}

void Tachometer::print_RPM() {
  if (new_dat) {
    Serial.print("RPM: ");
    Serial.println(rpm);
  }
}
