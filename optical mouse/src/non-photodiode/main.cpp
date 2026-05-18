/*
 -------------------------------------------------------------------------------------
  BioMoR — Cricket VR Closed-Loop Firmware (Arduino Mega 2560)

  Extreme-minimal slave mode:
    - Board powers on → immediately begins unconditional 200Hz streaming.
    - loop() top: single-byte listener.  No multi-byte commands.
        'L' → pin 24 (left  solenoid) HIGH for 50 ms
        'R' → pin 26 (right solenoid) HIGH for 50 ms
    - Output (200Hz, '\n'-terminated, 5 fields):
        millisNow,xSum,ySum,zSum,stimOn

  Hardware:
    2× ADNS2083 optical sensors (SPI bit-bang)
    Pin 24 = left  pneumatic valve
    Pin 26 = right pneumatic valve
    Pin 13 = TTL sync marker
 -------------------------------------------------------------------------------------
*/
#include <Arduino.h>
#include "ADNS2083.h"

// ==========================================================================
// Pin Map
// ==========================================================================
const int PIN_TTL         = 13;
const int PIN_VALVE_LEFT  = 24;
const int PIN_VALVE_RIGHT = 26;

// ==========================================================================
// SPI — Optical Sensors
// ==========================================================================
#define SCLK_X 8
#define SDIO_X 9
#define SCLK_Y 6
#define SDIO_Y 7

ADNS2083 OpticalX = ADNS2083(SCLK_X, SDIO_X);
ADNS2083 OpticalY = ADNS2083(SCLK_Y, SDIO_Y);

// ==========================================================================
// State
// ==========================================================================
long xSum = 0;
long ySum = 0;
long zSum = 0;

// 200 Hz cadence (5 ms epoch)
unsigned long millisPre = 0;
const unsigned long MILLIS_FRM = 5;

// Solenoid pulse
bool         stimOn  = false;
unsigned long stimEnd = 0;
const unsigned long STIM_DURATION_MS = 50;

// ==========================================================================
// setup()
// ==========================================================================
void setup() {
    Serial.begin(115200);
    OpticalX.begin();
    OpticalY.begin();

    pinMode(PIN_TTL, OUTPUT);
    digitalWrite(PIN_TTL, LOW);

    pinMode(PIN_VALVE_LEFT, OUTPUT);
    digitalWrite(PIN_VALVE_LEFT, LOW);

    pinMode(PIN_VALVE_RIGHT, OUTPUT);
    digitalWrite(PIN_VALVE_RIGHT, LOW);

    // Safety: ensure unused valve pins are LOW
    pinMode(28, OUTPUT);  digitalWrite(28, LOW);
    pinMode(30, OUTPUT);  digitalWrite(30, LOW);

    delay(1000);
}

// ==========================================================================
// loop()
// ==========================================================================
void loop() {
    unsigned long millisNow = millis();

    // ==================================================================
    // PRIORITY — Single-byte command listener (top of loop, zero parse)
    // ==================================================================
    while (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd == 'L') {
            // Left valve: pin 24, 50 ms pulse
            stimOn  = true;
            stimEnd = millisNow + STIM_DURATION_MS;
            digitalWrite(PIN_VALVE_LEFT, HIGH);
            digitalWrite(PIN_TTL, HIGH);
        }
        else if (cmd == 'R') {
            // Right valve: pin 26, 50 ms pulse
            stimOn  = true;
            stimEnd = millisNow + STIM_DURATION_MS;
            digitalWrite(PIN_VALVE_RIGHT, HIGH);
            digitalWrite(PIN_TTL, HIGH);
        }
        // All other bytes silently ignored — no 'C', 'S', etc.
    }

    // ==================================================================
    // Solenoid OFF after 50 ms pulse
    // ==================================================================
    if (stimOn && millisNow >= stimEnd) {
        digitalWrite(PIN_VALVE_LEFT, LOW);
        digitalWrite(PIN_VALVE_RIGHT, LOW);
        digitalWrite(PIN_TTL, LOW);
        stimOn = false;
    }

    // ==================================================================
    // Unconditional 200 Hz streaming — no enable/disable flag
    // ==================================================================
    if (millisNow - millisPre < MILLIS_FRM) {
        // Within epoch: accumulate optical flow
        if (OpticalX.motion()) {
            xSum += OpticalX.dx();
            zSum += OpticalX.dy();   // 2nd axis of sensor X → Z-DOF
        }
        if (OpticalY.motion()) {
            ySum += OpticalY.dx();
        }
    }
    else {
        // Epoch boundary → emit one data line
        Serial.print(millisNow);
        Serial.print(',');
        Serial.print(xSum);
        Serial.print(',');
        Serial.print(ySum);
        Serial.print(',');
        Serial.print(zSum);
        Serial.print(',');
        Serial.println(stimOn ? 1 : 0);

        xSum = ySum = zSum = 0;
        millisPre += MILLIS_FRM;
    }
}