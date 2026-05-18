/*
 -------------------------------------------------------------------------------------
  BioMoR — Cricket VR Closed-Loop Firmware (Arduino Mega 2560)

  Photodiode T₀ Architecture
  ===========================
  This firmware is the hardware counterpart of stimulus_controller.py.
  It implements a non-blocking state machine that:

    1. Parses packet commands from the host:  <DIR,DELAY_MS>
       e.g. <L,5729>  → arm valve 0, fire 5729 ms after T₀
            <R,300>   → arm valve 1, fire 300 ms after T₀
            <3,1000>  → arm valve 3, fire 1000 ms after T₀
            <L,0>     → arm valve 0, fire immediately on T₀

    2. Waits for a RISING-edge hardware interrupt on Pin 18 (photodiode).
       The ISR records T₀ = millis() and sets a flag.

    3. After T₀, counts down DELAY_MS using millis().  When elapsed,
       fires the target valve + TTL sync for pulseInterval ms, then resets to IDLE.

    4. Streams 200 Hz motion data unconditionally:
         t_ard,dx,dy,dz,stim_state\n

  Constraints
  -----------
  - ZERO delay() calls (hard real-time, non-blocking throughout).
  - ISR is minimal: one timestamp write + one flag set, both volatile.
  - Serial receive buffer is bounded at 16 bytes to prevent overflow.
  - Parse failures are silently discarded — never disrupts 200 Hz stream.

  Hardware Pin Map
  ----------------
    Pin 13  = TTL Sync marker (output)
    Pin 28  = Valve 0 (L / index 0)  [was pin 24, now shifted]
    Pin 30  = Valve 1 (R / index 1)  [was pin 26, now shifted]
    Pin 32  = Valve 2
    Pin 34  = Valve 3
    Pin 36  = Valve 4
    Pin 38  = Valve 5
    Pin 40  = Valve 6
    Pin 42  = Valve 7
    Pin 18  = Photodiode interrupt input (RISING edge)
    Pin  2  = SCLK Sensor X
    Pin  3  = SDIO Sensor X
    Pin 31  = SCLK Sensor Y
    Pin 33  = SDIO Sensor Y
 -------------------------------------------------------------------------------------
*/
#include <Arduino.h>
#include "ADNS2083.h"

// ==========================================================================
// Pin Map
// ==========================================================================
const int PIN_TTL = 13;
const int PIN_PHOTODIODE_INT = 18; // INT5 on Mega 2560

// Multi-channel valve array (pins 24 & 26 occupied by sensors → start at 28)
const int totalValves = 8;
const int valvePins[totalValves] = {28, 30, 32, 34, 36, 38, 40, 42};

// ==========================================================================
// SPI — Optical Sensors (unchanged)
// ==========================================================================
#define SCLK_X 2
#define SDIO_X 3
#define SCLK_Y 31
#define SDIO_Y 33

ADNS2083 OpticalX = ADNS2083(SCLK_X, SDIO_X);
ADNS2083 OpticalY = ADNS2083(SCLK_Y, SDIO_Y);

// ==========================================================================
// 200 Hz Streaming State
// ==========================================================================
long xSum = 0;
long ySum = 0;
long zSum = 0;

unsigned long millisPre = 0;
const unsigned long MILLIS_FRM = 5; // 200 Hz = 5 ms epoch

// ==========================================================================
// Photodiode ISR — T₀ Anchor (volatile, minimal)
// ==========================================================================
volatile bool t0_triggered = false;
volatile unsigned long t0_millis = 0;

void photodiodeISR()
{
    t0_millis = millis();
    t0_triggered = true;
}

// ==========================================================================
// Valve State Machine
// ==========================================================================
enum ValveState : uint8_t
{
    STATE_IDLE,      // No pending command, waiting for <DIR,DELAY>
    STATE_ARMED,     // Command received, waiting for photodiode T₀
    STATE_COUNTDOWN, // T₀ received, counting down DELAY_MS
    STATE_FIRING,    // Valve + TTL active, waiting pulseInterval ms
};

ValveState valveState = STATE_IDLE;
int targetValvePin = -1;         // pin number of the armed valve
uint16_t targetDelay = 0;        // 0–10000 ms delay after T₀
unsigned long valveOpenTime = 0; // millis() when valve was energised

const unsigned long PULSE_INTERVAL_MS = 5; // Valve open pulse duration (ms)

// ==========================================================================
// Non-blocking Serial Packet Parser
// --------------------------------------------------------------------------
// Format: <DIR,DELAY_MS>   e.g. <L,5729>, <R,0>, or <3,1000>
// DIR can be 'L'/'R' (→ index 0/1) or a digit 0–7 (→ direct valve index).
// Buffer hard-capped at 16 bytes.  Overflow or malformed → silent discard.
// ==========================================================================
const uint8_t CMD_BUF_SIZE = 16;
char cmdBuf[CMD_BUF_SIZE];
uint8_t cmdIdx = 0;
bool cmdInside = false; // true after '<' seen, false after '>' or overflow

void parseSerialPackets()
{
    while (Serial.available() > 0)
    {
        char c = Serial.read();

        if (c == '<')
        {
            // Start of new packet — reset buffer
            cmdIdx = 0;
            cmdInside = true;
            continue;
        }

        if (!cmdInside)
            continue; // Outside packet — discard byte

        if (c == '>')
        {
            // End of packet — attempt parse
            cmdBuf[cmdIdx] = '\0';
            cmdInside = false;

            // Expected: "L,5729"  or  "R,0"  or  "3,1000"
            if (cmdIdx < 3)
                continue; // too short
            char dir = cmdBuf[0];

            // --- Parse direction → valve index ---
            int valveIdx = -1;
            if (dir == 'L' || dir == 'l')
            {
                valveIdx = 0;
            }
            else if (dir == 'R' || dir == 'r')
            {
                valveIdx = 1;
            }
            else if (dir >= '0' && dir <= '7')
            {
                valveIdx = dir - '0';
            }
            else
            {
                continue; // invalid direction
            }

            if (cmdBuf[1] != ',')
                continue; // missing comma

            // --- Parse delay (digits only, 0–10000) ---
            unsigned long val = 0;
            bool validNum = true;
            for (uint8_t i = 2; i < cmdIdx; i++)
            {
                if (cmdBuf[i] < '0' || cmdBuf[i] > '9')
                {
                    validNum = false;
                    break;
                }
                val = val * 10 + (cmdBuf[i] - '0');
                if (val > 10000)
                {
                    validNum = false;
                    break;
                }
            }
            if (!validNum)
                continue;

            // ---- Valid command: arm the state machine ----
            targetValvePin = valvePins[valveIdx];
            targetDelay = (uint16_t)val;
            valveState = STATE_ARMED;

            // Clear any stale T₀ flag so we only respond to the NEXT flash
            noInterrupts();
            t0_triggered = false;
            interrupts();

            continue;
        }

        // Accumulate byte into buffer (guard overflow)
        if (cmdIdx < CMD_BUF_SIZE - 1)
        {
            cmdBuf[cmdIdx++] = c;
        }
        else
        {
            // Overflow — discard this packet silently
            cmdInside = false;
            cmdIdx = 0;
        }
    }
}

// ==========================================================================
// Valve State Machine — tick (called every loop iteration)
// ==========================================================================
void valveStateMachineTick(unsigned long now)
{
    switch (valveState)
    {

    case STATE_IDLE:
        // Nothing to do — waiting for host command
        break;

    case STATE_ARMED:
    {
        // Check if photodiode has fired
        bool fired;
        noInterrupts();
        fired = t0_triggered;
        interrupts();

        if (!fired)
            break; // Still waiting for flash

        // T₀ received — clear flag, record timestamp
        noInterrupts();
        t0_triggered = false;
        interrupts();

        if (targetDelay == 0)
        {
            // Zero delay — fire immediately at T₀
            valveState = STATE_FIRING;
            valveOpenTime = now;
            digitalWrite(targetValvePin, HIGH);
            digitalWrite(PIN_TTL, HIGH);
        }
        else
        {
            // Enter countdown from T₀
            valveState = STATE_COUNTDOWN;
        }
        break;
    }

    case STATE_COUNTDOWN:
    {
        unsigned long t0;
        noInterrupts();
        t0 = t0_millis;
        interrupts();

        if (now - t0 >= targetDelay)
        {
            // Delay elapsed — fire valve
            valveState = STATE_FIRING;
            valveOpenTime = now;
            digitalWrite(targetValvePin, HIGH);
            digitalWrite(PIN_TTL, HIGH);
        }
        break;
    }

    case STATE_FIRING:
        if (now - valveOpenTime >= PULSE_INTERVAL_MS)
        {
            // Pulse complete — turn valve + TTL off, return to IDLE
            digitalWrite(targetValvePin, LOW);
            digitalWrite(PIN_TTL, LOW);

            valveState = STATE_IDLE;
        }
        break;
    }
}

// ==========================================================================
// Helper: is any valve currently energised?
// ==========================================================================
inline bool isStimActive()
{
    return (valveState == STATE_FIRING);
}

// ==========================================================================
// setup()
// ==========================================================================
void setup()
{
    Serial.begin(115200);
    OpticalX.begin();
    OpticalY.begin();

    // TTL sync output
    pinMode(PIN_TTL, OUTPUT);
    digitalWrite(PIN_TTL, LOW);

    // Multi-channel valve outputs — init all LOW
    for (int i = 0; i < totalValves; i++)
    {
        pinMode(valvePins[i], OUTPUT);
        digitalWrite(valvePins[i], LOW);
    }

    // Photodiode interrupt — RISING edge on Pin 18 (INT5)
    pinMode(PIN_PHOTODIODE_INT, INPUT);
    attachInterrupt(digitalPinToInterrupt(PIN_PHOTODIODE_INT),
                    photodiodeISR, RISING);

    // Sensors already settled via begin() delay(1000) × 2.
    // Start the 200 Hz epoch timer from now.
    millisPre = millis();
}

// ==========================================================================
// loop()
// ==========================================================================
void loop()
{
    unsigned long millisNow = millis();

    // ==================================================================
    // 1. Non-blocking serial packet parser  (<DIR,DELAY_MS>)
    // ==================================================================
    parseSerialPackets();

    // ==================================================================
    // 2. Valve state machine tick (IDLE / ARMED / COUNTDOWN / FIRING)
    // ==================================================================
    valveStateMachineTick(millisNow);

    // ==================================================================
    // 3. Unconditional 200 Hz streaming — never interrupted
    // ==================================================================
    if (millisNow - millisPre < MILLIS_FRM)
    {
        // Within epoch: accumulate optical flow
        if (OpticalX.motion())
        {
            xSum += OpticalX.dx();
            zSum += OpticalX.dy(); // 2nd axis of sensor X → Z-DOF
        }
        if (OpticalY.motion())
        {
            ySum += OpticalY.dx();
        }
    }
    else
    {
        // Epoch boundary → emit one data line
        Serial.print(millisNow);
        Serial.print(',');
        Serial.print(xSum);
        Serial.print(',');
        Serial.print(ySum);
        Serial.print(',');
        Serial.print(zSum);
        Serial.print(',');
        Serial.println(isStimActive() ? 1 : 0);

        xSum = ySum = zSum = 0;
        millisPre = millisNow;
    }
}