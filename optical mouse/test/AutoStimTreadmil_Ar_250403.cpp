//-------------------
// Read velocity data from TrackTaro and trigger airflow stimulation
// Trigger stimulation if the cricket does not move for a certain period
//-------------------

// Include necessary libraries
#include "QuickStats.h"  // calculating median
#include <Wire.h>        // I2C communication
#include <Adafruit_MCP4725.h>  // DAC
Adafruit_MCP4725 dac;  // Instance of the DAC

//----------------------------
// Pin settings
const int totalValves = 8;  // Total number of solenoid valves
const int valvePins[] = {24, 26, 28, 30, 32, 34, 36, 38};  // Pin numbers for the valves
int currentValveIndex = 4;  // Index for the current valve to be triggered

const int pinTriggerInput = 42; // Pin for external trigger input
const int pinStimMonitor = 46;  // Pin for stimulation timing output
const int pinTrackTaroAnalogInput = A0;  // Analog input pin for TrackTaro data

//----------------------------
// Constants
const int pulseInterval     = 5;    // Interval for output pulse (ms)

const int movementThreshold = 5;    // Threshold for movement detection
const int preFrameCount     = 200;  // Frame count for determining no movement

const int velocityBufferSize = 9;  // Buffer size for median filter
const int stimInterval = 10000;  // Delay after stimulation (ms)
const int loopDelay = 5;  // Loop delay time in milliseconds

//----------------------------
// Variables
int currentVelocity;  // Velocity data read from TrackTaro
int serialInputIndex = 0;  // Index for parsed serial input string
bool modeAuto = false;  // Flag for automatic mode (false: manual, true: automatic)
int stimulationDuration = 200;  // Duration of stimulation (ms)
int noMovementCounter = 0;  // Counter for no movement condition
int previousTTLState = 0;  // Previous state of the TTL input
unsigned long lastStimulusMillis = 0;  // Timestamp for non-blocking delay
unsigned long lastLoopMillis = 0;  // Timestamp for loop interval control
unsigned long lastVelocityCheckMillis = 0;

//----------------------------
// Function to split a string by a delimiter
int split(String data, char delimiter, String *result) {
    int index = 0;
    int dataLength = data.length();
    for (int i = 0; i < dataLength; i++) {
        if (data.charAt(i) == delimiter) {
            index++;
        } else {
            result[index] += data.charAt(i);
        }
    }
    return (index + 1);  // Return the number of elements after splitting
}

//----------------------------
// Setup function for initialization
void setup() {
    Serial.begin(9600);  // Start serial communication
    delay(1000);  // Wait for stable operation of Arduino

    // Initialize pins for valves
    for (int i = 0; i < totalValves; i++) {
        pinMode(valvePins[i], OUTPUT);
        digitalWrite(valvePins[i], LOW);  // Set initial state to LOW
    }

    // Set trigger input and monitor pins
    pinMode(pinTriggerInput, INPUT);
    pinMode(pinStimMonitor, OUTPUT);
    digitalWrite(pinStimMonitor, LOW);  // Set initial state to LOW

    // Initialize MCP4725 DAC
    dac.begin(0x61);
    dac.setVoltage(0, false);
}

//----------------------------
// Main loop
void loop() {
    unsigned long currentMillis = millis();  // Get current time

    // Process serial input if available
    if (Serial.available() > 0) {
      handleSerialInput();
    }

    // Process manual or automatic mode
    if (modeAuto) {
        if (currentMillis - lastVelocityCheckMillis >= loopDelay) {
            lastVelocityCheckMillis = currentMillis;
            processVelocityInput();
        }
    } else {
        checkTTLInput();
    }

    // Non-blocking delay example for periodic tasks
    if (currentMillis - lastLoopMillis >= loopDelay) {
        lastLoopMillis = currentMillis;
        // Additional periodic code can be added here if necessary
    }
}

//----------------------------
// Function to handle serial input and command processing
void handleSerialInput() {
    String inputString = Serial.readStringUntil(';');
    String parsedStrings[2];
    serialInputIndex = split(inputString, ',', parsedStrings);

    if (serialInputIndex == 2) {
        int command = parsedStrings[0].toInt();  // Command type
        int parameter = parsedStrings[1].toInt();  // Parameter value

        switch (command) {
            case 0:  // Set the current valve to be triggered
                currentValveIndex = parameter;
                break;
            case 1:  // Trigger the stimulation
                triggerStimulus();
                break;
            case 2:  // Set voltage for the DAC
                dac.setVoltage(parameter, false);
                break;
            case 3:  // Enable automatic mode
                modeAuto = true;
                break;
            case 4:  // Disable automatic mode
                modeAuto = false;
                noMovementCounter = 0;  // Reset the counter
                break;
            case 5:  // Set stimulation duration
                setStimDuration(parameter);
                break;
            default:  // Handle unknown commands
                Serial.println("Invalid command");
                break;
        }
    } else {
        Serial.println("Invalid input format");
    }
}

//----------------------------
// Function to trigger stimulation
void triggerStimulus() {
    digitalWrite(valvePins[currentValveIndex], HIGH);
    digitalWrite(pinStimMonitor, HIGH);
    delay(stimulationDuration);  // Hold stimulation for the set duration
    digitalWrite(valvePins[currentValveIndex], LOW);
    digitalWrite(pinStimMonitor, LOW);
    delay(pulseInterval);
    outputPulse();  // Send output pulse
    Serial.println("1");  // Notify that stimulation is complete
    lastStimulusMillis = millis();  // Update timestamp for non-blocking delay
}

//----------------------------
// Function to send output pulse
void outputPulse() {
    for (int i = 0; i < currentValveIndex + 1; i++) {
        digitalWrite(pinStimMonitor, HIGH);
        delay(pulseInterval);
        digitalWrite(pinStimMonitor, LOW);
        delay(pulseInterval);
    }
}

//----------------------------
// Function to set the stimulation duration
void setStimDuration(int value) {
    if (value == 0) {
        stimulationDuration = 100;  // Short stimulation
    } else if (value == 1) {
        stimulationDuration = 200;  // Standard stimulation
    } else if (value == 2) {
        stimulationDuration = 10000;  // Long stimulation
    } else {
        Serial.println("Invalid stimulation duration value");
    }
}

//----------------------------
// Function to check TTL input (for manual mode)
void checkTTLInput() {
    if (digitalRead(pinTriggerInput) != previousTTLState) {
        if (previousTTLState == 0) {  // Rising edge detection
            triggerStimulus();
            previousTTLState = 1;
        } else {
            previousTTLState = 0;  // Falling edge detection
        }
    }
}

//----------------------------
// Function to process velocity input (for automatic mode)
void processVelocityInput() {
    unsigned long currentMillis = millis();
    
    if (currentMillis - lastStimulusMillis < stimInterval) {
        return;
    }
    currentVelocity = analogRead(pinTrackTaroAnalogInput);  // Read analog input from TrackTaro

    // Increment counter if velocity is below the threshold
    if (currentVelocity <= movementThreshold) {
        noMovementCounter++;
    } else {
        noMovementCounter = 0;  // Reset counter if velocity exceeds the threshold
    }

    // Trigger stimulation if no movement is detected for a certain period
    if (noMovementCounter >= preFrameCount) {
        triggerStimulus();
        noMovementCounter = 0;  // Reset the counter
        lastStimulusMillis = currentMillis;  // Update timestamp for non-blocking delay
    }
}
