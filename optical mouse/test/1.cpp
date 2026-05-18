#include <Arduino.h>

/*
-------------------------------------------------------------------------------------
 Get the change in x-coordinate from a optical sensor via SPI communication.
 Send the change in x-coordinate at 200 Hz to Processing via serial communication.
 If the x-coordinate does not change for a certain period, the stimulus is triggered.
-------------------------------------------------------------------------------------
*/
// Include libraries
#include "ADNS2083.h" // Optical sensor

// Pin Settings
const int pinTTL = 13;

//------------------(Added)---------------------------
// Pin number of solenoid valve for airflow stimulation
int numStimCh = 4;               // Total number of valves
int chList[] = {24, 26, 28, 30}; // Pin number of each valve
int chNow = 0;                   // Index of the first valve to open
//----------------------------------------------------

// SPI communication
#define SCLK_X 2 // <- Optical sensor [SCLK]
#define SDIO_X 3 // <- Optical sensor [SDIO]

#define SCLK_Y 31 // <- Optical sensor [SCLK]
#define SDIO_Y 33 // <- Optical sensor [SDIO]

// Create PAN2083 device
ADNS2083 OpticalX = ADNS2083(SCLK_X, SDIO_X);
ADNS2083 OpticalY = ADNS2083(SCLK_Y, SDIO_Y);

// Mode
bool modeRecord = true; // Mode to output serial communication
bool modeStim = false;  // Stimulation mode based on animal movement
bool modeStimTemp = false;
bool stimReady = true; // Whether or not the stimulus can be triggered
bool stimOn = false;   // State of the trigger signal
bool stimOff = false;  // State of the trigger signal

// Serial communication
int indStr = 0; // Number of input characters in serial communication

// Optical sensor
signed char xSum = 0; // Total change in X-coordinate per unit time
signed char ySum = 0; // Total change in Y-coordinate per unit time
signed char zSum = 0; // Total change in Z-coordinate per unit time

float correctionX = 1.595;
float correctionY = 1.233;
float inch2mm = 25.4 / 800; // Conversion factor [25,4inch/mm/dpi]

// Loop control
unsigned long millisPre = 0;   // Time of the previous serial communication
unsigned long millisFrm = 5;   // Interval of serial communication output
unsigned long millisStim = 0;  // Time of the previous stim_onset
unsigned long millisReady = 0; // Time of the previous stim_mode_onset

// Judgement for Walking
int threshold = 10;      // Threshold to determine walking [mm/s]
int count = 0;           // Number of consecutive frames above threshold
int count2 = 0;          // Number of consecutive frames above threshold
int framePre = 200;      // (Period to determine non-walking condition / millis_frm) [frame]
int stimDur = 200;       // Stimulus duration
int stimInterval = 5000; // Interval of stim_mode

// Mean filter
const int listLen = 7; // Frame length of filter
int listNum = 0;
float listTmp[listLen]; // list for store data
float velTmp = 0;
float velMean = 0;

int split(String data, char delimiter, String *dst)
{
  int index = 0;
  int arraySize = data.length();
  for (int i = 0; i < arraySize; i++)
  {
    char tmp = data.charAt(i);
    if (tmp == delimiter)
    {
      index++;
    }
    else
    {
      dst[index] = dst[index] + tmp;
    }
  }
  return (index + 1);
}

float calcDist(float x, float y, float corrX, float corrY, float inch2mm)
{
  float xTmp = x * corrX * inch2mm;
  float yTmp = y * corrY * inch2mm;
  return sqrt(xTmp * xTmp + yTmp * yTmp);
}

void startStim()
{
  stimOn = true; // Stim ON
  digitalWrite(chList[chNow], HIGH);
  digitalWrite(pinTTL, HIGH);
  unsigned long millisNow = millis(); // Time of loop start
  millisStim = millisNow + stimDur;
}

void setup()
{
  // Initialization

  Serial.begin(115200);
  OpticalX.begin();
  OpticalY.begin();

  pinMode(pinTTL, OUTPUT);
  digitalWrite(pinTTL, LOW);
  //------------------(Added)---------------------------
  // Initial setting of output pins for valves
  for (int i = 0; i < numStimCh; i++)
  {                               // Repeat the block for the total number of valves
    pinMode(chList[i], OUTPUT);   // Put pins connected to valves into output mode
    digitalWrite(chList[i], LOW); // Set the pins to low state
  }
  //----------------------------------------------------
  delay(1000);
}

void loop()
{
  unsigned long millisNow = millis(); // Time of loop start
  // Read serial input
  if (Serial.available() > 0)
  {
    String str = Serial.readStringUntil(';');
    String strs[2];
    indStr = split(str, ',', strs);
    if (indStr == 2)
    {
      int mode = strs[0].toInt();
      int param = strs[1].toInt();
      // Switch of mode_record
      if (mode == 1)
      {
        chNow = param;
      }
      // Switch of mode_record
      if (mode == 2)
      {
        stimDur = (param == 0) ? 200 : 5000;
      }
      // Switch of mode_record
      if (mode == 3)
      {
        modeRecord = (param == 1);
      }
      // Switch of mode_stim
      if (mode == 4)
      {
        modeStim = (param == 1);
      }
      if (mode == 5)
      {
        modeStimTemp = (param == 1);
      }
    }
  }
  // mode_record
  if (modeRecord)
  {
    // Sum up coordinate changes until serial output
    if (millisNow - millisPre < millisFrm)
    {
      // Get coordination changes from optical sensor
      if (OpticalX.motion())
      {
        xSum += OpticalX.dx();
        zSum += OpticalX.dy();
      }
      if (OpticalY.motion())
      {
        ySum += OpticalY.dx();
      }
    }
    // Serial output to Processing
    else
    {
      Serial.print(millisNow);
      Serial.print(",");
      Serial.print(xSum);
      Serial.print(",");
      Serial.print(ySum);
      Serial.print(",");
      Serial.print(zSum);
      Serial.print(",");
      Serial.print(stimOn);
      Serial.print(",");
      Serial.println(stimOff);
      // If the coordinate change is above the threshold, the count is increased.
      listNum = (listNum + 1) % listLen; // Update the index of the buffer for storing velocity data
      // Calculate velocity
      velTmp = calcDist(xSum, ySum, correctionX, correctionY, inch2mm) * (1000 / millisFrm);
      listTmp[listNum] = velTmp; // Store the calculated velocity
      float result = 0;
      for (int i = 0; i < listLen; i++)
      {
        result += listTmp[i]; // Accumulate velocity values
      }
      velMean = result / listLen; // Compute the mean velocity

      // Check if the mean velocity exceeds the threshold
      // Increment the count if velocity is above threshold
      if (velMean >= threshold)
      {
        count += 1;
      }
      else
      {
        count = 0;
      }
      // Reset the sum of coordinate changes and update the previous millisecond timestamp
      xSum = ySum = zSum = 0;
      millisPre = millisNow;
    }

    // mode_stim// When stimulus standby is OK// When no walking for a period
    if (modeStim && stimReady && count >= framePre)
    {
      startStim();
      stimReady = false;
      millisReady = millisNow + stimInterval;
    }

    // "stim_on" is "false" after "millis_stim" from the stim onset.
    if (stimOn && millisNow >= millisStim)
    {
      digitalWrite(chList[chNow], LOW);
      digitalWrite(pinTTL, LOW);
      stimOn = false;
      stimOff = true;
    }
    if (stimOff && !stimOn && millisNow >= millisStim + millisFrm)
    {
      stimOff = false;
    }
    // "stim_ready" is "true" after "millis_ready" from the stim onset.
    if (!stimReady && millisNow >= millisReady)
    {
      if (velMean < threshold)
      {
        count2 += 1;
      }
      else
      {
        count2 = 0;
      }
      if (count2 >= framePre)
      {
        stimReady = true;
        count = 0;
        count2 = 0;
      }
    }
    if (modeStimTemp)
    {
      startStim();
      if (stimOn && millisNow >= millisStim)
      {
        // digitalWrite(sp_seq[chNow], LOW)
        digitalWrite(pinTTL, LOW);
        stimOn = false;
        stimOff = true;
      }
      if (stimOff && !stimOn && millisNow >= millisStim + millisFrm)
      {
        stimOff = false;
        modeStimTemp = false;
      }
    }
  }
}
