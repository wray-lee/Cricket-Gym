#include <Arduino.h>
#include "ADNS2083.h"

// --- 引脚约束定义 ---
const int SCLK_X = 2;
const int SDIO_X = 3;

const int SCLK_Y = 11;
const int SDIO_Y = 12; // 避开板载指示灯引脚

const int LED_PIN = 13;

ADNS2083 SensoryX = ADNS2083(SCLK_X, SDIO_X);
ADNS2083 SensoryY = ADNS2083(SCLK_Y, SDIO_Y);

// --- UI 渲染器 ---
// 统一分配15个字符宽度，防止终端字符跳动错位
void renderChannelUI(const char *label, bool isConnected, int amplitude)
{
  Serial.print(label);
  if (!isConnected)
  {
    // 物理层断开连接
    Serial.print("OFFLINE        ");
  }
  else if (amplitude > 0)
  {
    // 正常对焦且有运动
    int bars = (amplitude > 15) ? 15 : amplitude;
    for (int i = 0; i < bars; i++)
      Serial.print("█");
    for (int i = bars; i < 15; i++)
      Serial.print(" ");
  }
  else
  {
    // 对焦丢失或绝对静止
    Serial.print("LOST           ");
  }
}

void setup()
{
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);

  // 初始化总线电平

  pinMode(SCLK_X, OUTPUT);
  digitalWrite(SCLK_X, HIGH);

  pinMode(SCLK_Y, OUTPUT);
  digitalWrite(SCLK_Y, HIGH);

  delay(100);

  // 硬件级启动
  SensoryX.begin();
  SensoryY.begin();
}

// --- 物理层引脚连接探测 ---
// 原理：用原始 bit-bang 读取 Product_ID 寄存器 (地址 0x00)。
// ADNS2083 会返回固定的产品 ID（非 0xFF、非 0x00）；
// 悬空引脚在内部上拉作用下读回 0xFF。
bool isPinConnected(uint8_t sclkPin, uint8_t sdioPin)
{
  uint8_t address = 0x00; // Product_ID 寄存器
  uint8_t result = 0;

  // 1. 发送寄存器地址（8 bit，MSB first）
  pinMode(sdioPin, OUTPUT);
  for (int i = 7; i >= 0; i--)
  {
    digitalWrite(sclkPin, LOW);
    digitalWrite(sdioPin, address & (1 << i));
    digitalWrite(sclkPin, HIGH);
  }

  // 2. 切换 SDIO 为带上拉的输入，等待 tHOLD
  pinMode(sdioPin, INPUT_PULLUP);
  delayMicroseconds(100);

  // 3. 读取返回数据（8 bit，MSB first）
  for (int i = 7; i >= 0; i--)
  {
    digitalWrite(sclkPin, LOW);
    digitalWrite(sclkPin, HIGH);
    result |= (digitalRead(sdioPin) << i);
  }

  // 4. 恢复 SDIO 为普通输入
  pinMode(sdioPin, INPUT);
  delayMicroseconds(100);

  // 5. 判定：Product_ID 应为有效值；悬空引脚读回 0xFF 或 0x00
  return (result != 0xFF && result != 0x00);
}

void loop()
{
  // --- 物理层断联检测（纯 GPIO 级别，不依赖传感器寄存器数据）---
  bool connectedX = isPinConnected(SCLK_X, SDIO_X);
  bool connectedY = isPinConnected(SCLK_Y, SDIO_Y);

  int ampX = 0;
  int ampY = 0;

  // --- 只有在线时才读取传感器数据 ---
  if (connectedX)
  {
    SensoryX.motion(); // 清除 motion 标志
    delayMicroseconds(120);
    int8_t dx1 = SensoryX.dx();
    delayMicroseconds(120);
    int8_t dy1 = SensoryX.dy();
    delayMicroseconds(120);
    ampX = abs(dx1) + abs(dy1);
  }

  if (connectedY)
  {
    SensoryY.motion();
    delayMicroseconds(120);
    int8_t dx2 = SensoryY.dx();
    delayMicroseconds(120);
    int8_t dy2 = SensoryY.dy();
    delayMicroseconds(120);
    ampY = abs(dx2) + abs(dy2);
  }

  // --- 物理状态反馈 ---
  // 约束条件：只有当两颗传感器都在线、且均切入有效焦平面时，板载灯才点亮
  if (connectedX && connectedY && ampX > 0 && ampY > 0)
  {
    digitalWrite(LED_PIN, HIGH);
  }
  else
  {
    digitalWrite(LED_PIN, LOW);
  }

  // --- 终端双通道并行渲染 ---
  renderChannelUI("Sensory-X: ", connectedX, ampX);
  Serial.print(" | ");
  renderChannelUI("Sensory-Y: ", connectedY, ampY);
  Serial.println();

  delay(50);
}