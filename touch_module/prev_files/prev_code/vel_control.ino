#include <Arduino.h>
#include <SPI.h>

#if defined(__AVR_ATmega4809__)
  #include <avr/io.h>
#endif

// ─── 핀 정의 ────────────────────────────────────────────────────────────
const int latchPin       = 10;           // ST_CP
const int numShiftRegs   = 10;           // 10×8 = 80행
const int numRows        = numShiftRegs * 8;

const int muxSelectPins[3] = {2, 3, 4};   // S0, S1, S2
const int numMuxBits       = 3;
const int muxAnalogPins[8] = {A0, A1, A2, A3, A4, A5, A6, A7};
const int numMuxDevices    = 8;

const int numCols = 63;  // 실제로 읽을 열 수

void setup() {
  Serial.begin(2000000);

  // SPI 설정
  pinMode(latchPin, OUTPUT);
  SPI.begin();
  SPI.setDataMode(SPI_MODE0);
  SPI.setClockDivider(SPI_CLOCK_DIV4);  // 안정적인 속도로 낮춤
  SPI.setBitOrder(MSBFIRST);

  // MUX select 핀
  for (int i = 0; i < numMuxBits; i++) {
    pinMode(muxSelectPins[i], OUTPUT);
  }

  // ADC 분주비 16
  #if defined(__AVR_ATmega4809__)
    ADC0.CTRLC = ADC_PRESC_DIV16_gc;
  #else
    ADCSRA = (ADCSRA & ~((1<<ADPS2)|(1<<ADPS1)|(1<<ADPS0)))
             | (1<<ADPS2);
  #endif
}

inline void selectRow(int row) {
  digitalWrite(latchPin, LOW);
  for (int r = 0; r < numShiftRegs; r++) {
    uint8_t b = 0;
    int base = (numShiftRegs - r - 1) * 8;
    if (row >= base && row < base + 8) {
      b = 1 << (row - base);
    }
    SPI.transfer(b);
  }
  digitalWrite(latchPin, HIGH);
}

inline void selectMux(int ch) {
  for (int i = 0; i < numMuxBits; i++) {
    digitalWrite(muxSelectPins[i], (ch >> i) & 1);
  }
}

void loop() {
  // ── 프레임 헤더(싱크 바이트) 전송 ──
  Serial.write(0xAA);
  Serial.write(0x55);

  // ── 센서값 전송 ──
  for (int row = 0; row < numRows; row++) {
    selectRow(numRows - row - 1);
    delayMicroseconds(10);    // 안정화 지연
    for (int ch = 0; ch < (1 << numMuxBits); ch++) {
      selectMux(ch);
      delayMicroseconds(10);
      for (int dev = 0; dev < numMuxDevices; dev++) {
        int col = dev * 8 + ch;
        if (col >= numCols) continue;
        uint8_t v = analogRead(muxAnalogPins[dev]) >> 2;  // 10비트를 8비트로
        Serial.write(v);
      }
    }
  }
}
