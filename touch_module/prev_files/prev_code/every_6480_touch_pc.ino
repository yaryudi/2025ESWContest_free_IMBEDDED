#include <Arduino.h>
#if defined(__AVR_ATmega4809__)
  #include <avr/io.h>
#endif

// ─── 핀 정의 ─────────────────────────────────────────────
const int shiftDataPin   = 11;
const int shiftClockPin  = 13;
const int shiftLatchPin  = 10;

const int numShiftRegs   = 10;   // 10×8=80행
const int numRows        = numShiftRegs * 8;

// MUX 선택 핀 (S0, S1, S2)
const int muxSelectPins[3] = {2, 3, 4};
const int numMuxBits       = 3;

// MUX 공통 출력(COM) – A6부터 연결 순서로 재정렬
//    dev=0 → A6
//    dev=1 → A7
//    dev=2 → A0
//    dev=3 → A1
//    dev=4 → A2
//    dev=5 → A3
//    dev=6 → A4
//    dev=7 → A5
const int muxAnalogPins[8] = {
  A2, A3, A4, A5, A6, A7, A0, A1
};
const int numMuxDevices    = 8;

// 실제로 읽을 열 개수
const int numCols          = 63;

void setup() {
  Serial.begin(115200);
  pinMode(shiftDataPin,  OUTPUT);
  pinMode(shiftClockPin, OUTPUT);
  pinMode(shiftLatchPin, OUTPUT);
  for (int i = 0; i < numMuxBits; i++) {
    pinMode(muxSelectPins[i], OUTPUT);
  }

  // ADC 분주비16
  #if defined(__AVR_ATmega4809__)
    ADC0.CTRLC = ADC_PRESC_DIV16_gc;
  #else
    ADCSRA &= ~((1<<ADPS2)|(1<<ADPS1)|(1<<ADPS0));
    ADCSRA |= (1<<ADPS2);
  #endif
}

void loop() {
  for (int row = 0; row < numRows; row++) {
    selectRow(row);
    delayMicroseconds(30);

    for (int ch = 0; ch < (1<<numMuxBits); ch++) {
      selectMux(ch);
      delayMicroseconds(30);

      for (int dev = 0; dev < numMuxDevices; dev++) {
        int col = dev * 8 + ch;
        if (col >= numCols) continue;
        uint8_t v = analogRead(muxAnalogPins[dev]) >> 2;
        Serial.write(v);
      }
    }
  }
  delay(10);
}

void selectRow(int row) {
  digitalWrite(shiftLatchPin, LOW);
  for (int reg = 0; reg < numShiftRegs; reg++) {
    uint8_t b = 0;
    int base = reg * 8;
    if (row >= base && row < base + 8) b = 1 << (row - base);
    shiftOut(shiftDataPin, shiftClockPin, LSBFIRST, b);
  }
  digitalWrite(shiftLatchPin, HIGH);
}

void selectMux(int channel) {
  for (int i = 0; i < numMuxBits; i++) {
    digitalWrite(muxSelectPins[i], (channel >> i) & 1);
  }
}
