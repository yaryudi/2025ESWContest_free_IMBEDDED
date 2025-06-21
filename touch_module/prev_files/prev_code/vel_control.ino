#include <Arduino.h>

//최적화용 코드

// Velostat 센서 어레이 제어용 Arduino 코드 (40×30)
// • 5×8-bit Shift Register → 40개 행(Row) 제어
// • 4×8-ch MUX(CD4051) → 32개 열 중 앞 30개 열(Column) 선택
// • MUX Select 핀 3개는 공통, COM 핀은 A0~A3로 분리

// ─── 핀 정의 ─────────────────────────────────────────────
const int shiftDataPin  = 2;   // DS (시리얼 데이터 입력)
const int shiftClockPin = 3;   // SH_CP (클럭, 모든 SR에 병렬)
const int shiftLatchPin = 4;   // ST_CP (래치)

const int numShiftRegs = 5;    // 5 × 8 = 40개 행

// MUX 선택 핀 (S0, S1, S2) – 공통으로 물림
const int muxSelectPins[3] = {5, 6, 7};
const int numMuxBits       = 3;

// MUX 공통 출력(COM) – 각각 A0~A3에 연결
const int muxAnalogPins[] = {A0, A1, A2, A3};
const int numMuxDevices   = sizeof(muxAnalogPins) / sizeof(muxAnalogPins[0]);

const int numCols = 30;      // 실제로 읽을 열 개수 (4×8=32 중 앞 30개)
const int numRows = numShiftRegs * 8;  // 40

void setup() {
  Serial.begin(115200);
  pinMode(shiftDataPin,  OUTPUT);
  pinMode(shiftClockPin, OUTPUT);
  pinMode(shiftLatchPin, OUTPUT);
  for (int i = 0; i < numMuxBits; i++) {
    pinMode(muxSelectPins[i], OUTPUT);
  }

  // ADC Prescaler 변경 (기본값 128 -> 16)
  ADCSRA &= ~((1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0));
  ADCSRA |= (1 << ADPS2);
}

void loop() {
  // 각 행을 차례로 활성화
  for (int row = 0; row < numRows; row++) {
    selectRow(row);
    delayMicroseconds(50);  // 행 선택 안정화

    // MUX 채널별로 데이터 수집 (0-7)
    for (int mux_ch = 0; mux_ch < 8; mux_ch++) {
      selectMux(mux_ch);
      delayMicroseconds(5);  // MUX 안정화 시간

      // 각 MUX 디바이스에서 데이터 읽기
      for (int dev = 0; dev < numMuxDevices; dev++) {
        int col = dev * 8 + mux_ch;  // 실제 열 인덱스 계산
        if (col >= numCols) continue;

        unsigned char val_byte = analogRead(muxAnalogPins[dev]) >> 2;
        Serial.write(val_byte);
      }
    }
  }
  delay(50);  // 전체 스캔 주기
}

// ─── 행 선택 (40비트 중 하나만 '1') ──────────────────────
void selectRow(int row) {
  digitalWrite(shiftLatchPin, LOW);
  for (int reg = 0; reg < numShiftRegs; reg++) {
    uint8_t b = 0;
    int startBit = reg * 8;
    if (row >= startBit && row < startBit + 8) {
      b = 1 << (row - startBit);
    }
    shiftOut(shiftDataPin, shiftClockPin, LSBFIRST, b);
  }
  digitalWrite(shiftLatchPin, HIGH);
}

// ─── MUX 채널 선택 (0~7) ────────────────────────────────
void selectMux(int channel) {
  for (int i = 0; i < numMuxBits; i++) {
    digitalWrite(muxSelectPins[i], (channel >> i) & 1);
  }
}