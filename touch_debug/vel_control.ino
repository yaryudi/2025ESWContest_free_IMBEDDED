#include <Arduino.h>

// =========================================================================
// ─── 설정 및 핀 정의 (2-Data-Chain 구조) ─────────────────────────────────
// =========================================================================

// ─── 하드웨어 정의 ──────────────────────────────────────────────────────
const int latchPin        = 10;           // 모든 칩의 RCLK (ST_CP) - 공유
const int clockPin        = 13;           // 모든 칩의 SRCLK (SH_CP) - 공유
const int dataPin1        = 11;           // 1번 체인(SR1-SR5)의 SER (DS)
const int dataPin2        = 8;            // 2번 체인(SR6-SR10)의 SER (DS) - 새로 추가

const int numRegsPerChain = 5;            // 각 체인당 쉬프트 레지스터 개수
const int numShiftRegs    = 10;           // 총 쉬프트 레지스터 개수
const int numRows         = numShiftRegs * 8;

// MUX 관련 핀 정의 (기존과 동일)
const int muxSelectPins[3] = {2, 3, 4};
const int numMuxDevices    = 8;
const int muxAnalogPins[8] = {A0, A1, A2, A3, A4, A5, A6, A7};
const int numCols          = 64;

// =========================================================================
// ─── 초기 설정 (Setup) ───────────────────────────────────────────────────
// =========================================================================
void setup() {
  Serial.begin(2000000);

  // 제어 핀 모두 OUTPUT으로 설정
  pinMode(latchPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin1, OUTPUT);
  pinMode(dataPin2, OUTPUT);
  
  // MUX, ADC 설정 (기존과 동일)
  for (int i = 0; i < 3; i++) {
    pinMode(muxSelectPins[i], OUTPUT);
  }
  #if defined(__AVR_ATmega4809__)
    ADC0.CTRLC = ADC_PRESC_DIV16_gc;
  #else
    ADCSRA = (ADCSRA & ~((1<<ADPS2)|(1<<ADPS1)|(1<<ADPS0))) | (1<<ADPS2);
  #endif

  Serial.println("2-Data-Chain sensor scan started.");
}

// =========================================================================
// ─── 핵심 기능 함수 (shiftOut 기반으로 수정됨) ───────────────────────────
// =========================================================================

/**
 * @brief 2개의 데이터 경로를 사용해 지정된 행(row) 하나만 활성화합니다.
 * @param row 활성화할 행의 번호 (0-79)
 */
inline void selectRow(int row) {
  // 1. 목표 행이 어느 체인에 속하는지 파악 (0-39: 첫째, 40-79: 둘째)
  bool isFirstChain = (row >= 0 && row < 40);
  
  // 2. 각 체인에 보낼 5바이트 데이터 계산
  uint8_t data1[numRegsPerChain] = {0}; // 1번 체인용 데이터
  uint8_t data2[numRegsPerChain] = {0}; // 2번 체인용 데이터
  
  if (row >= 0 && row < numRows) {
    int rowInChain = row % 40;
    int regIndex   = rowInChain / 8;
    int bitIndex   = rowInChain % 8;
    
    // 목표 체인에만 켜기 신호를, 다른 체인에는 끄기 신호(0)를 준비
    if (isFirstChain) {
      data1[numRegsPerChain - 1 - regIndex] = 1 << bitIndex;
    } else {
      data2[numRegsPerChain - 1 - regIndex] = 1 << bitIndex;
    }
  }

  // 3. 래치 핀을 LOW로 내려 데이터 전송 준비
  digitalWrite(latchPin, LOW);

  // 4. 두 데이터 경로로 5바이트씩 동시에 전송 (shiftOut 사용)
  for (int i = 0; i < numRegsPerChain; i++) {
    shiftOut(dataPin1, clockPin, MSBFIRST, data1[i]);
    shiftOut(dataPin2, clockPin, MSBFIRST, data2[i]);
  }

  // 5. 래치 핀을 HIGH로 올려 모든 칩의 출력을 동시에 갱신
  digitalWrite(latchPin, HIGH);
}

/**
 * @brief 8채널 MUX의 채널을 선택합니다.
 */
inline void selectMux(int ch) {
  for (int i = 0; i < 3; i++) {
    digitalWrite(muxSelectPins[i], (ch >> i) & 1);
  }
}

// =========================================================================
// ─── 메인 루프 (Loop) ─────────────────────────────────────────────────────
// =========================================================================
void loop() {
  Serial.write(0xAA);
  Serial.write(0x55);

  // 0번 행부터 79번 행까지 순차적으로 스캔
  for (int row = 0; row < numRows; row++) {
    selectRow(row);
    delayMicroseconds(10);

    for (int ch = 0; ch < 8; ch++) {
      selectMux(ch);
      delayMicroseconds(10);
      for (int dev = 0; dev < numMuxDevices; dev++) {
        uint8_t v = analogRead(muxAnalogPins[dev]) >> 2;
        Serial.write(v);
      }
    }
  }
}