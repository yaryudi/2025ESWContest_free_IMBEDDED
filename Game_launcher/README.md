# 🎛️ 터치 캘리브레이션 & 게임 런처

**터치패드 캘리브레이션과 게임 런처를 결합한 통합 시스템**

[![License](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE)  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)  [![PyQt5](https://img.shields.io/badge/framework-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)  [![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-red.svg)](https://raspberrypi.org)

## 📋 프로젝트 개요

**이 프로젝트**는 Velostat 기반 터치패드 캘리브레이션과 GUI 기반 게임 런처를 결합하여, 사용자가 터치패드를 통해 마우스를 제어하고 다양한 파이썬 게임을 손쉽게 실행할 수 있도록 설계된 통합 솔루션입니다.

### 🎯 프로젝트 목표

* 🎯 **정밀 캘리브레이션**: 4지점 터치 좌표 보정 지원
* 🚀 **실시간 반응성**: 스레드를 이용한 빠른 터치 감지 및 마우스 제어
* 🎮 **원클릭 게임 실행**: 앨범 스타일의 직관적인 게임 런처
* 🔧 **확장성**: games 디렉터리 내 신규 게임 자동 로드
* 🛠️ **안정성**: psutil 기반 프로세스 중복 실행 방지

## 🚀 주요 기능

1. **cali\_launcher.py**

   * 터치패드 캘리브레이션 GUI와 게임 런처를 순차 실행
2. **calibration\_gui.py**

   * PyQt5 풀스크린 캘리브레이션 인터페이스
   * CalibrationThread로 터치 감지, MouseControlThread로 보정된 마우스 이동
   * 4개 지점(좌상·우상·좌하·우하) 캘리브레이션 단계 지원
3. **launcher.py**

   * PyQt5 기반 게임 런처 UI
   * `games/` 디렉터리 자동 스캔 및 앨범 스타일 목록 생성
   * 로딩 다이얼로그, 실행 중인 게임 중복 실행 방지 기능

## 🔧 하드웨어 구성

* **제어부**: Raspberry Pi / PC
* **터치 시스템**: Velostat 기반 압력 센서 매트
* **통신**: Serial (예: `/dev/ttyACM0`, 115200 baud)

## 💻 소프트웨어 스택

* **언어**: Python 3.8+
* **GUI**: PyQt5
* **수치 연산**: numpy
* **시리얼 통신**: pyserial
* **마우스 자동화**: pyautogui
* **프로세스 관리**: psutil

## 📁 프로젝트 구조

```
/  
├── cali_launcher.py        # 캘리브레이션 및 런처 실행기
├── calibration_gui.py      # 터치캘리브레이션 GUI
└── launcher.py             # 게임 런처 GUI
```

## 🛠️ 설치 및 실행

### 필수 요구사항

* Python 3.8 이상
* PyQt5
* numpy, pyserial, pyautogui, psutil

### 설치 방법

```bash
git clone <repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 실행 방법

```bash
# 통합 실행
python cali_launcher.py

# 개별 모듈 실행
python calibration_gui.py
python launcher.py
```

## 🔄 개발 진행 상황

* ✅ 캘리브레이션 실행기 구현
* ✅ 캘리베이션 GUI 완성
* ✅ 게임 런처 GUI 완성
* 🚧 신규 게임 자동 업데이트 기능 추가 예정

## 🔧 기술적 특징

* **CalibrationThread**: 10ms 단위 터치 프레임 실시간 감지
* **MouseControlThread**: 보정 행렬 기반 마우스 좌표 변환
* **PyQt5**: 풀스크린 모드 & 반응형 UI 구성
* **Process Guard**: psutil 활용한 중복 실행 방지 로직

## 📄 라이선스

이 프로젝트는 [GPL License](LICENSE) 하에 배포됩니다.
