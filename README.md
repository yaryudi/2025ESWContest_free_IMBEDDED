# 🎮 Imbedded - 테이블탑 프로젝터
**디지털과 아날로그가 만나는 새로운 게임 플랫폼**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Raspberry Pi](https://img.shields.io/badge/platform-Raspberry%20Pi-red.svg)](https://raspberrypi.org)

## 📋 프로젝트 개요

**Imbedded - 테이블탑 프로젝터**는 라즈베리파이 기반의 테이블탑 게임 플랫폼입니다. 프로젝터와 터치 시스템, AI 객체 인식을 결합하여 전통적인 보드게임에 디지털 요소를 더한 혁신적인 게임 경험을 제공합니다.

### 🎯 프로젝트 목표
- 💡 **편의성**: 직관적인 터치 인터페이스와 조작감
- ⚡ **성능**: 신속하고 정밀한 반응성
- 🔧 **확장가능성**: 다양한 게임 장르 지원
- 🛡️ **내구성**: 안정적인 하드웨어 구성
- 💰 **경제성**: 합리적인 제작 비용

## 🚀 주요 기능

### 🎮 지원 게임
1. **♟️ 체스 (4인 체스)**
   - 전통 체스의 4인 변형 버전
   - 실시간 게임 진행 상황 표시

2. **♠️ 카드 게임 (포커)**
   - YOLO 기반 실시간 카드 인식
   - 16,000장 카드 데이터베이스 지원
   - OpenCV 이미지 처리 파이프라인

3. **🎲 TRPG 게임**
   - GPT API 연동 스토리텔링
   - 자동 맵 생성 시스템
   - 실시간 상황 처리

### 🔧 하드웨어 구성
- **제어부**: Raspberry Pi 5
- **카메라**: IMX-708 광각 카메라
- **터치 시스템**: Velostat 기반 압력 센서
- **출력**: 프로젝터 기반 디스플레이
- **통신**: Arduino-Raspberry Pi 연동

### 💻 소프트웨어 스택
- **언어**: Python 3.8+
- **이미지 처리**: OpenCV, PIL
- **AI/ML**: YOLO (객체 인식)
- **GUI**: PyQt/Tkinter
- **API**: OpenAI GPT API
- **개발 도구**: VS Code, KiCad (PCB), Fusion 360

## 📁 프로젝트 구조

```
25Embedded/
├── Game_launcher/
│   ├── data/
│   ├── games/
│   │   ├── 4PlayerChess/        # 4인 체스 게임
│   │   ├── CivilizedKingdom/    # TRPG 게임 ‘Civilized Kingdom’
│   │   ├── PokemonTCG/          # 포켓몬 카드 게임
│   │   └── Poker/               # 포커 게임
│   ├── cali_launcher.py        # 캘리브레이션 & 런처 실행기
│   ├── calibration_gui.py      # 터치 캘리브레이션 GUI
│   └── launcher.py             # 게임 런처 GUI
├── touch_module/               # 터치 시스템 모듈
├── LICENSE                     # 라이선스 파일
└── README.md                   # 전체 프로젝트 설명
```

## 🛠️ 설치 및 실행

### 필수 요구사항
- Raspberry Pi 5 (4GB RAM 권장)
- Python 3.8 이상
- OpenCV 4.5+
- 카메라 모듈 (IMX-708 권장)

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/hyobon99/25Embedded.git
cd 25Embedded

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 게임 런처 실행
cd game_launcher
python main.py
```

### 🎮 게임별 실행
```bash
# 포커 게임 (최종 버전)
cd Poker_game_final_ver
python main.py

# 체스 게임
cd chess
python chess_game.py

# TRPG 게임
cd trpg
python trpg_main.py
```

## 🔄 개발 진행 상황

### ✅ 완료된 기능
- [o] 기본 하드웨어 설계 및 제작
- [o] 터치 시스템 구현
- [o] 카메라 기반 객체 인식
- [o] 체스 게임 완성
- [o] 포커 게임 완성 (카드 인식 포함)
- [o] TRPG 기본 시스템
- [o] 게임 런처 시스템

### 🚧 개발 중인 기능
- [ ] 터치 정확도 개선 (목표: 5mm)
- [ ] 추가 카드 게임 지원
- [ ] 온라인 멀티플레이어
- [ ] 모바일 앱 연동

## 🔧 기술적 특징

### AI 객체 인식
- **YOLO 모델**: 실시간 카드 인식
- **정확도**: 95% 이상 카드 식별
- **처리 속도**: 30fps 실시간 처리

### 터치 시스템
- **Velostat 센서**: 압력 감지 방식
- **해상도**: 현재 1cm, 목표 5mm
- **반응성**: <100ms 지연시간

### 이미지 처리 파이프라인
```python
# 예시: 카드 인식 파이프라인
def process_card_detection(frame):
    # 전처리
    processed = preprocess_image(frame)
    
    # YOLO 추론
    detections = yolo_model.detect(processed)
    
    # 후처리
    cards = postprocess_detections(detections)
    
    return cards
```

## 👥 팀 구성

| 팀원 | 역할 | 담당 업무 |
|------|------|-----------|
| **구○○** (팀장) | 프로젝트 관리, 터치 시스템 | 전체 일정 관리, 터치 하드웨어 설계 |
| **김○○** | 터치 시스템, 통합 | PCB 설계, 터치패드 설계 |
| **류○○** | 터치 시스템 | 체스 게임 디자인, 터치 프로그램 개발 |
| **조○○** | 기구 설계, 통합 | 외장 디자인, 3D 모델링, 하드웨어 조립 |
| **전○○** | AI/객체인식, API | YOLO 카드 인식, GPT API 연동 |

## 📊 성과 및 평가

### 주요 성과
- 🏆 **하드웨어 완성도**: 85%
- 💻 **소프트웨어 완성도**: 90%
- 🎮 **게임 구현**: 2개 장르 완성
- 🤖 **AI 정확도**: 95% 이상

### 예산 관리
- **초기 예산**: 183,000원
- **최종 예산**: 404,800원
- **주요 증액 요인**: 성능 향상을 위한 부품 업그레이드

## 🔮 향후 계획

### 단기 계획 (1-3개월)
- 터치 정확도 5mm 달성
- 추가 카드 게임 지원
- UI/UX 개선

### 장기 계획 (6개월+)
- 온라인 멀티플레이어 지원
- AI 대전 상대 구현
- 모바일 앱 연동
- 상용화 검토

## 📚 참고 자료

### 활용 오픈소스
- 체스 게임: GNU GPL 라이선스
- YOLO 카드 인식: GitHub 공개 모델
- 포켓몬 카드 API: 16,000장 데이터베이스

### 기술 문헌
1. M. Kciuk et al., "Intelligent medical Velostat pressure sensor mat based on artificial neural network and Arduino embedded system," Applied System Innovation, vol. 6, no. 5, p. 84, Sep. 2023
2. L. Yuan et al., "Velostat sensor array for object recognition," IEEE Sensors Journal, vol. 22, no. 2, pp. 1692–1701, Jan. 2022

## 📄 라이선스

이 프로젝트는 MIT License 하에 배포됩니다.

---

**"디지털과 아날로그가 만나는 새로운 게임 플랫폼"**

🎮 **Imbedded - 테이블탑 프로젝터** 🎮 
