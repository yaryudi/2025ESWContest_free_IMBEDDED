# 🎮 Imbedded - 테이블탑 프로젝터
**디지털과 아날로그가 만나는 새로운 게임 플랫폼**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-Jetson%20Nano-orange.svg)](https://developer.nvidia.com/embedded/jetson-nano)

## 📋 프로젝트 개요

**Imbedded - 테이블탑 프로젝터**는 젯슨나노 기반의 테이블탑 게임 플랫폼입니다. 프로젝터와 터치 시스템, AI 객체 인식을 결합하여 전통적인 보드게임에 디지털 요소를 더한 혁신적인 게임 경험을 제공합니다.

### 🎯 프로젝트 목표
- 💡 **편의성**: 직관적인 터치 인터페이스와 조작감
- ⚡ **성능**: 신속하고 정밀한 반응성
- 🔧 **확장가능성**: 다양한 게임 장르 지원
- 🛡️ **내구성**: 안정적인 하드웨어 구성
- 💰 **경제성**: 합리적인 제작 비용
- 🚀 **젯슨나노 최적화**: ARM64 아키텍처에 최적화된 성능

## 🚀 주요 기능

### 🎮 지원 게임
1. **♟️ 4인 체스 (4-Player Chess)**
   - 전통 체스의 4인 변형 버전
   - 팀 기반 전투: 빨강·노랑 vs 파랑·초록
   - 160칸 확장 보드 (기존 8×8에서 상하좌우 3줄씩 확장)
   - AI 플레이어 지원 (Minimax, Random, Custom)
   - 터치패드 최적화 및 풀스크린 모드

2. **🎹 피아노 타일 (Piano Tiles)**
   - 화면 아래로 내려오는 피아노 타일을 터치하여 음악 연주
   - 4열 구조의 리듬 게임
   - 5개의 다른 곡 패턴 제공
   - 롱프레스 기능으로 연속 타일 자동 터치
   - 실제 피아노 음표 재생

3. **🎨 간단한 그림판 (SimplePaint)**
   - 터치패드 드래그 테스트용 그림판
   - 색상 선택 및 선 굵기 조절 (1-20px)
   - 지우개 및 전체 지우기 기능
   - 반응형 캔버스 시스템
   - 터치 이벤트 실시간 모니터링

4. **🃏 텍사스 홀덤 포커 (Poker)**
   - YOLO 기반 실시간 카드 인식
   - 52장 포커 카드 이미지 데이터베이스
   - AI 플레이어와 함께하는 포커 게임
   - 베팅 시스템 (체크, 베트, 콜, 레이즈, 폴드)
   - 포커 핸드 평가 엔진

### 🔧 하드웨어 구성
- **제어부**: NVIDIA Jetson Nano
- **카메라**: USB 카메라
- **터치 시스템**: 터치패드 또는 터치스크린
- **출력**: 프로젝터 기반 디스플레이
- **통신**: USB, GPIO 인터페이스

### 💻 소프트웨어 스택
- **언어**: Python 3.8+
- **이미지 처리**: OpenCV-headless 4.x, PIL
- **AI/ML**: Ultralytics YOLO
- **GUI**: PyQt5
- **게임 엔진**: Pygame (Piano Tiles)
- **운영체제**: Ubuntu 18.04+ / JetPack 4.6+

## 📁 프로젝트 구조

```
25Embedded/
├── Game_launcher/              # 게임 런처 시스템
│   ├── data/                   # 게임 데이터
│   │   ├── characters/         # 캐릭터 정보
│   │   └── game/              # 게임 설정
│   ├── games/                  # 게임 모음
│   │   ├── 4PlayerChess/      # 4인 체스 게임
│   │   │   ├── 4PlayerChess_README.md
│   │   │   ├── app.py
│   │   │   ├── gui/           # GUI 모듈
│   │   │   ├── actors/        # 알고리즘 관련
│   │   │   └── resources/     # 게임 리소스
│   │   ├── Piano Tiles/       # 피아노 타일 게임
│   │   │   ├── PianoTiles_README.md
│   │   │   ├── app.py
│   │   │   ├── objects.py
│   │   │   ├── Assets/        # 게임 이미지
│   │   │   ├── Sounds/        # 게임 사운드
│   │   │   └── Fonts/         # 게임 폰트
│   │   ├── SimplePaint/       # 간단한 그림판
│   │   │   ├── SimplePaint_README.md
│   │   │   ├── app.py
│   │   │   ├── simple_paint.py
│   │   │   └── thumbnail.png
│   │   └── Poker/             # 포커 게임
│   │       ├── Poker_README.md
│   │       ├── app.py
│   │       ├── poker_game.py
│   │       ├── hand_evaluator.py
│   │       ├── assets/        # 카드 이미지
│   │       └── playingCards.pt # YOLO 모델
│   ├── launcher.py             # 메인 게임 런처
│   ├── main.py                 # 런처 진입점
│   └── DRAG_SUPPORT_README.md # 드래그 지원 시스템
├── image_test/                 # 이미지 테스트 모듈
├── touch_test/                 # 터치 테스트 모듈
├── LICENSE                     # 라이선스 파일
└── README.md                   # 전체 프로젝트 설명
```

## 🛠️ 설치 및 실행

### 필수 요구사항
- NVIDIA Jetson Nano
- Python 3.8 이상
- OpenCV-headless 4.x
- PyQt5
- 터치패드 또는 터치스크린

### 젯슨나노 환경 설정
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade

# Python 및 기본 패키지 설치
sudo apt install python3-pip python3-pyqt5 python3-opencv-headless

# YOLO 및 딥러닝 라이브러리 설치
pip3 install ultralytics torch torchvision

# 추가 의존성 설치
sudo apt install python3-dev libpython3-dev libopencv-dev
```

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/hyobon99/25Embedded.git
cd 25Embedded

# 캘리브레이션 활성화 및 게임 런처 실행
cd Game_launcher
python3 main.py

# 게임 런처 실행
cd Game_launcher
python3 launcher.py
```

### 🎮 게임별 실행
```bash
# 게임 런처를 통한 실행 (권장)
python3 launcher.py

# 개별 게임 직접 실행
cd games/4PlayerChess
python3 app.py

cd ../Piano\ Tiles
python3 app.py

cd ../SimplePaint
python3 app.py

cd ../Poker
python3 app.py
```

## 🔄 개발 진행 상황

### ✅ 완료된 기능
- [x] 기본 하드웨어 설계 및 제작
- [x] 터치 시스템 구현
- [x] 카메라 기반 객체 인식
- [x] 4인 체스 게임 완성
- [x] 피아노 타일 게임 완성
- [x] 간단한 그림판 완성
- [x] 포커 게임 완성 (카드 인식 포함)
- [x] 게임 런처 시스템
- [x] 젯슨나노 환경 최적화

### 🚧 개발 중인 기능
- [ ] 터치 정확도 개선 (목표: 5mm)
- [ ] 추가 게임 지원

## 🔧 기술적 특징

### AI 객체 인식
- **YOLO 모델**: 실시간 카드 인식
- **정확도**: 95% 이상 카드 식별

### 터치 시스템
- **터치패드**: 터치 인터페이스 최적화
- **해상도**: 현재 1cm, 목표 5mm
- **반응성**: <100ms 지연시간
- **드래그 지원**: 롱프레스 및 드래그 앤 드롭

### 게임 엔진
```python
# 예시: 게임 런처 구조
class GameLauncher:
    def __init__(self):
        self.games = {
            '4PlayerChess': ChessGame(),
            'PianoTiles': PianoTilesGame(),
            'SimplePaint': SimplePaintGame(),
            'Poker': PokerGame()
        }
    
    def launch_game(self, game_name):
        if game_name in self.games:
            self.games[game_name].start()
```

## 👥 팀 구성

| 팀원 | 역할 | 담당 업무 |
|------|------|-----------|
| **구○○** (팀장) | 프로젝트 관리, 터치 시스템 | 전체 일정 관리, 터치 하드웨어 설계 |
| **김○○** | 터치 시스템, 통합 | PCB 설계, 터치패드 설계 |
| **류○○** | 터치 시스템 | 회로 설계, 터치 프로그램 개발 |
| **조○○** | 기구 설계, 통합 | 외장 디자인, 3D 모델링, 하드웨어 조립 |
| **전○○** | AI/객체인식, API | YOLO 카드 인식, GPT API 연동 |

## 📊 성과 및 평가

### 주요 성과
- 🏆 **하드웨어 완성도**: 90%
- 💻 **소프트웨어 완성도**: 95%
- 🎮 **게임 구현**: 4개 장르 완성
- 🤖 **AI 정확도**: 95% 이상
- 🚀 **젯슨나노 최적화**: 완료

### 성능 지표
- **카드 인식 정확도**: 95% 이상
- **터치 반응성**: <100ms
- **게임 실행 속도**: 평균 30초/라운드
- **메모리 사용량**: 기본 실행 시 <100MB

## 📚 참고 자료

### 활용 오픈소스
- **4인 체스**: GNU GPL v3.0 라이선스
- **피아노 타일**: MIT 라이선스
- **간단한 그림판**: MIT 라이선스
- **포커 게임**: MIT 라이선스
- **YOLO 카드 인식**: Ultralytics YOLO

### 기술 문헌
1. M. Kciuk et al., "Intelligent medical Velostat pressure sensor mat based on artificial neural network and Arduino embedded system," Applied System Innovation, vol. 6, no. 5, p. 84, Sep. 2023
2. L. Yuan et al., "Velostat sensor array for object recognition," IEEE Sensors Journal, vol. 22, no. 2, pp. 1692–1701, Jan. 2022
3. NVIDIA Jetson Nano Developer Guide, NVIDIA Corporation, 2023

## 🐛 알려진 이슈

### 하드웨어 관련
- 일부 터치패드에서 압력 감지 정확도 차이
- 특정 조명 조건에서 카드 인식 정확도 감소

### 소프트웨어 관련
- 대용량 모델 로딩 시 초기 실행 시간 증가
- 매우 빠른 게임 진행 시 UI 지연 가능
- 일부 해상도에서 UI 요소 배치 조정 필요

## 📄 라이선스

이 프로젝트는 MIT License 하에 배포됩니다.

### 출처
- **4인 체스**: [mvvollmer/4PlayerChess](https://github.com/mvvollmer/4PlayerChess)
- **피아노 타일**: [pyGuru123/Python-Games](https://github.com/pyGuru123/Python-Games)
- **간단한 그림판**: 터치패드 드래그 테스트용으로 제작
- **포커 게임**: 텍사스 홀덤 포커 게임으로 제작

---

**"디지털과 아날로그가 만나는 새로운 게임 플랫폼"**

🎮 **Imbedded - 테이블탑 프로젝터** 🎮

*젯슨나노 환경에 최적화된 게임 플랫폼*
