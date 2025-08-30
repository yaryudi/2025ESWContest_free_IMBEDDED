# 🎹 Piano Tiles - 피아노 타일 터치 게임

**젯슨나노 터치패드 최적화 피아노 타일 게임**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)  [![Pygame](https://img.shields.io/badge/framework-Pygame-green.svg)](https://pygame.org)  [![Platform](https://img.shields.io/badge/platform-Jetson%20Nano-orange.svg)](https://developer.nvidia.com/embedded/jetson-nano)

<p align='center'>
	<img src='app.png' width=200 height=300 alt='Piano Tiles Game'>
</p>

## 📋 프로젝트 개요

**Piano Tiles**는 화면 아래로 내려오는 피아노 타일을 터치하여 음악을 연주하는 리듬 게임입니다. 젯슨나노 터치패드 환경에 최적화되어 있으며, 실제 피아노 음을 재생하면서 게임을 즐길 수 있습니다.

### 🎯 프로젝트 목표

* 🎹 **음악 게임**: 피아노 타일을 터치하여 음악 연주
* 🎮 **터치 최적화**: 젯슨나노 터치패드에 최적화된 UI/UX
* 🎵 **실제 음악**: 실제 피아노 음표 재생
* 📱 **반응형 디자인**: 다양한 화면 크기에 대응
* 🏆 **점수 시스템**: 점수 기록 및 최고 점수 추적

## 🚀 주요 기능

### 1. **피아노 타일 게임플레이**
* **4열 구조**: 화면을 4개 열로 나누어 타일 생성
* **랜덤 생성**: 각 열에 랜덤하게 타일이 생성됨
* **속도 증가**: 점수가 올라갈수록 타일 속도 증가
* **연속 타일**: 같은 열에 연속으로 타일이 생성될 수 있음

### 2. **터치 인터페이스**
* **터치/마우스**: 터치패드 또는 마우스로 타일 클릭
* **롱프레스**: 같은 열의 연속 타일을 순서대로 자동 터치
* **정확한 터치**: 빈 공간을 터치하면 게임오버
* **반응형 UI**: 화면 크기에 맞춰 자동 조정

### 3. **음악 시스템**
* **실제 피아노 음**: 4개의 다른 곡 패턴 제공
* **음표 재생**: 타일 터치 시 해당 음표 재생
* **배경음악**: 게임 중 계속 재생되는 배경음악
* **사운드 제어**: 사운드 온/오프 기능

### 4. **게임 제어**
* **시작 버튼**: 게임 시작
* **재시작 버튼**: 게임오버 후 재시작
* **사운드 버튼**: 배경음악 및 효과음 제어
* **종료 버튼**: 게임 종료

## 🔧 하드웨어 요구사항

* **제어부**: NVIDIA Jetson Nano / PC
* **터치 시스템**: 터치패드 또는 터치스크린
* **디스플레이**: 최소 800×600 해상도 권장
* **메모리**: 최소 2GB RAM 권장
* **저장공간**: 최소 4GB eMMC/SSD
* **오디오**: 스피커 또는 헤드폰

## 💻 소프트웨어 스택

* **언어**: Python 3.8+
* **게임 엔진**: Pygame
* **오디오**: Pygame Mixer
* **UI**: Pygame Graphics
* **운영체제**: Ubuntu 18.04+ / JetPack 4.6+

## 📁 프로젝트 구조

```
Piano Tiles/
├── app.py                    # 메인 게임 애플리케이션
├── objects.py                # 게임 객체 클래스들
├── note_editor.py            # 음표 편집기
├── notes.json                # 음악 패턴 데이터
├── Assets/                   # 게임 이미지 리소스
│   ├── bg.png               # 배경 이미지
│   ├── piano.png            # 피아노 이미지
│   ├── title.png            # 게임 제목
│   ├── start.png            # 시작 버튼
│   ├── replay.png           # 재시작 버튼
│   ├── closeBtn.png         # 종료 버튼
│   ├── soundOnBtn.png       # 사운드 온 버튼
│   ├── soundOffBtn.png      # 사운드 오프 버튼
│   └── red overlay.png      # 게임오버 오버레이
├── Sounds/                   # 게임 사운드 파일
│   ├── piano-buzzer.mp3     # 게임오버 효과음
│   ├── piano-bgmusic.mp3    # 배경음악
│   └── *.ogg                # 피아노 음표 파일들
└── Fonts/                    # 게임 폰트
    ├── Alternity-8w7J.ttf   # 제목 폰트
    ├── Futura condensed.ttf  # 점수 폰트
    └── Marker Felt.ttf      # 게임 폰트
```

## 🛠️ 설치 및 실행

### 필수 요구사항

* Python 3.8 이상
* Pygame
* 터치패드 또는 마우스 입력 장치
* NVIDIA Jetson Nano (권장)

### 젯슨나노 환경 설정

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade

# Python 및 Pygame 설치
sudo apt install python3-pip python3-pygame

# 추가 의존성 설치
sudo apt install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
```

### 실행 방법

```bash
# 게임 런처를 통한 실행 (권장)
python3 launcher.py

# 직접 실행
python3 app.py
```

## 🎮 게임 조작법

### 기본 조작
* **타일 터치**: 피아노 타일을 터치하여 점수 획득
* **정확한 터치**: 빈 공간을 터치하면 게임오버
* **롱프레스**: 같은 열의 연속 타일을 순서대로 자동 터치

### 게임 제어
* **시작**: START 버튼으로 게임 시작
* **재시작**: REPLAY 버튼으로 게임 재시작
* **사운드**: 사운드 버튼으로 음악 제어
* **종료**: CLOSE 버튼으로 게임 종료

### 게임 규칙
* 타일이 화면 아래에 도달하면 게임오버
* 점수가 올라갈수록 타일 속도 증가
* 최고 점수 기록 및 추적

## 🎼 음악 시스템

### 곡 패턴
* **곡 1**: "Twinkle Twinkle Little Star"
* **곡 2**: "Happy Birthday"
* **곡 3**: "Three Bears"
* **곡 4**: "Airplane"

### 음표 시스템
* 실제 피아노 음표 사용 (c4, g4, a4, c5 등)
* .ogg 포맷의 고품질 사운드
* 다중 채널 오디오 지원

## 🔄 개발 진행 상황

* ✅ 기본 게임플레이 구현
* ✅ 터치 인터페이스 최적화
* ✅ 음악 시스템 통합
* ✅ 반응형 UI 구현
* ✅ 롱프레스 기능 추가
* ✅ 젯슨나노 환경 최적화
* 🚧 추가 곡 패턴 개발 예정

## 🔧 기술적 특징

* **터치 최적화**: 터치 인터페이스에 맞춘 UI/UX 설계
* **반응형 디자인**: 다양한 화면 크기에 대응하는 레이아웃
* **오디오 엔진**: Pygame Mixer 기반 고품질 사운드
* **게임 엔진**: Pygame 기반 게임 로직
* **젯슨나노 최적화**: ARM64 아키텍처에 최적화된 성능
* **메모리 효율성**: 최적화된 리소스 관리

## 📊 성능 최적화

* **렌더링 최적화**: 화면 크기에 맞춘 동적 스케일링
* **메모리 관리**: 효율적인 스프라이트 그룹 관리
* **오디오 최적화**: 다중 채널 오디오 시스템
* **터치 반응성**: 빠른 터치 입력 처리

## 🐛 알려진 이슈

* 일부 오디오 파일에서 지연 발생 가능
* 매우 빠른 속도에서 터치 인식 정확도 감소
* 특정 화면 해상도에서 UI 배치 조정 필요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

### 출처
* **원본 프로젝트**: [pyGuru123/Python-Games](https://github.com/pyGuru123/Python-Games)
* **개발자**: Prajjwal Pathak (pyguru)
* **수정**: 젯슨나노 환경 최적화

---
