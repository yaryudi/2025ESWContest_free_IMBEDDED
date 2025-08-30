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

### 1버
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
