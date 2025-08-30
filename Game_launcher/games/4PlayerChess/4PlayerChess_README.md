 # ♟️ 4인 체스 (4-Player Chess)

**젯슨나노 터치패드 최적화 4인 체스 게임**

[![License](https://img.shields.io/badge/license-GPL%20v3.0-blue.svg)](COPYING.md)  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)  [![PyQt5](https://img.shields.io/badge/framework-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)  [![Platform](https://img.shields.io/badge/platform-Jetson%20Nano-orange.svg)](https://developer.nvidia.com/embedded/jetson-nano)

<p align='center'>
	<img src='4Pchess_thumbnail.jpg' width=300 height=200 alt='4-Player Chess Game'>
</p>

## 📋 프로젝트 개요

**이 프로젝트**는 기존 4인 체스 게임을 젯슨나노 터치패드 환경에 최적화하여 수정한 버전입니다. 터치 인터페이스에 맞춰 UI를 개선하고, 게임 종료 및 카운터 기능을 추가하여 사용자 경험을 향상시켰습니다.

### 🎯 프로젝트 목표

* 🎮 **터치 최적화**: 젯슨나노 터치패드에 최적화된 UI/UX
* ♟️ **4인 체스 게임**: 빨강·노랑 vs 파랑·초록 팀 전투
* 🖥️ **풀스크린 모드**: 전체 화면 게임 경험
* 🔧 **게임 제어**: ESC, Ctrl+Q 등 다양한 종료 방법 지원
* 🎵 **사운드 시스템**: 체스 이동 및 게임 효과음 지원

## 🚀 주요 기능

1. **4인 체스 게임플레이**
   * 160칸 확장 보드 (기존 8×8에서 상하좌우 3줄씩 확장)
   * 팀 기반 전투: 빨강·노랑 vs 파랑·초록
   * 폰 승진 시스템 (11행 도달 시)
   * 체크메이트/스테일메이트 승리 조건

2. **터치패드 최적화**
   * 터치 인터페이스에 최적화된 UI
   * 직관적인 드래그 앤 드롭 체스 기물 이동
   * 반응형 보드 레이아웃

3. **게임 제어 시스템**
   * ESC 키: 즉시 게임 종료
   * Ctrl+Q: 프로그램 종료
   * Ctrl+C: 터미널에서 강제 종료
   * 창 최상위 유지 기능

## 🛠️ 설치 및 실행

### 필수 요구사항

* Python 3.8 이상
* PyQt5

## 🎮 게임 조작법

### 기본 조작
* **체스 기물 이동**: 터치하여 기물 선택 후 목적지 터치
* **드래그 앤 드롭**: 기물을 드래그하여 이동

### 게임 제어
* **ESC**: 게임 종료
* **Ctrl+Q**: 프로그램 종료
* **Ctrl+C**: 터미널에서 강제 종료

## 📄 라이선스

이 프로젝트는 [GNU General Public License v3.0](COPYING.md) 하에 배포됩니다.

### 출처
* **원본 프로젝트**: [mvvollmer/4PlayerChess](https://github.com/mvvollmer/4PlayerChess/tree/miles-branch)
* **라이선스**: [GNU GPL v3.0](https://github.com/GammaDeltaII/4PlayerChess/blob/master/COPYING.md)
