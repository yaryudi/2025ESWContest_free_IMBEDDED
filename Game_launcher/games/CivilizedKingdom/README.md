# TRPG Game

간단한 텍스트 기반 TRPG(테이블탑 롤플레잉 게임) 관리용 데스크탑 애플리케이션입니다.  
pyqt5를 사용한 Qt GUI로, 게임(세션) 생성·삭제, 캐릭터 등록·선택, 맵·전투 화면 전환 등의 기본 기능을 제공(할 예정)
pyqt5는 GPL v3.0 라이센스를 가진다. https://www.gnu.org/licenses/gpl-3.0.html


맵 생성의 다음의 오픈소스를 활용했다. https://github.com/QuasiStellar/Polytopia-Map-Generator
QuasiStellar의 Polytopia-Map-Generator는 GPL v3.0 라이센스를 가진다. https://www.gnu.org/licenses/gpl-3.0.html

/asset 이미지파일의 용량에 의해 이미지 파일을 제외하고 push하였음, https://github.com/QuasiStellar/Polytopia-Map-Generator에서 다운받기를 권장함.


## 주요 파일 역할

- **main.py**  
  애플리케이션 진입점.
  ```bash
  $ python main.py
  ```
- **ui/**
프로그램 실행 전 반드시 아래 명령으로 파이썬 UI 모듈을 생성해야 합니다.
  ```bash
  $ pyuic5 -x ui/mainwindow.ui -o ui/mainwindow.py
- **core/game_manager.py**

  게임 세션(생성시간 기반)의 로드·저장·추가·삭제 비즈니스 로직 담당.
  
- **gui/main_window.py**

  core와 ui를 연결하여 실제 창을 띄우고, 사용자 입력(버튼·리스트 클릭) 처리를 담당.
  
- **data/games.json**

  실행 시 동적으로 생성되는 게임 세션 정보를 보관하는 JSON 파일.
  

