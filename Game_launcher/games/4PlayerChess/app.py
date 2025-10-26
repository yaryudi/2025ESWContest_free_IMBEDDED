#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of the Four-Player Chess project, a four-player chess GUI.
#
# Copyright (C) 2018, GammaDeltaII
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import signal
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QPlainTextEdit, QFrame
from PyQt5.QtGui import QIcon, QResizeEvent
from PyQt5.QtCore import QRect, Qt, QTimer
from gui.main import MainWindow
sys.path.append('./4PlayerChess-master/')
from actors.generate_actors import generate_actors


def signal_handler(signum, frame):
    """시스템 시그널을 처리하여 프로그램을 안전하게 종료합니다."""
    print(f"\n시스템 시그널 {signum}을 받았습니다. 프로그램을 종료합니다.")
    # 현재 프로세스 종료
    os._exit(0)


def main():

    """Creates application and main window and sets application icon."""
    # 시스템 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 종료 시그널
    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('resources/img/icon.svg'))
    moves = None
    actors = []
    if len(sys.argv) > 1 and sys.argv[1] == 'moves':
      moves = eval(sys.argv[2])
    else:
      actors = generate_actors([*sys.argv])
    window = MainWindow(actors, moves)
    
    # 창을 맨 앞으로 가져오기 (showFullScreen은 main.py에서 처리됨)
    window.raise_()
    window.activateWindow()
    window.setWindowState(window.windowState() | Qt.WindowActive)
    
    print("\n=== 체스 게임 조작 방법 ===")
    print("ESC: 전체화면/창모드 전환")
    print("Ctrl+Q: 프로그램 종료")
    print("Ctrl+C: 프로그램 종료 (터미널에서)")
    print("S: 체스 말 소리 토글")
    print("B: BGM 토글")
    print("방향키: 게임 진행/되돌리기")
    print("==========================\n")
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
