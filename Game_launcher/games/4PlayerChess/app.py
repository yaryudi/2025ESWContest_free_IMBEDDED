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
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QPlainTextEdit, QFrame
from PyQt5.QtGui import QIcon, QResizeEvent
from PyQt5.QtCore import QRect, Qt, QTimer
from gui.main import MainWindow
sys.path.append('./4PlayerChess-master/')
from actors.generate_actors import generate_actors


def main():

    """Creates application and main window and sets application icon."""
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('resources/img/icon.svg'))
    moves = None
    actors = []
    if len(sys.argv) > 1 and sys.argv[1] == 'moves':
      moves = eval(sys.argv[2])
    else:
      actors = generate_actors([*sys.argv])
    window = MainWindow(actors, moves)
    
    # 창을 항상 최상위에 유지하도록 설정 (한 번만)
    window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
    
    # 전체 화면으로 시작하므로 창 크기 설정 제거
    # window.resize(800, 800)
    
    # 전체 화면으로 시작하므로 창 중앙 배치 제거
    # screen = QRect(app.desktop().availableGeometry())
    # x = screen.left() + int((screen.width() - window.width()) / 2)
    # y = screen.top() + int((screen.height() - window.height()) / 2)
    # window.move(x, y)
    
    # Show window normally instead of fullscreen
    window.show()
    
    # 창을 맨 앞으로 가져오기
    window.raise_()
    window.activateWindow()
    window.setWindowState(window.windowState() | Qt.WindowActive)
    
    # 1초 간격으로 창을 맨 앞으로 가져오는 타이머
    def bring_to_front():
        # 최소화된 상태라면 복원
        if window.isMinimized():
            window.showNormal()
        
        # 강제로 전체화면으로 설정 (다른 창이 전체화면이어도)
        window.showFullScreen()
        
        # 창을 맨 앞으로 가져오기 (깜박임 방지)
        window.raise_()
        window.activateWindow()
        window.setWindowState(window.windowState() | Qt.WindowActive)
    
    timer = QTimer()
    timer.timeout.connect(bring_to_front)
    timer.start(1000)  # 1초 간격
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
