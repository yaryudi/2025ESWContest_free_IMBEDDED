# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer
from gui.main_gui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 키보드 단축키 안내 출력
    print("\n=== Civilized Kingdom 게임 키보드 단축키 ===")
    print("ESC: 전체 화면 켜기/끄기")
    print("==============================\n")
    
    w = MainWindow()
    
    # 창을 항상 최상위에 유지하도록 설정 (한 번만)
    w.setWindowFlags(w.windowFlags() | Qt.WindowStaysOnTopHint)
    
    w.show()
    
    # 창을 맨 앞으로 가져오기
    w.raise_()
    w.activateWindow()
    w.setWindowState(w.windowState() | Qt.WindowActive)
    
    # 1초 간격으로 창을 맨 앞으로 가져오는 타이머
    def bring_to_front():
        # 최소화된 상태라면 복원
        if w.isMinimized():
            w.showNormal()
        
        # 강제로 전체화면으로 설정 (다른 창이 전체화면이어도)
        w.showFullScreen()
        
        # 창을 맨 앞으로 가져오기 (깜박임 방지)
        w.raise_()
        w.activateWindow()
        w.setWindowState(w.windowState() | Qt.WindowActive)
    
    timer = QTimer()
    timer.timeout.connect(bring_to_front)
    timer.start(1000)  # 1초 간격
    
    sys.exit(app.exec_())
