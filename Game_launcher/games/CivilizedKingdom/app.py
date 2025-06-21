# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_gui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 키보드 단축키 안내 출력
    print("\n=== Civilized Kingdom 게임 키보드 단축키 ===")
    print("ESC: 전체 화면 켜기/끄기")
    print("==============================\n")
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
