"""
포커 게임의 메인 진입점
플레이어 수를 입력받고 게임을 시작하는 역할을 합니다.
"""

import sys
from PyQt5.QtWidgets import QApplication
from player_count_dialog import PlayerCountDialog

if __name__ == '__main__':
    # PyQt 애플리케이션 초기화
    app = QApplication(sys.argv)
    
    # PlayerCountDialog를 사용하여 플레이어 수 입력 받기
    dialog = PlayerCountDialog()
    dialog.show()
    
    # 애플리케이션 이벤트 루프 시작
    sys.exit(app.exec_())