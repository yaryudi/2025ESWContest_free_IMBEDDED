"""
플레이어 수를 선택하는 다이얼로그 창
2~5명의 플레이어 수를 선택할 수 있으며, 선택 후 게임을 시작합니다.
"""

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from poker_game import PokerGame


class PlayerCountDialog(QWidget):
    def __init__(self):
        super().__init__()
        # 기본 창 설정
        self.setWindowTitle("플레이어 수 선택")
        self.setFixedSize(1280, 720)
        self.setStyleSheet("background-color: black;")
        self.player_count = 2  # 기본 플레이어 수는 2명

        # 메인 컨테이너 박스 설정
        self.box = QWidget(self)
        self.box.setStyleSheet("background-color: white; border-radius: 15px;")
        self.box.setGeometry(440, 210, 400, 300)

        # 안내 텍스트 라벨
        self.label = QLabel("플레이어 수를 선택하세요", self.box)
        self.label.setFont(QFont("Arial", 16))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(50, 30, 300, 40)

        # 플레이어 수 표시 라벨
        self.num_display = QLabel(str(self.player_count), self.box)
        self.num_display.setFont(QFont("Arial", 48, QFont.Bold))
        self.num_display.setAlignment(Qt.AlignCenter)
        self.num_display.setGeometry(120, 100, 100, 100)

        # 플레이어 수 증가 버튼
        self.up_button = QPushButton("↑", self.box)
        self.up_button.setFixedSize(60, 60)
        self.up_button.setFont(QFont("Arial", 18, QFont.Bold))
        self.up_button.setStyleSheet("QPushButton { background-color: #2E8B57; color: white; border-radius: 30px; } QPushButton:hover { background-color: #3CB371; }")
        self.up_button.move(260, 95)
        self.up_button.clicked.connect(self.increment)

        # 플레이어 수 감소 버튼
        self.down_button = QPushButton("↓", self.box)
        self.down_button.setFixedSize(60, 60)
        self.down_button.setFont(QFont("Arial", 18, QFont.Bold))
        self.down_button.setStyleSheet("QPushButton { background-color: #2E8B57; color: white; border-radius: 30px; } QPushButton:hover { background-color: #3CB371; }")
        self.down_button.move(260, 165)
        self.down_button.clicked.connect(self.decrement)

        # 게임 시작 버튼
        self.start_button = QPushButton("게임 시작", self.box)
        self.start_button.setFixedSize(200, 60)
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.setStyleSheet("QPushButton { background-color: #d9534f; color: white; border-radius: 30px; } QPushButton:hover { background-color: #c9302c; }")
        self.start_button.move(100, 230)
        self.start_button.clicked.connect(self.accept)

    def increment(self):
        """플레이어 수를 1 증가시킵니다. 최대 5명까지 가능합니다."""
        if self.player_count < 5:
            self.player_count += 1
            self.num_display.setText(str(self.player_count))

    def decrement(self):
        """플레이어 수를 1 감소시킵니다. 최소 2명까지 가능합니다."""
        if self.player_count > 2:
            self.player_count -= 1
            self.num_display.setText(str(self.player_count))

    def accept(self):
        """선택된 플레이어 수로 게임을 시작합니다."""
        self.close()
        self.game = PokerGame(self.player_count)
        self.game.show()