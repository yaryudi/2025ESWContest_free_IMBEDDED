from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget
)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal

class StadiumManagementDialog(QDialog):
    stadium_completed = pyqtSignal(int)  # player_number
    def __init__(self, parent=None, current_player=1):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("스타디움 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.setup_ui()

    def setup_ui(self):
        content_widget = QWidget()
        content_widget.setFixedSize(250, 300)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 안내문구
        self.description = QLabel("스타디움 카드를\n배치한 뒤\n완료 버튼을 눌러주세요!")
        self.description.setAlignment(Qt.AlignCenter)
        self.description.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #2c3e50;
                margin: 8px;
            }
        """)
        main_layout.addWidget(self.description, alignment=Qt.AlignHCenter)

        # 카드 배치 대기 화면
        self.card_placement_widget = QWidget()
        placement_layout = QVBoxLayout()
        placement_layout.setContentsMargins(0, 0, 0, 0)
        placement_layout.setSpacing(0)

        # 중앙 정사각형 영역
        center_widget = QWidget()
        center_widget.setFixedSize(200, 100)
        center_widget.setStyleSheet("""
            QWidget {
                background-color: #f5f6fa;
                border: 2px dashed #b2bec3;
                border-radius: 10px;
            }
        """)
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(10, 10, 10, 10)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        back_button = QPushButton("뒤로")
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        back_button.clicked.connect(self.reject)

        complete_button = QPushButton("완료")
        complete_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        complete_button.clicked.connect(self.complete_stadium)

        button_layout.addWidget(back_button)
        button_layout.addWidget(complete_button)
        center_layout.addLayout(button_layout)

        placement_layout.addWidget(center_widget, alignment=Qt.AlignCenter)
        self.card_placement_widget.setLayout(placement_layout)
        main_layout.addWidget(self.card_placement_widget)

        content_widget.setLayout(main_layout)
        scene = QGraphicsScene(self)
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(content_widget)
        proxy.setTransformOriginPoint(QPointF(125, 150))
        proxy.setRotation(self.rotation)
        scene.addItem(proxy)
        view = QGraphicsView(scene, self)
        view.setStyleSheet("background: transparent;")
        view.setFrameShape(QGraphicsView.NoFrame)
        view.setAlignment(Qt.AlignCenter)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setFixedSize(250, 300)
        scene.setSceneRect(0, 0, 250, 300)
        proxy.setPos(0, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)
        self.setLayout(layout)
        if self.parent():
            parent_rect = self.parent().geometry()
            if self.current_player == 1:
                self.move(parent_rect.x() + 10, parent_rect.y() + 300)
            else:
                self.move(parent_rect.x() + parent_rect.width() - 260, parent_rect.y() + 120)

    def complete_stadium(self):
        self.stadium_completed.emit(self.current_player)
        self.accept()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = StadiumManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 