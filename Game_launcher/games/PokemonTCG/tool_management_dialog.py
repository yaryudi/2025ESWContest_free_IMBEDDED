from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont

class ToolManagementDialog(QDialog):
    slot_selected = pyqtSignal(int, str)
    def __init__(self, parent=None, current_player=1, slot_tool_state=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("도구 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.slot_tool_state = slot_tool_state if slot_tool_state else {}
        self.setup_ui()

    def setup_ui(self):
        content_widget = QWidget()
        content_widget.setFixedSize(250, 300)
        content_widget.setContentsMargins(0, 0, 0, 0)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        # 슬롯 선택 화면
        self.slot_select_widget = QWidget()
        slot_layout = QVBoxLayout()
        description = QLabel("도구를 관리할 슬롯을 선택하세요")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                margin: 2px;
            }
        """)
        slot_layout.addSpacing(18)
        slot_layout.addWidget(description, alignment=Qt.AlignHCenter)
        slot_layout.addSpacing(0)
        player_layout = QHBoxLayout()
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(2)
        player_layout.addStretch(1)
        # 플레이어 1 슬롯
        player1_widget = QWidget()
        player1_widget.setMinimumWidth(110)
        player1_widget.setMaximumWidth(110)
        player1_layout = QVBoxLayout(player1_widget)
        player1_layout.setAlignment(Qt.AlignCenter)
        player1_label = QLabel("플레이어 1")
        player1_label.setAlignment(Qt.AlignCenter)
        player1_layout.addWidget(player1_label)
        player1_slots = QGridLayout()
        player1_slots.setAlignment(Qt.AlignCenter)
        self.create_slot_buttons(player1_slots, 1)
        player1_layout.addLayout(player1_slots)
        player_layout.addWidget(player1_widget, 2)
        player_layout.addStretch(1)
        # 플레이어 2 슬롯
        player2_widget = QWidget()
        player2_widget.setMinimumWidth(110)
        player2_widget.setMaximumWidth(110)
        player2_layout = QVBoxLayout(player2_widget)
        player2_layout.setAlignment(Qt.AlignCenter)
        player2_label = QLabel("플레이어 2")
        player2_label.setAlignment(Qt.AlignCenter)
        player2_layout.addWidget(player2_label)
        player2_slots = QGridLayout()
        player2_slots.setAlignment(Qt.AlignCenter)
        self.create_slot_buttons(player2_slots, 2)
        player2_layout.addLayout(player2_slots)
        player_layout.addWidget(player2_widget, 2)
        player_layout.addStretch(1)
        slot_layout.addLayout(player_layout, stretch=4)
        slot_layout.addStretch(1)
        cancel_button = QPushButton("취소")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: black;
                border: none;
                padding: 10px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        slot_layout.addWidget(cancel_button, alignment=Qt.AlignHCenter)
        slot_layout.addSpacing(18)
        self.slot_select_widget.setLayout(slot_layout)
        main_layout.addWidget(self.slot_select_widget)
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

    def create_slot_buttons(self, grid, player_number):
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for row in range(3):
            grid.setRowMinimumHeight(row, 0)
            grid.setRowStretch(row, 1)
        grid.addWidget(self.create_slot_button("필드", player_number, "active"), 0, 0)
        grid.addWidget(self.create_slot_button("1", player_number, "bench_0"), 0, 1)
        grid.addWidget(self.create_slot_button("2", player_number, "bench_1"), 1, 0)
        grid.addWidget(self.create_slot_button("3", player_number, "bench_2"), 1, 1)
        grid.addWidget(self.create_slot_button("4", player_number, "bench_3"), 2, 0)
        grid.addWidget(self.create_slot_button("5", player_number, "bench_4"), 2, 1)

    def create_slot_button(self, text, player_number, slot_type):
        button = QPushButton(text)
        key = (player_number, slot_type)
        # 흐릿한 스타일
        dim_style = '''
            QPushButton {
                background-color: #b2bec3;
                color: #888;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                min-width: 44px;
                max-width: 44px;
                min-height: 44px;
                max-height: 44px;
                font-weight: bold;
            }
        '''
        normal_style = '''
            QPushButton {
                background-color: #3498db;
                color: black;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                min-width: 44px;
                max-width: 44px;
                min-height: 44px;
                max-height: 44px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        '''
        if self.slot_tool_state.get(key, (False, None))[0]:
            button.setStyleSheet(dim_style)
        else:
            button.setStyleSheet(normal_style)
        button.clicked.connect(lambda: self.on_slot_button_clicked(player_number, slot_type))
        return button

    def on_slot_button_clicked(self, player_number, slot_type):
        key = (player_number, slot_type)
        if self.slot_tool_state.get(key):
            # 이미 장착된 경우: 해제
            self.slot_selected.emit(player_number, slot_type)
            self.accept()
        else:
            self.slot_selected.emit(player_number, slot_type)
            self.accept()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = ToolManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 