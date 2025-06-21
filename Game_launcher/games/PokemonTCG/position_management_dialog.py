from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont

class PositionManagementDialog(QDialog):
    position_swapped = pyqtSignal(int, str, str)  # (player_number, slot1_type, slot2_type)
    
    def __init__(self, parent=None, current_player=1):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("포지션 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.first_selected_slot = None
        self.slot_buttons = {}  # Store references to slot buttons

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
        description = QLabel("포지션을 관리할 슬롯을 선택하세요")
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
        self.slot_buttons[player_number] = {}
        # tool_management_dialog.py와 동일하게 2x3 그리드로 배치
        btns = [
            ("필드", "active", 0, 0),
            ("1", "bench_0", 0, 1),
            ("2", "bench_1", 1, 0),
            ("3", "bench_2", 1, 1),
            ("4", "bench_3", 2, 0),
            ("5", "bench_4", 2, 1),
        ]
        for text, slot_type, row, col in btns:
            btn = self.create_slot_button(text, player_number, slot_type)
            grid.addWidget(btn, row, col)
            self.slot_buttons[player_number][slot_type] = btn

    def create_slot_button(self, text, player_number, slot_type):
        button = QPushButton(text)
        button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: black;
                border: none;
                padding: 0px;
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
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        button.clicked.connect(lambda: self.handle_slot_click(player_number, slot_type))
        return button

    def handle_slot_click(self, player_number, slot_type):
        if player_number != self.current_player:
            return
            
        if self.first_selected_slot is None:
            # First selection
            self.first_selected_slot = slot_type
            # Disable other player's buttons
            other_player = 2 if player_number == 1 else 1
            for slot_btn in self.slot_buttons[other_player].values():
                slot_btn.setEnabled(False)
            # Make selected button appear disabled
            self.slot_buttons[player_number][slot_type].setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: #7f8c8d;
                    border: none;
                    padding: 0px;
                    border-radius: 5px;
                    font-size: 13px;
                    min-width: 44px;
                    max-width: 44px;
                    min-height: 44px;
                    max-height: 44px;
                    font-weight: bold;
                }
            """)
        else:
            if self.first_selected_slot == slot_type:
                # Deselect if clicking the same slot
                self.first_selected_slot = None
                # Re-enable other player's buttons
                other_player = 2 if player_number == 1 else 1
                for slot_btn in self.slot_buttons[other_player].values():
                    slot_btn.setEnabled(True)
                # Reset button style
                self.slot_buttons[player_number][slot_type].setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: black;
                        border: none;
                        padding: 0px;
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
                """)
            else:
                # Second selection - swap positions
                self.position_swapped.emit(player_number, self.first_selected_slot, slot_type)
                self.accept()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = PositionManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 