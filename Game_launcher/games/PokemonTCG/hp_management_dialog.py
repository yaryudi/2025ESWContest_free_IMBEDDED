from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QSpinBox, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont

class HPManagementDialog(QDialog):
    slot_selected = pyqtSignal(int, str)  # (player_number, slot_type)
    hp_updated = pyqtSignal(int, str, int, int)  # (player_number, slot_type, bonus_hp, damage)
    
    def __init__(self, parent=None, current_player=1, slot_hp_state=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("체력 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.selected_slot = None
        self.bonus_hp = 0
        self.damage = 0

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
        description = QLabel("체력을 관리할 슬롯을 선택하세요")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: black;
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

        # 체력 관리 화면
        self.hp_manage_widget = QWidget()
        hp_layout = QVBoxLayout()
        hp_layout.setContentsMargins(10, 10, 10, 10)  # 여백 추가
        
        # 현재 선택된 슬롯 표시
        self.selected_slot_label = QLabel()
        self.selected_slot_label.setAlignment(Qt.AlignCenter)
        self.selected_slot_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: black;
                margin: 2px;
            }
        """)
        hp_layout.addWidget(self.selected_slot_label)
        
        # 기본 체력 표시
        self.current_hp_label = QLabel()
        self.current_hp_label.setAlignment(Qt.AlignCenter)
        self.current_hp_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: black;
                margin: 10px;
            }
        """)
        hp_layout.addWidget(self.current_hp_label)
        
        # 추가 체력/누적 피해/KO 버튼을 한 그리드에 정렬
        stat_grid = QGridLayout()
        stat_grid.setHorizontalSpacing(8)
        stat_grid.setVerticalSpacing(8)
        stat_grid.setColumnStretch(0, 0)
        stat_grid.setColumnStretch(1, 0)
        stat_grid.setColumnStretch(2, 0)
        stat_grid.setColumnMinimumWidth(2, 0)
        stat_grid.setAlignment(Qt.AlignCenter)
        # 추가 체력
        bonus_hp_label = QLabel("추가 체력:")
        self.bonus_hp_spin = QSpinBox()
        self.bonus_hp_spin.setRange(0, 999)
        self.bonus_hp_spin.setSingleStep(10)
        self.bonus_hp_spin.setFixedWidth(70)
        self.bonus_hp_spin.setFixedHeight(40)
        self.bonus_hp_spin.setKeyboardTracking(False)
        self.bonus_hp_spin.lineEdit().setReadOnly(True)
        self.bonus_hp_spin.valueChanged.connect(self.update_bonus_hp)
        stat_grid.addWidget(bonus_hp_label, 0, 0, alignment=Qt.AlignRight)
        stat_grid.addWidget(self.bonus_hp_spin, 0, 1)
        # KO 버튼 (빈칸 대신 QSpacerItem)
        stat_grid.addItem(QSpacerItem(0, 0), 0, 2)
        # 누적 피해
        damage_label = QLabel("누적 피해:")
        self.damage_spin = QSpinBox()
        self.damage_spin.setRange(0, 999)
        self.damage_spin.setSingleStep(10)
        self.damage_spin.setFixedWidth(70)
        self.damage_spin.setFixedHeight(40)
        self.damage_spin.setKeyboardTracking(False)
        self.damage_spin.lineEdit().setReadOnly(True)
        self.damage_spin.valueChanged.connect(self.update_damage)
        stat_grid.addWidget(damage_label, 1, 0, alignment=Qt.AlignRight)
        stat_grid.addWidget(self.damage_spin, 1, 1)
        # KO 버튼
        self.ko_button = QPushButton("KO")
        self.ko_button.setFixedSize(40, 40)
        self.ko_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: black;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.ko_button.clicked.connect(self.set_ko_damage)
        stat_grid.addWidget(self.ko_button, 1, 2)
        hp_layout.addLayout(stat_grid)
        
        # 버튼들 (중앙, 잘림 방지)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)
        button_layout.setAlignment(Qt.AlignCenter)
        self.back_button = QPushButton("뒤로")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: black;
                border: none;
                padding: 12px 28px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.back_button.setMinimumHeight(44)
        self.back_button.setMinimumWidth(90)
        self.back_button.clicked.connect(self.back_to_slot_select)
        self.apply_button = QPushButton("적용")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: black;
                border: none;
                padding: 12px 28px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.apply_button.setMinimumHeight(44)
        self.apply_button.setMinimumWidth(90)
        self.apply_button.clicked.connect(self.apply_hp_changes)
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.apply_button)
        # 버튼을 hp_manage_widget의 레이아웃에 바로 추가
        hp_layout.addLayout(button_layout)
        hp_layout.addSpacing(16)
        self.hp_manage_widget.setLayout(hp_layout)
        self.hp_manage_widget.hide()

        # 메인 레이아웃에 위젯 추가
        main_layout.addWidget(self.slot_select_widget)
        main_layout.addWidget(self.hp_manage_widget)
        content_widget.setLayout(main_layout)

        # 그래픽스 씬 설정
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
        """)
        button.clicked.connect(lambda: self.on_slot_selected(player_number, slot_type))
        return button

    def on_slot_selected(self, player_number, slot_type):
        self.selected_slot = (player_number, slot_type)
        self.selected_slot_label.setText(f"플레이어 {player_number} - {self.get_slot_kor(slot_type)}")
        # 부모에서 해당 슬롯의 추가체력/누적피해 불러오기
        parent = self.parent()
        bonus = 0
        damage = 0
        if parent and hasattr(parent, 'slot_hp_state'):
            state = parent.slot_hp_state.get((player_number, slot_type))
            if state:
                bonus = state.get('bonus', 0)
                damage = state.get('damage', 0)
        self.bonus_hp = bonus
        self.damage = damage
        self.bonus_hp_spin.setValue(bonus)
        self.damage_spin.setValue(damage)
        self.update_current_hp_display()
        self.slot_select_widget.hide()
        self.hp_manage_widget.show()

    def update_bonus_hp(self, value):
        self.bonus_hp = value
        self.update_current_hp_display()

    def update_damage(self, value):
        self.damage = value
        self.update_current_hp_display()

    def update_current_hp_display(self):
        if self.selected_slot:
            current_hp = 300 + self.bonus_hp - self.damage  # 기본 체력 300
            self.current_hp_label.setText(f"현재 체력: {current_hp}")

    def back_to_slot_select(self):
        self.hp_manage_widget.hide()
        self.slot_select_widget.show()

    def apply_hp_changes(self):
        if self.selected_slot:
            player_number, slot_type = self.selected_slot
            # SpinBox의 값을 직접 읽어옴
            bonus_hp = self.bonus_hp_spin.value()
            damage = self.damage_spin.value()
            current_hp = 300 + bonus_hp - damage
            parent = self.parent()
            if current_hp <= 0 and parent and hasattr(parent, 'message_window') and parent.message_window is not None:
                parent.message_window.setText('포켓몬이 기절했습니다!')
            self.hp_updated.emit(player_number, slot_type, bonus_hp, damage)
            self.accept()

    def get_slot_kor(self, slot_type):
        if slot_type == "active":
            return "배틀필드"
        elif slot_type.startswith("bench_"):
            return f"벤치 {int(slot_type.split('_')[1])+1}"
        return slot_type

    def set_ko_damage(self):
        # 현재체력 = 0이 되도록 누적피해를 자동으로 증가
        current_hp = 300 + self.bonus_hp - self.damage
        if current_hp > 0:
            self.damage = 300 + self.bonus_hp
            self.damage_spin.setValue(self.damage)
            self.update_current_hp_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 다이얼로그 크기 변경 시 버튼 위치 재조정 (왼쪽으로 10px 더 이동)
        if hasattr(self, 'button_widget'):
            self.button_widget.move((self.width() - self.button_widget.width()) // 2 - 10, self.height() - self.button_widget.height() - 10)

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = HPManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 