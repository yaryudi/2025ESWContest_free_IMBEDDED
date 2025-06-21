from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor

class PoisonButton(QPushButton):
    def __init__(self, text, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.setFixedSize(60, 60)
        self.is_active = True
        self.setStyleSheet('''
            QPushButton {
                background-color: #e84393;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d6306a;
            }
        ''')
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 보라색 원
        center = self.rect().center()
        radius = 16
        if self.is_active:
            painter.setBrush(QColor('#6c3483'))
        else:
            painter.setBrush(QColor('#b39ddb'))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius, radius)
        # 텍스트 직접 그리기
        painter.setPen(QColor(self.palette().buttonText().color()))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class BurnButton(QPushButton):
    def __init__(self, text, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.setFixedSize(60, 60)
        self.is_active = True
        self.setStyleSheet('''
            QPushButton {
                background-color: #f9ca7b;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f6b93b;
            }
        ''')
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 붉은 원
        center = self.rect().center()
        radius = 16
        if self.is_active:
            painter.setBrush(QColor('#e74c3c'))
        else:
            painter.setBrush(QColor('#ffb3b3'))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius, radius)
        # 텍스트 직접 그리기
        painter.setPen(QColor(self.palette().buttonText().color()))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class StatusManagementDialog(QDialog):
    slot_selected = pyqtSignal(int, str)
    status_selected = pyqtSignal(int, str, str)  # (player_number, slot_type, status)
    status_removed = pyqtSignal(int, str, str)  # (player_number, slot_type, status)
    def __init__(self, parent=None, current_player=1, slot_status_state=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("상태이상 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.selected_slot = None
        self.slot_status_state = slot_status_state if slot_status_state else {}
        # 수면/마비/혼란 버튼 저장용
        self.sleep_btn = None
        self.paralysis_btn = None
        self.confusion_btn = None
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
        description = QLabel("상태이상을 관리할 슬롯을 선택하세요")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                margin: 2px;
            }
        """)
        slot_layout.addStretch(1)
        slot_layout.addWidget(description, alignment=Qt.AlignHCenter)
        slot_layout.addStretch(1)
        player_layout = QHBoxLayout()
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(2)
        player_layout.addStretch(1)
        player1_widget = QWidget()
        player1_widget.setMinimumWidth(110)
        player1_widget.setMaximumWidth(110)
        player1_layout = QVBoxLayout(player1_widget)
        player1_layout.setAlignment(Qt.AlignCenter)
        player1_label = QLabel("플레이어 1")
        player1_label.setAlignment(Qt.AlignCenter)
        player1_layout.addWidget(player1_label)
        player1_battle_btn = self.create_slot_button("배틀\n필드", 1, "active")
        player1_battle_btn.setMinimumSize(60, 48)
        player1_battle_btn.setMaximumSize(80, 60)
        player1_battle_btn.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: black;
                border: none;
                padding: 2px;
                border-radius: 6px;
                font-size: 12px;
                min-width: 60px;
                min-height: 48px;
                max-width: 80px;
                max-height: 60px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        player1_layout.addWidget(player1_battle_btn, alignment=Qt.AlignCenter)
        player_layout.addWidget(player1_widget, 2)
        player_layout.addStretch(1)
        player2_widget = QWidget()
        player2_widget.setMinimumWidth(110)
        player2_widget.setMaximumWidth(110)
        player2_layout = QVBoxLayout(player2_widget)
        player2_layout.setAlignment(Qt.AlignCenter)
        player2_label = QLabel("플레이어 2")
        player2_label.setAlignment(Qt.AlignCenter)
        player2_layout.addWidget(player2_label)
        player2_battle_btn = self.create_slot_button("배틀\n필드", 2, "active")
        player2_battle_btn.setMinimumSize(60, 48)
        player2_battle_btn.setMaximumSize(80, 60)
        player2_battle_btn.setStyleSheet('''
            QPushButton {
                background-color: #3498db;
                color: black;
                border: none;
                padding: 2px;
                border-radius: 6px;
                font-size: 12px;
                min-width: 60px;
                min-height: 48px;
                max-width: 80px;
                max-height: 60px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        ''')
        player2_layout.addWidget(player2_battle_btn, alignment=Qt.AlignCenter)
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
        slot_layout.addStretch(1)
        self.slot_select_widget.setLayout(slot_layout)

        # 상태이상 선택 화면
        self.status_select_widget = QWidget()
        self.status_layout = QVBoxLayout()
        self.status_layout.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50; margin: 2px;")
        self.status_layout.addWidget(self.status_label)
        # 1행: 수면/마비/혼란
        row1 = QHBoxLayout()
        self.sleep_btn = QPushButton("수면")
        self.sleep_btn.setFixedSize(60, 60)
        self.sleep_btn.clicked.connect(lambda: self.on_status_button_clicked("sleep"))
        row1.addWidget(self.sleep_btn)
        self.paralysis_btn = QPushButton("마비")
        self.paralysis_btn.setFixedSize(60, 60)
        self.paralysis_btn.clicked.connect(lambda: self.on_status_button_clicked("paralysis"))
        row1.addWidget(self.paralysis_btn)
        self.confusion_btn = QPushButton("혼란")
        self.confusion_btn.setFixedSize(60, 60)
        self.confusion_btn.clicked.connect(lambda: self.on_status_button_clicked("confusion"))
        row1.addWidget(self.confusion_btn)
        self.status_layout.addLayout(row1)
        # 2행: 독/화상/이전
        row2 = QHBoxLayout()
        self.poison_btn = PoisonButton("독")
        self.poison_btn.clicked.connect(lambda: self.on_poison_burn_clicked("poison"))
        row2.addWidget(self.poison_btn)
        self.burn_btn = BurnButton("화상")
        self.burn_btn.clicked.connect(lambda: self.on_poison_burn_clicked("burn"))
        row2.addWidget(self.burn_btn)
        back_btn = QPushButton("이전")
        back_btn.setFixedSize(60, 60)
        back_btn.setStyleSheet('''
            QPushButton {
                background-color: #b2bec3;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #636e72;
                color: black;
            }
        ''')
        back_btn.clicked.connect(self.show_slot_select)
        row2.addWidget(back_btn)
        self.status_layout.addLayout(row2)
        self.status_select_widget.setLayout(self.status_layout)
        self.status_select_widget.hide()

        # 메인 레이아웃에 위젯 추가
        main_layout.addWidget(self.slot_select_widget)
        main_layout.addWidget(self.status_select_widget)
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

    def create_slot_button(self, text, player_number, slot_type):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.on_slot_selected(player_number, slot_type))
        return button

    def on_slot_selected(self, player_number, slot_type):
        self.selected_slot = (player_number, slot_type)
        slot_kor = "배틀필드" if slot_type == "active" else slot_type
        self.status_label.setText(f"플레이어 {player_number} - {slot_kor} 슬롯\n상태이상을 선택하세요")
        key = (player_number, slot_type)
        status_dict = self.slot_status_state.get(key, {})
        # 독/화상 버튼 스타일 갱신 (기존 코드)
        poison_normal = '''\
            QPushButton {
                background-color: #e84393;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d6306a;
            }
        '''
        poison_dim = '''\
            QPushButton {
                background-color: #f8bbd0;
                color: #b2bec3;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
        '''
        if status_dict.get('poison'):
            self.poison_btn.setStyleSheet(poison_dim)
            self.poison_btn.is_active = False
        else:
            self.poison_btn.setStyleSheet(poison_normal)
            self.poison_btn.is_active = True
        burn_normal = '''\
            QPushButton {
                background-color: #f9ca7b;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f6b93b;
            }
        '''
        burn_dim = '''\
            QPushButton {
                background-color: #ffe5b4;
                color: #b2bec3;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
        '''
        if status_dict.get('burn'):
            self.burn_btn.setStyleSheet(burn_dim)
            self.burn_btn.is_active = False
        else:
            self.burn_btn.setStyleSheet(burn_normal)
            self.burn_btn.is_active = True
        # 수면/마비/혼란 버튼 스타일 갱신
        sleep_normal = '''\
            QPushButton {
                background-color: #f1c40f;
                color: black;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
        '''
        sleep_dim = '''\
            QPushButton {
                background-color: #f7e6a2;
                color: #b2bec3;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
        '''
        # 수면
        if status_dict.get('sleep'):
            self.sleep_btn.setStyleSheet(sleep_dim)
        else:
            self.sleep_btn.setStyleSheet(sleep_normal)
        # 마비
        if status_dict.get('paralysis'):
            self.paralysis_btn.setStyleSheet(sleep_dim)
        else:
            self.paralysis_btn.setStyleSheet(sleep_normal)
        # 혼란
        if status_dict.get('confusion'):
            self.confusion_btn.setStyleSheet(sleep_dim)
        else:
            self.confusion_btn.setStyleSheet(sleep_normal)
        self.slot_select_widget.hide()
        self.status_select_widget.show()

    def show_slot_select(self):
        self.status_select_widget.hide()
        self.slot_select_widget.show()

    def on_status_button_clicked(self, status):
        if self.selected_slot:
            player_number, slot_type = self.selected_slot
            key = (player_number, slot_type)
            status_dict = self.slot_status_state.get(key, {})
            # 이미 해당 상태가 걸려있으면 해제
            if status_dict.get(status):
                self.status_removed.emit(player_number, slot_type, status)
                self.accept()
            else:
                self.emit_status_selected(status)

    def on_poison_burn_clicked(self, status):
        if self.selected_slot:
            player_number, slot_type = self.selected_slot
            key = (player_number, slot_type)
            status_dict = self.slot_status_state.get(key, {})
            if status_dict.get(status):
                # 이미 적용되어 있으면 해제
                self.status_removed.emit(player_number, slot_type, status)
                self.accept()
            else:
                self.emit_status_selected(status)

    def emit_status_selected(self, status):
        if self.selected_slot:
            player_number, slot_type = self.selected_slot
            self.status_selected.emit(player_number, slot_type, status)
            self.accept()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = StatusManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 