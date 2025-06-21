from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QGridLayout, QScrollArea, QWidget, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QSpinBox, QStackedLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont, QTransform, QPainter, QPixmap
import os

class RotatedWidget(QWidget):
    def __init__(self, parent=None, rotation=0):
        super().__init__(parent)
        self.rotation = rotation
        # 회전에 맞춰 크기 조정
        if abs(rotation) == 90:
            self.setMinimumSize(400, 300)  # 가로/세로 교체
        else:
            self.setMinimumSize(300, 400)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 배경 지우기
        painter.eraseRect(self.rect())
        
        # 회전 변환
        transform = QTransform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.rotate(self.rotation)
        transform.translate(-self.width() / 2, -self.height() / 2)
        painter.setTransform(transform)
        
        # 배경 그리기
        painter.fillRect(self.rect(), self.palette().window())
        
        # 자식 위젯 그리기
        for child in self.children():
            if isinstance(child, QWidget):
                child.render(painter, child.pos())

class EnergyManagementDialog(QDialog):
    # 슬롯 선택 시그널 정의
    slot_selected = pyqtSignal(int, str)  # (player_number, slot_type)
    energy_selected = pyqtSignal(int, str, dict)  # player, slot_type, {energy_type: count}
    
    def __init__(self, parent=None, current_player=1, slot_energy_state=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("에너지 관리")
        self.setModal(True)
        self.current_player = current_player
        self.rotation = 90 if current_player == 1 else -90
        self.selected_slot = None
        self.energy_images = {}
        self.energy_counts = {}
        self.slot_energy_state = slot_energy_state if slot_energy_state else {}  # {(player, slot_type): {energy_type: count}}
        self.load_energy_images()
        self.init_ui()

    def load_energy_images(self):
        energy_dir = os.path.join(os.path.dirname(__file__), '에너지_이미지')
        for file in os.listdir(energy_dir):
            if file.endswith('.jpg'):
                energy_type = file.replace('_energy.jpg', '')
                self.energy_images[energy_type] = os.path.join(energy_dir, file)

    def init_ui(self):
        self.content_widget = QWidget()
        self.content_widget.setFixedSize(250, 300)
        self.content_widget.setContentsMargins(0, 0, 0, 0)
        main_layout = QVBoxLayout(self.content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.slot_select_widget = self.create_slot_select_widget()
        self.energy_select_widget = self.create_energy_select_widget()
        main_layout.addWidget(self.slot_select_widget)
        main_layout.addWidget(self.energy_select_widget)
        self.energy_select_widget.hide()

        scene = QGraphicsScene(self)
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(self.content_widget)
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

    def create_slot_select_widget(self):
        widget = QWidget()
        slot_layout = QVBoxLayout()
        description = QLabel("에너지를 관리할 슬롯을 선택하세요")
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
        widget.setLayout(slot_layout)
        return widget

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
        button_style = '''
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
        '''
        for i in range(grid.count()):
            widget = grid.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                widget.setStyleSheet(button_style)

    def create_slot_button(self, text, player_number, slot_type):
        button = QPushButton(text)
        button.clicked.connect(lambda: self.on_slot_selected(player_number, slot_type))
        return button

    def on_slot_selected(self, player_number, slot_type):
        self.selected_slot = (player_number, slot_type)
        # 해당 슬롯의 에너지 상태를 불러와서 반영
        key = (player_number, slot_type)
        current = self.slot_energy_state.get(key, {})
        self.energy_counts = {k: current.get(k, 0) for k in self.energy_images.keys()}
        self.update_energy_select_widget()
        self.slot_select_widget.hide()
        self.energy_select_widget.show()

    def create_energy_select_widget(self):
        widget = QWidget()
        # 에너지 타입 순서 조정 (Water가 Other보다 앞에 오도록)
        energy_types = list(self.energy_images.keys())
        if 'Other' in energy_types and 'Water' in energy_types:
            idx_other = energy_types.index('Other')
            idx_water = energy_types.index('Water')
            if idx_other < idx_water:
                energy_types[idx_other], energy_types[idx_water] = energy_types[idx_water], energy_types[idx_other]
        # 한글 이름 매핑
        energy_kor = {
            'Darkness': '악',
            'Fairy': '페어리',
            'Fighting': '격투',
            'Fire': '불꽃',
            'Grass': '풀',
            'Lightning': '번개',
            'Metal': '강철',
            'Psychic': '에스퍼',
            'Colorless': '무색',
            'Water': '물',
            'Other': '기타',
        }
        # 2열 그리드 생성 (최소화된 크기)
        self.energy_grid = QGridLayout()
        self.energy_grid.setContentsMargins(2, 2, 2, 2)
        self.energy_grid.setSpacing(6)  # 간격 늘림
        self.energy_widgets = {}
        n = len(energy_types)
        for idx, energy_type in enumerate(energy_types):
            row = idx // 2
            col = idx % 2
            energy_widget = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            image_label = QLabel()
            pixmap = QPixmap(self.energy_images[energy_type])
            image_label.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setFixedSize(28, 28)
            layout.addWidget(image_label)
            name_label = QLabel(energy_kor.get(energy_type, energy_type))
            name_label.setMinimumWidth(22)
            name_label.setMaximumWidth(32)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(False)
            name_label.setStyleSheet("font-size: 13px; padding: 0px;")
            layout.addWidget(name_label)
            spinbox = QSpinBox()
            spinbox.setRange(0, 10)
            spinbox.setFixedWidth(38)
            spinbox.setFixedHeight(32)
            spinbox.setStyleSheet("font-size: 15px; padding: 0px; QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 18px; height: 16px; }")
            spinbox.setValue(0)
            spinbox.valueChanged.connect(lambda value, et=energy_type: self.update_energy_count(et, value))
            spinbox.lineEdit().setReadOnly(True)
            spinbox.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(spinbox)
            energy_widget.setLayout(layout)
            energy_widget.setFixedHeight(36)
            self.energy_grid.addWidget(energy_widget, row, col)
            self.energy_widgets[energy_type] = spinbox
        # 버튼을 마지막 빈칸(2열 기준)에 배치
        btn_widget = QWidget()
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(2)
        confirm_btn = QPushButton("확인")
        confirm_btn.setFixedHeight(22)
        confirm_btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        confirm_btn.clicked.connect(self.on_confirm_energy)
        btn_layout.addWidget(confirm_btn)
        back_btn = QPushButton("뒤로")
        back_btn.setFixedHeight(22)
        back_btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        back_btn.clicked.connect(self.show_slot_select)
        btn_layout.addWidget(back_btn)
        btn_widget.setLayout(btn_layout)
        btn_widget.setFixedHeight(22)
        # 마지막 칸 위치 계산
        last_idx = n
        last_row = last_idx // 2
        last_col = last_idx % 2
        self.energy_grid.addWidget(btn_widget, last_row, last_col)
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        main_layout.addLayout(self.energy_grid)
        widget.setLayout(main_layout)
        return widget

    def update_energy_select_widget(self):
        for energy_type, spinbox in self.energy_widgets.items():
            spinbox.setValue(self.energy_counts.get(energy_type, 0))

    def update_energy_count(self, energy_type, count):
        self.energy_counts[energy_type] = count

    def on_confirm_energy(self):
        if self.selected_slot:
            player_number, slot_type = self.selected_slot
            selected = {k: v for k, v in self.energy_counts.items() if v > 0}
            # 슬롯별 상태 갱신
            self.slot_energy_state[(player_number, slot_type)] = dict(self.energy_counts)
            self.energy_selected.emit(player_number, slot_type, dict(self.energy_counts))
        self.accept()

    # 뒤로 버튼에서 슬롯 선택창으로 돌아가기
    def show_slot_select(self):
        self.energy_select_widget.hide()
        self.slot_select_widget.show()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dlg = EnergyManagementDialog(current_player=1)
    dlg.show()
    sys.exit(app.exec_()) 