from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QGroupBox, QCheckBox, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt
from dnd_character.spellcasting import spells_for_class_level

class SpellSelectionDialog(QDialog):
    def __init__(self, parent=None, character_class=None):
        super().__init__(parent)
        self.character_class = character_class
        self.selected_spells = []
        self.checkboxes = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("스펠 선택")
        self.setMinimumWidth(400)
        layout = QVBoxLayout()
        label = QLabel("사용 가능한 스펠을 선택하세요. (0, 1레벨)")
        layout.addWidget(label)
        # 스크롤 영역
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        # 0, 1레벨별로 그룹박스 생성
        for level in [0, 1]:
            spells = []
            if self.character_class:
                try:
                    spells = list(spells_for_class_level(self.character_class.lower(), level))
                except Exception:
                    spells = []
            if not spells:
                continue
            group = QGroupBox(f"{level}레벨 주문" + (" (Cantrips)" if level == 0 else ""))
            group_layout = QVBoxLayout()
            for spell in sorted(spells):
                checkbox = QCheckBox(spell)
                checkbox.stateChanged.connect(self.update_selected_spells)
                self.checkboxes.append(checkbox)
                group_layout.addWidget(checkbox)
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        # 버튼
        button_layout = QHBoxLayout()
        select_button = QPushButton("선택")
        select_button.clicked.connect(self.accept)
        button_layout.addWidget(select_button)
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)
    def update_selected_spells(self):
        self.selected_spells = [cb.text() for cb in self.checkboxes if cb.isChecked()]
    def get_selected_spells(self):
        return self.selected_spells 