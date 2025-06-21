import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QWidget, QStyleOptionButton)
from PyQt5.QtGui import QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QSize, QTimer
import random

class CustomRotatedButton(QPushButton):
    def __init__(self, text, rotation=0, font_family=None, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.rotation = rotation
        if font_family:
            self.setFont(QFont(font_family, 10))

    def paintEvent(self, event):
        if self.rotation == 0:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        # 텍스트를 임시로 빈 문자열로 바꿔서 스타일만 그림
        original_text = opt.text
        opt.text = ""
        self.style().drawControl(self.style().CE_PushButton, opt, painter, self)
        # 회전된 텍스트만 직접 그림
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        painter.translate(-self.width() / 2, -self.height() / 2)
        painter.setFont(self.font())
        painter.setPen(self.palette().buttonText().color())
        painter.drawText(self.rect(), Qt.AlignCenter, original_text)
        painter.restore()

class RotatedNumberLabel(QLabel):
    def __init__(self, text, rotation=0, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.rotation = rotation

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        painter.translate(-self.width() / 2, -self.height() / 2)
        
        painter.setFont(self.font())
        painter.setPen(QColor("black"))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())
        painter.restore()

class RandomNumberDialog(QDialog):
    def __init__(self, parent=None, current_player=1):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowTitle("랜덤 숫자")
        self.setModal(True)
        self.current_player = current_player
        self.selected_number = 2  # 기본값 2
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_count = 0
        self.max_animation_count = 40  # 애니메이션 횟수 증가 (2초 동안)
        self.final_number = 0
        self.setup_ui()

    def setup_ui(self):
        # 다이얼로그 크기 설정
        self.setFixedSize(250, 300)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 버튼 레이아웃 (수직)
        button_layout = QVBoxLayout()
        
        # 업/다운 버튼
        up_button = CustomRotatedButton("▲", rotation=90 if self.current_player == 1 else -90)
        down_button = CustomRotatedButton("▼", rotation=90 if self.current_player == 1 else -90)
        
        # 버튼 스타일 설정
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: black;
                border: 2px solid #2c3e50;
                border-radius: 8px;
                padding: 5px;
                font-size: 16px;
                font-weight: bold;
                min-width: 40px;
                min-height: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
        """
        up_button.setStyleSheet(button_style)
        down_button.setStyleSheet(button_style)
        
        # 업/다운 버튼 클릭 이벤트
        up_button.clicked.connect(self.increase_number)
        down_button.clicked.connect(self.decrease_number)
        
        # 숫자 표시 레이블 (회전 적용)
        self.number_label = RotatedNumberLabel(str(self.selected_number), 
                                             rotation=90 if self.current_player == 1 else -90)
        self.number_label.setAlignment(Qt.AlignCenter)
        self.number_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: black;
            }
        """)
        
        # 확정 버튼
        confirm_button = CustomRotatedButton("확정", rotation=90 if self.current_player == 1 else -90)
        confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: black;
                border: 2px solid #2c3e50;
                border-radius: 8px;
                padding: 5px;
                font-size: 16px;
                font-weight: bold;
                min-width: 40px;
                min-height: 80px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #219a52;
            }
        """)
        confirm_button.clicked.connect(self.confirm_number)
        
        # 버튼 레이아웃에 추가
        if self.current_player == 1:
            # 플레이어 1: 업/다운/확정 순서
            button_layout.addWidget(up_button)
            button_layout.addWidget(down_button)
            button_layout.addWidget(confirm_button)
            button_layout.addStretch()
        else:
            # 플레이어 2: 업/다운/확정 순서 (플레이어 2 시점 기준)
            button_layout.addWidget(confirm_button)
            button_layout.addWidget(down_button)
            button_layout.addWidget(up_button)
            button_layout.addStretch()
        
        # 레이아웃에 위젯 추가
        if self.current_player == 1:
            main_layout.addLayout(button_layout)
            main_layout.addWidget(self.number_label)
        else:
            main_layout.addWidget(self.number_label)
            main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # 다이얼로그 위치 설정
        if self.parent():
            parent_rect = self.parent().geometry()
            if self.current_player == 1:
                self.move(parent_rect.x() + 10, parent_rect.y() + 300)
            else:
                self.move(parent_rect.x() + parent_rect.width() - 260, parent_rect.y() + 120)

    def increase_number(self):
        self.selected_number += 1
        self.number_label.setText(str(self.selected_number))

    def decrease_number(self):
        if self.selected_number > 2:  # 최소값 2
            self.selected_number -= 1
            self.number_label.setText(str(self.selected_number))

    def update_animation(self):
        if self.animation_count < self.max_animation_count:
            # 랜덤 숫자 생성 (1부터 선택된 숫자까지)
            random_num = random.randint(1, self.selected_number)
            self.parent().message_window.setText(f"<div style='text-align: center; font-size: 24px;'>{random_num}</div>")
            self.animation_count += 1
            
            # 점점 느려지도록 타이머 간격 조정
            if self.animation_count < 20:  # 처음 1초는 빠르게
                self.animation_timer.setInterval(25)  # 25ms 간격
            elif self.animation_count < 30:  # 다음 0.5초는 중간 속도
                self.animation_timer.setInterval(50)  # 50ms 간격
            else:  # 마지막 0.5초는 느리게
                self.animation_timer.setInterval(100)  # 100ms 간격
        else:
            self.animation_timer.stop()
            # 최종 결과 표시
            self.parent().message_window.setText(f"<div style='text-align: center; font-size: 24px;'>결과: {self.final_number}</div>")
            self.accept()

    def confirm_number(self):
        # 확정 버튼 클릭 시 애니메이션 시작
        self.final_number = random.randint(1, self.selected_number)
        self.animation_count = 0
        self.animation_timer.start(25)  # 처음에는 25ms 간격으로 시작

    def get_selected_number(self):
        return self.selected_number 