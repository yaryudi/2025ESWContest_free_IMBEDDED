#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QColorDialog, 
                             QSlider, QSpinBox, QFrame)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QFont

class Canvas(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QFrame { background-color: white; border: 2px solid #ccc; }")
        
        # 마우스 이벤트를 받을 수 있도록 설정
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 그림 그리기 관련 변수들
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(0, 0, 0)  # 검은색
        self.pen_width = 3
        
        # 캔버스 초기화
        self.canvas = QPixmap(800, 600)
        self.canvas.fill(Qt.white)
        
    def resizeEvent(self, event):
        """위젯 크기 변경 시 캔버스도 조정"""
        super().resizeEvent(event)
        new_canvas = QPixmap(self.size())
        new_canvas.fill(Qt.white)
        
        # 기존 캔버스 내용을 새 캔버스에 복사
        painter = QPainter(new_canvas)
        painter.drawPixmap(0, 0, self.canvas)
        painter.end()
        
        self.canvas = new_canvas
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)
        
    def mousePressEvent(self, event):
        print(f"DEBUG: Canvas 마우스 클릭 - 버튼: {event.button()}, 위치: {event.pos()}")
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            print(f"DEBUG: 그리기 시작 - 위치: {self.last_point}")
            
    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.LeftButton:
            print(f"DEBUG: Canvas 마우스 이동 - 위치: {event.pos()}")
            painter = QPainter(self.canvas)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            painter.end()  # QPainter 명시적 종료
            self.last_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            print(f"DEBUG: Canvas 마우스 릴리즈 - 위치: {event.pos()}")
            self.drawing = False
            

            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def clear_canvas(self):
        self.canvas.fill(Qt.white)
        self.update()
        
    def set_pen_color(self, color):
        self.pen_color = color
        
    def set_pen_width(self, width):
        self.pen_width = width

class SimplePaintGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("터치패드 드래그 테스트 - 간단한 그림판")
        self.setGeometry(100, 100, 1000, 700)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 제목
        title_label = QLabel("터치패드 드래그 테스트 - 그림판")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("QLabel { color: #333; padding: 10px; }")
        main_layout.addWidget(title_label)
        
        # 설명
        instruction_label = QLabel("터치패드로 드래그하여 그림을 그려보세요!")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("QLabel { color: #666; padding: 5px; }")
        main_layout.addWidget(instruction_label)
        
        # 컨트롤 패널
        control_layout = QHBoxLayout()
        
        # 색상 선택 버튼
        color_btn = QPushButton("색상 선택")
        color_btn.clicked.connect(self.choose_color)
        color_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        control_layout.addWidget(color_btn)
        
        # 색상 미리보기
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("background-color: black; border: 1px solid #ccc;")
        control_layout.addWidget(self.color_preview)
        
        # 선 굵기 조절
        control_layout.addWidget(QLabel("선 굵기:"))
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 20)
        self.width_slider.setValue(3)
        self.width_slider.valueChanged.connect(self.change_pen_width)
        control_layout.addWidget(self.width_slider)
        
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 20)
        self.width_spinbox.setValue(3)
        self.width_spinbox.valueChanged.connect(self.width_slider.setValue)
        self.width_slider.valueChanged.connect(self.width_spinbox.setValue)
        control_layout.addWidget(self.width_spinbox)
        
        # 지우개 버튼
        eraser_btn = QPushButton("지우개")
        eraser_btn.clicked.connect(self.use_eraser)
        eraser_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        control_layout.addWidget(eraser_btn)
        
        # 지우기 버튼
        clear_btn = QPushButton("전체 지우기")
        clear_btn.clicked.connect(self.clear_canvas)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        control_layout.addWidget(clear_btn)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # 캔버스
        self.canvas = Canvas()
        main_layout.addWidget(self.canvas)
        
        # 상태 표시
        status_layout = QHBoxLayout()
        self.status_label = QLabel("준비됨 - 터치패드로 드래그해보세요!")
        self.status_label.setStyleSheet("QLabel { color: #666; padding: 5px; }")
        status_layout.addWidget(self.status_label)
        
        # 드래그 테스트 정보
        self.drag_info = QLabel("드래그 거리: 0px")
        self.drag_info.setStyleSheet("QLabel { color: #666; padding: 5px; }")
        status_layout.addWidget(self.drag_info)
        
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
        
    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_pen_color(color)
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #ccc;")
            
    def change_pen_width(self, width):
        self.canvas.set_pen_width(width)
        
    def use_eraser(self):
        self.canvas.set_pen_color(Qt.white)
        self.color_preview.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        
    def clear_canvas(self):
        self.canvas.clear_canvas()
        
    def mousePressEvent(self, event):
        self.status_label.setText("마우스 클릭 감지됨")
        
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.status_label.setText("드래그 중...")
            
    def mouseReleaseEvent(self, event):
        self.status_label.setText("마우스 릴리즈됨")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 모던한 스타일 적용
    
    window = SimplePaintGame()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
