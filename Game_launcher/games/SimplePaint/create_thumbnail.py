#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QFont
from PyQt5.QtCore import Qt
import sys

def create_thumbnail():
    # 200x200 크기의 썸네일 생성
    pixmap = QPixmap(200, 200)
    pixmap.fill(QColor(255, 255, 255))  # 흰색 배경
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 제목
    painter.setFont(QFont("Arial", 16, QFont.Bold))
    painter.setPen(QColor(0, 0, 0))
    painter.drawText(10, 30, "그림판")
    
    # 간단한 그림 예시
    painter.setPen(QPen(QColor(255, 0, 0), 3))  # 빨간색 선
    painter.drawLine(30, 60, 80, 100)
    painter.drawLine(80, 100, 130, 80)
    
    painter.setPen(QPen(QColor(0, 255, 0), 3))  # 초록색 선
    painter.drawLine(50, 120, 100, 140)
    painter.drawLine(100, 140, 150, 120)
    
    painter.setPen(QPen(QColor(0, 0, 255), 3))  # 파란색 선
    painter.drawLine(70, 160, 120, 180)
    
    # 원 그리기
    painter.setPen(QPen(QColor(255, 165, 0), 2))  # 주황색
    painter.drawEllipse(140, 50, 30, 30)
    
    painter.end()
    
    # 파일로 저장
    pixmap.save("thumbnail.png")
    print("썸네일 이미지가 생성되었습니다: thumbnail.png")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    create_thumbnail()
    app.quit()
