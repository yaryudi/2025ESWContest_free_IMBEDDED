#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap

class PromotionDialog(QDialog):
    """폰 승진을 위한 다이얼로그"""
    
    pieceSelected = pyqtSignal(str)  # 선택된 기물 시그널
    
    def __init__(self, player_color, parent=None):
        super().__init__(parent)
        self.player_color = player_color
        self.selected_piece = None
        self.setup_ui()
        
    def setup_ui(self):
        """UI 설정"""
        self.setWindowTitle("폰 승진")
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 제목 라벨
        title_label = QLabel("승진할 기물을 선택하세요:")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 기물 버튼들
        pieces = [
            ('Q', 'Queen', '퀸'),
            ('R', 'Rook', '룩'),
            ('B', 'Bishop', '비숍'),
            ('N', 'Knight', '나이트')
        ]
        
        for piece_code, piece_name, piece_korean in pieces:
            button = QPushButton()
            button.setFixedSize(80, 80)
            button.setToolTip(f"{piece_name} ({piece_korean})")
            
            # 기물 아이콘 설정
            icon_path = f'resources/img/pieces/{self.player_color}{piece_code}.svg'
            try:
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QPixmap(icon_path).size())
            except:
                # 아이콘이 없으면 텍스트로 표시
                button.setText(piece_korean)
                button.setStyleSheet("""
                    QPushButton {
                        font-size: 14px;
                        font-weight: bold;
                        background-color: #f0f0f0;
                        border: 2px solid #ccc;
                        border-radius: 10px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                        border-color: #999;
                    }
                    QPushButton:pressed {
                        background-color: #d0d0d0;
                    }
                """)
            
            # 버튼 클릭 이벤트 연결
            button.clicked.connect(lambda checked, code=piece_code: self.select_piece(code))
            
            button_layout.addWidget(button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        # 창을 화면 중앙에 배치
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        
    def select_piece(self, piece_code):
        """기물 선택"""
        self.selected_piece = piece_code
        self.pieceSelected.emit(piece_code)
        self.accept()
        
    def get_selected_piece(self):
        """선택된 기물 반환"""
        return self.selected_piece 