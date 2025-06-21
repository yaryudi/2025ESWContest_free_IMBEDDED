"""
텍사스 홀덤 포커 게임 메인 모듈
게임 로직, UI, 카드 처리, 베팅 시스템 등을 구현합니다.
"""

import random
import math
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QInputDialog, QSpinBox, QVBoxLayout, QHBoxLayout, QDialog, QGraphicsDropShadowEffect, QGraphicsScene, QGraphicsView, QGraphicsProxyWidget
from PyQt5.QtGui import QFont, QColor, QTransform, QPainter, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QPoint
from hand_evaluator import HandEvaluator
from test import CardDetector
import cv2
import numpy as np
from ultralytics import YOLO
import time
from picamera2 import Picamera2


class RaiseDialog(QDialog):
    """레이즈 금액을 입력받는 다이얼로그 창"""
    
    def __init__(self, parent, min_amount, max_amount, current_bet):
        super().__init__(parent)
        self.setWindowTitle("레이즈 금액 설정")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()
        
        # 현재 베팅 정보 표시
        current_bet_label = QLabel(f"현재 베팅: {current_bet}", self)
        current_bet_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(current_bet_label)
        
        # 최소 레이즈 금액 표시
        min_raise_label = QLabel(f"최소 레이즈: {min_amount}", self)
        min_raise_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(min_raise_label)

        # 금액 입력을 위한 스핀박스
        self.spin = QSpinBox()
        self.spin.setMinimum(min_amount)
        self.spin.setMaximum(max_amount)
        self.spin.setSingleStep(100)
        self.spin.setValue(min_amount)

        # 확인/취소 버튼
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("확인")
        cancel_btn = QPushButton("취소")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout.addWidget(self.spin)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_value(self):
        """입력된 레이즈 금액을 반환합니다."""
        return self.spin.value()

    def set_position(self, x, y):
        """다이얼로그의 위치를 설정합니다."""
        self.move(x, y)


class PokerGame(QWidget):
    """텍사스 홀덤 포커 게임의 메인 클래스"""
    
    def __init__(self, num_players):
        super().__init__()
        # 기본 UI 설정
        self.card_width = 80  # 카드 슬롯 크기
        self.card_height = 112
        self.setWindowTitle("텍사스 홀덤 포커")
        self.setMinimumSize(1280, 720)
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #0B6623;")

        # 게임 상태 변수 초기화
        self.num_players = num_players
        self.starting_chips = 10000
        self.chips = [self.starting_chips] * num_players
        self.player_hands = [[] for _ in range(num_players)]  # 빈 리스트로 초기화
        self.community_cards = [None] * 5  # None으로 초기화
        self.player_labels = []
        self.bet_labels = []
        self.chip_labels = []
        self.total_bet_labels = []
        self.action_buttons = []
        
        # 베팅 관련 변수들
        self.player_bets = [0] * num_players
        self.player_total_bets = [0] * num_players
        self.folded_players = [False] * num_players
        self.all_in_players = [False] * num_players
        self.community_stage = 0
        self.current_round = "preflop"
        self.max_bet = 0
        self.min_raise = 0
        self.last_raiser = -1
        self.acted_players = set()
        self.small_blind = 100
        self.big_blind = 200
        self.total_pot = 0
        self.current_round_pot = 0
        self.accumulated_pot = 0  # 누적된 총 팟

        # 포지션 설정
        self.sb_index = 0  # SB는 0번 플레이어
        self.bb_index = 1  # BB는 1번 플레이어
        self.utg_index = 2  # UTG는 2번 플레이어
        
        # 현재 턴 설정
        self.current_turn = self.utg_index if self.num_players > 3 else self.sb_index

        # 팟 효과 관련 변수
        self.pot_animation_timer = None
        self.pot_animation_offset = 0
        self.pot_animation_direction = 1
        self.pot_animation_intensity = 0

        # 쇼다운 효과 관련 변수
        self.showdown_effect_timer = None
        self.showdown_alpha = 0
        self.showdown_overlay = None
        self.showdown_text = None

        # 쇼다운 상태 변수
        self.is_showdown = False

        # 게임 초기화 및 UI 설정
        self.init_game()
        self.setup_players_ui()
        self.init_round()

        # 창을 화면 중앙에 배치
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        # 카드 인식 기능 초기화
        self.card_detector = CardDetector()
        
        # 화면 보기 버튼 추가
        self.view_button = QPushButton("화면 보기", self)
        self.view_button.setStyleSheet("""
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
        self.view_button.clicked.connect(self.show_camera_view)
        
        # 버튼 위치 설정 (우측 상단)
        self.view_button.setGeometry(self.width() - 120, 20, 100, 40)

        # 카메라 뷰를 위한 라벨 추가
        self.camera_view_label = QLabel(self)
        self.camera_view_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #444;
            border-radius: 4px;
        """)
        self.camera_view_label.hide()

    def init_game(self):
        """게임 보드 및 주요 UI 컴포넌트를 초기화합니다."""
        self.board = QWidget(self)
        self.board.setGeometry(0, 0, self.width(), self.height())  # 창 전체 사용
        self.board.setStyleSheet("background-color: #0B6623;")

        # 쇼다운 오버레이 추가
        self.showdown_overlay = QWidget(self.board)
        self.showdown_overlay.setGeometry(0, 0, self.width(), self.height())
        self.showdown_overlay.setStyleSheet("background-color: rgba(255, 215, 0, 0);")
        self.showdown_overlay.hide()

        # 쇼다운 텍스트를 위한 그래픽스 씬과 뷰 설정
        self.showdown_scene = QGraphicsScene(self.showdown_overlay)
        self.showdown_view = QGraphicsView(self.showdown_scene, self.showdown_overlay)
        self.showdown_view.setGeometry(0, 0, self.width(), self.height())
        self.showdown_view.setStyleSheet("background: transparent; border: none;")
        self.showdown_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.showdown_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.showdown_view.setRenderHint(QPainter.Antialiasing)

        # 쇼다운 텍스트 추가
        self.showdown_text = QLabel("Show Down !!!")
        self.showdown_text.setStyleSheet("""
            color: rgba(255, 0, 0, 0);
            font-weight: bold;
            font-family: 'Arial Black';
            background: transparent;
        """)
        self.showdown_text.setAlignment(Qt.AlignCenter)
        self.showdown_proxy = self.showdown_scene.addWidget(self.showdown_text)
        self.showdown_proxy.hide()

        # 커뮤니티 카드 영역 컨테이너
        self.community_container = QWidget(self.board)
        self.community_container.setStyleSheet("""
            background-color: rgba(80, 80, 100, 0.92);
            border: 1px solid #444a54;
            border-radius: 5px;
            padding: 0px;
        """)
        
        # 커뮤니티 카드 왼쪽에 덱(카드 뭉치) 영역 추가
        self.deck_label = QLabel(self.board)
        self.deck_label.setFixedSize(self.card_width, self.card_height)
        self.deck_label.setStyleSheet("""
            background-color: #2d2d3a;
            border: 2px solid #222b36;
            border-radius: 8px;
            color: white;
            font-size: 22px;
            font-weight: bold;
        """)
        self.deck_label.setAlignment(Qt.AlignCenter)
        self.deck_label.setText("DECK")
        
        # 카드 크기 설정
        self.card_width = 80  # 134에서 80으로 축소
        self.card_height = 112  # 192에서 112로 축소
        card_spacing = 10  # 20에서 10으로 축소
        
        # 컨테이너 크기 계산
        container_width = (self.card_width * 5) + (card_spacing * 4) + 20  # 카드 5장 + 간격 4개 + 좌우 패딩
        container_height = self.card_height + 20  # 카드 높이 + 상하 패딩
        self.community_container.setFixedSize(container_width, container_height)
        
        # 커뮤니티 카드 라벨들 생성
        self.community_labels = []
        for i in range(5):
            card_label = QLabel(self.community_container)
            card_label.setStyleSheet("""
                background-color: white;
                border: 1px solid #444;
                border-radius: 3px;
            """)
            card_label.setFixedSize(self.card_width, self.card_height)
            card_label.setAlignment(Qt.AlignCenter)
            # 카드 위치 계산
            x = 10 + (i * (self.card_width + card_spacing))
            y = 10
            card_label.move(x, y)
            self.community_labels.append(card_label)

        # 메시지 라벨을 위한 그래픽스 씬과 뷰 설정
        self.message_scene = QGraphicsScene(self.board)
        self.message_view = QGraphicsView(self.message_scene, self.board)
        self.message_view.setStyleSheet("background: transparent; border: none;")
        self.message_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.message_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.message_view.setRenderHint(QPainter.Antialiasing)
        
        # 새로운 메시지 라벨 생성 (독립적인 위젯으로)
        self.rotated_message_label = QLabel()
        self.rotated_message_label.setStyleSheet("""
            background-color: rgba(51, 51, 51, 0.9);
            color: white;
            border-radius: 12px;
            padding: 10px;
            border: 2px solid #666;
        """)
        self.rotated_message_label.setFont(QFont("Arial", 12))
        self.rotated_message_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.rotated_message_label.setWordWrap(True)
        self.rotated_message_label.setFixedSize(250, 180)
        self.rotated_message_label.setContentsMargins(10, 10, 10, 10)
        self.rotated_message_label.setMinimumHeight(120)
        
        # 회전된 메시지 라벨을 그래픽스 씬에 추가
        self.message_proxy = self.message_scene.addWidget(self.rotated_message_label)
        self.message_proxy.setRotation(90)  # 90도 회전
        self.message_proxy.setTransformOriginPoint(self.rotated_message_label.width()/2, self.rotated_message_label.height()/2)

        self.next_stage_button = QPushButton("다음 카드 공개", self.board)
        self.next_stage_button.setStyleSheet("""
            background-color: darkred;
            color: white;
            font-size: 16px;
            padding: 8px;
            border-radius: 8px;
            border: 2px solid #800000;
        """)
        self.next_stage_button.setFixedSize(200, 60)  # 버튼 크기도 키움
        self.next_stage_button.clicked.connect(self.show_next_stage)
        self.next_stage_button.setEnabled(False)

        self.pot_label = QLabel("총 팟: 0", self.board)
        self.pot_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.pot_label.setStyleSheet("color: gold; background-color: #222; border: 2px solid #555; border-radius: 10px; padding: 8px;")
        self.pot_label.setFixedSize(340, 70)
        self.update_pot_position()

    def update_pot_position(self):
        # 팟 라벨을 화면 중앙에 위치
        center_x = self.width() // 2
        center_y = self.height() // 2
        # 커뮤니티 카드 컨테이너의 위치 계산
        offset_y = 75
        community_y = center_y - 120 + offset_y
        community_bottom = community_y + self.community_container.height()
        # 총 팟 라벨을 커뮤니티 카드 컨테이너 아래쪽에 일정한 간격(160px)을 두고 배치
        pot_y = community_bottom + 160
        # 팟 라벨 위치 설정
        self.pot_label.move(center_x - self.pot_label.width() // 2, pot_y)
        # 원래 위치 저장 (팟 애니메이션용)
        self.original_pot_x = center_x - self.pot_label.width() // 2
        self.original_pot_y = pot_y

    def setup_players_ui(self):
        # 카드 크기 및 컨테이너 크기 재설정 (모든 위치 계산에 사용)
        self.card_width = 80
        self.card_height = 112
        card_spacing = 10
        container_padding = 4 # 최소 패딩
        card_area_width = self.card_width * 2 + card_spacing + container_padding * 2
        card_area_height = self.card_height + container_padding * 2
        player_container_width = card_area_width + 8  # 여유
        player_container_height = card_area_height + 8

        # UI 위치 계산에 필요한 변수 미리 선언
        name_w = card_area_width
        btn_w = 220

        # 커뮤니티 카드 컨테이너 중심 좌표 계산 (카드 슬롯용)
        center_x = self.width() // 2
        center_y = self.height() // 2
        comm_width = self.community_container.width()
        comm_height = self.community_container.height()
        # 커뮤니티 창을 창의 세로 중앙에 배치
        community_y = (self.height() - comm_height) // 2
        community_x = center_x - comm_width // 2

        gap_x = 50
        gap_y = 0  # 커뮤니티 창과 거의 붙임
        offset_x = 100

        top_gap_x = 60  # 위쪽 슬롯 간격
        bottom_gap_x = -20  # 아래쪽 슬롯 간격
        slot_offset_y = 21
        slot_positions = [
            # 위쪽
            (community_x - top_gap_x, community_y - card_area_height - gap_y + slot_offset_y),  # P1: 왼쪽
            (community_x + comm_width//2 - card_area_width//2, community_y - card_area_height - gap_y + slot_offset_y),  # P2: 중앙
            (community_x + comm_width - card_area_width + top_gap_x, community_y - card_area_height - gap_y + slot_offset_y),  # P3: 오른쪽
            # 아래쪽
            (community_x + comm_width - card_area_width + bottom_gap_x, community_y + comm_height + gap_y + slot_offset_y),  # P4: 오른쪽(아래)
            (community_x - bottom_gap_x, community_y + comm_height + gap_y + slot_offset_y),  # P5: 왼쪽(아래)
        ]
        slot_indices = [0, 1, 2, 3, 4]

        # 플레이어 UI 위치 (이름+버튼+여유 고려, 창 끝에 붙임)
        ui_positions = [
            (10, 20),  # Player 1 (왼쪽 위)
            (self.width()//2 - name_w//2 - 100, 20),  # Player 2 (더 왼쪽)
            (self.width() - name_w - btn_w - 10, 20),  # Player 3 (오른쪽 위)
            (self.width() - name_w - btn_w - 10, self.height() - card_area_height - 100 + 60),  # Player 4 (아래로 60 이동)
            (10, self.height() - card_area_height - 100 + 60),  # Player 5 (아래로 60 이동)
        ]
        ui_indices = [0, 1, 2, 3, 4]

        self.player_labels.clear()
        self.name_labels = []
        self.bet_labels.clear()
        self.chip_labels.clear()
        self.total_bet_labels.clear()
        self.action_buttons.clear()

        for i in range(self.num_players):
            # 카드 슬롯 컨테이너는 중앙 레이아웃에
            slot_x, slot_y = slot_positions[slot_indices[i]]
            card_container = QWidget(self)
            card_container.setGeometry(slot_x, slot_y, card_area_width, card_area_height)
            card_container.setStyleSheet("""
                background-color: rgba(80, 80, 100, 0.92);
                border: 1px solid #444a54;
                border-radius: 5px;
                padding: 0px;
            """)
            card_labels = []
            for j in range(2):
                card_label = QLabel(card_container)
                card_label.setStyleSheet("""
                    background-color: white;
                    border: 1px solid #444;
                    border-radius: 3px;
                """)
                card_label.setFixedSize(self.card_width, self.card_height)
                card_label.setAlignment(Qt.AlignCenter)
                card_label.move(container_padding + (j * (self.card_width + card_spacing)), container_padding)
                card_labels.append(card_label)
            self.player_labels.append(card_labels)

            # 나머지 UI(이름, 베팅, 버튼 등)는 기존 위치에
            x, y = ui_positions[ui_indices[i]]
            position = ""
            if i == self.sb_index:
                position = " (SB)"
            elif i == self.bb_index:
                position = " (BB)"
            elif i == self.utg_index and self.num_players > 3:
                position = " (UTG)"

            # 플레이어 이름
            name_label = QLabel(f"Player {i + 1}{position}", self)
            name_label.setStyleSheet("""
                background-color: skyblue;
                border: 2px solid navy;
                border-radius: 6px;
                padding: 5px;
                color: black;
            """)
            name_label.setFont(QFont("Arial", 12))
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setFixedSize(card_area_width, 30)
            name_label.move(x, y)
            self.name_labels.append(name_label)

            # 베팅 정보 컨테이너
            bet_container = QWidget(self)
            bet_container.setGeometry(x, y + 60, 300, 75)
            bet_container.setStyleSheet("background-color: transparent;")

            bet_label = QLabel("베팅: 0", bet_container)
            bet_label.setStyleSheet("color: white;")
            bet_label.setFont(QFont("Arial", 12))
            bet_label.setFixedSize(300, 25)
            bet_label.move(0, 0)
            self.bet_labels.append(bet_label)

            total_bet_label = QLabel("누적: 0", bet_container)
            total_bet_label.setStyleSheet("color: #FFD700;")
            total_bet_label.setFont(QFont("Arial", 12))
            total_bet_label.setFixedSize(300, 25)
            total_bet_label.move(0, 25)
            self.total_bet_labels.append(total_bet_label)

            chip_label = QLabel(f"칩: {self.chips[i]}", bet_container)
            chip_label.setStyleSheet("color: gold;")
            chip_label.setFont(QFont("Arial", 12))
            chip_label.setFixedSize(300, 25)
            chip_label.move(0, 50)
            self.chip_labels.append(chip_label)

            # 액션 버튼들
            buttons = []
            actions = [("콜", self.call), ("체크", self.check), ("폴드", self.fold), ("레이즈", self.raise_bet), ("올인", self.all_in)]
            for j, (text, func) in enumerate(actions):
                btn = QPushButton(text, self)
                btn.setStyleSheet("""
                    font-size: 12px;
                    padding: 5px;
                    background-color: #444;
                    color: white;
                    border-radius: 5px;
                """)
                btn.setFixedSize(100, 35)
                btn_x = x + name_label.width() - 20 + (j % 2) * 105  # 이름/베팅정보 바로 오른쪽에 버튼
                btn_y = y + (j // 2) * 40
                btn.move(btn_x, btn_y)
                btn.clicked.connect(lambda _, idx=i, f=func: f(idx))
                buttons.append(btn)
            self.action_buttons.append(buttons)

    def resizeEvent(self, event):
        self.board.setGeometry(0, 0, self.width(), self.height())  # 창 전체 사용
        if self.showdown_overlay:
            self.showdown_overlay.setGeometry(0, 0, self.width(), self.height())
            if self.showdown_view:
                self.showdown_view.setGeometry(0, 0, self.width(), self.height())
                self.showdown_scene.setSceneRect(0, 0, self.width(), self.height())
            self.update_showdown_text_positions()

        center_x = self.width() // 2
        center_y = self.height() // 2

        # 커뮤니티 카드 컨테이너 위치 조정
        offset_y = 75
        community_y = center_y - 120 + offset_y  # 커뮤니티 창의 위치를 약간 위로 조정
        community_x = center_x - self.community_container.width() // 2
        self.community_container.move(community_x, community_y)
        
        # 덱(카드 뭉치) 위치 조정 (커뮤니티 카드 왼쪽)
        offset_x = 80
        deck_x = community_x - self.deck_label.width() - 150 + offset_x
        deck_y = community_y + (self.community_container.height() - self.deck_label.height()) // 2
        self.deck_label.move(deck_x, deck_y)
        
        # 다음 단계 버튼을 덱 왼쪽에 배치
        button_x = deck_x - self.next_stage_button.width() - 30
        button_y = deck_y + (self.deck_label.height() - self.next_stage_button.height()) // 2
        self.next_stage_button.move(button_x, button_y)
        
        # 화면보기 버튼을 '다음 카드 공개' 버튼 위로 이동
        view_btn_x = button_x
        view_btn_y = button_y - self.view_button.height() - 10  # 10px 여유
        self.view_button.setGeometry(view_btn_x, view_btn_y, 100, 40)

        # 메시지 라벨을 오른쪽 가운데에 배치
        message_x = community_x + self.community_container.width() + 50  # 커뮤니티 카드 오른쪽에 50px 간격
        # 회전된 메시지 라벨을 위한 여백 추가
        padding = 40
        label_w = self.rotated_message_label.width()
        label_h = self.rotated_message_label.height()
        view_size = max(label_w, label_h) + padding * 2

        # 메시지 뷰의 중앙이 커뮤니티 카드 컨테이너의 중앙에 오도록
        community_center_y = community_y + self.community_container.height() // 2
        message_y = community_center_y - (view_size // 2)

        self.message_view.setGeometry(message_x, message_y, view_size, view_size)
        self.message_scene.setSceneRect(0, 0, view_size, view_size)
        # 라벨을 QGraphicsView의 중앙에 배치
        label_x = (view_size - label_w) // 2
        label_y = (view_size - label_h) // 2
        self.message_proxy.setPos(label_x, label_y)
        self.message_proxy.setTransformOriginPoint(label_w / 2, label_h / 2)

        self.update_pot_position()
        self.reposition_players()
        super().resizeEvent(event)

    def reposition_players(self):
        # 카드 크기 및 컨테이너 크기 재설정 (모든 위치 계산에 사용)
        self.card_width = 80
        self.card_height = 112
        card_spacing = 10
        container_padding = 4
        card_area_width = self.card_width * 2 + card_spacing + container_padding * 2
        card_area_height = self.card_height + container_padding * 2
        player_container_width = card_area_width + 8
        player_container_height = card_area_height + 8

        # UI 위치 계산에 필요한 변수 미리 선언
        name_w = card_area_width
        btn_w = 220

        # 카드 슬롯은 중앙 레이아웃, 나머지 UI는 기존 위치
        center_x = self.width() // 2
        center_y = self.height() // 2
        comm_width = self.community_container.width()
        comm_height = self.community_container.height()
        # 커뮤니티 창을 창의 세로 중앙에 배치
        community_y = (self.height() - comm_height) // 2
        community_x = center_x - comm_width // 2

        gap_x = 50
        gap_y = 0  # 커뮤니티 창과 거의 붙임
        offset_x = 100

        top_gap_x = 60  # 위쪽 슬롯 간격
        bottom_gap_x = -20  # 아래쪽 슬롯 간격
        slot_offset_y = 21
        slot_positions = [
            (community_x - top_gap_x, community_y - card_area_height - gap_y + slot_offset_y),
            (community_x + comm_width//2 - card_area_width//2, community_y - card_area_height - gap_y + slot_offset_y),
            (community_x + comm_width - card_area_width + top_gap_x, community_y - card_area_height - gap_y + slot_offset_y),
            (community_x + comm_width - card_area_width + bottom_gap_x, community_y + comm_height + gap_y + slot_offset_y),
            (community_x - bottom_gap_x, community_y + comm_height + gap_y + slot_offset_y),
        ]
        slot_indices = [0, 1, 2, 3, 4]

        ui_positions = [
            (10, 20),
            (self.width()//2 - name_w//2 - 100, 20),
            (self.width() - name_w - btn_w - 10, 20),
            (self.width() - name_w - btn_w - 10, self.height() - card_area_height - 100 + 60),
            (10, self.height() - card_area_height - 100 + 60),
        ]
        ui_indices = [0, 1, 2, 3, 4]

        player_container_width = self.card_width * 2 + 20 + 20 + 20
        for i in range(self.num_players):
            slot_x, slot_y = slot_positions[slot_indices[i]]
            self.player_labels[i][0].parent().move(slot_x, slot_y)
            x, y = ui_positions[ui_indices[i]]
            self.bet_labels[i].parent().move(x, y + 60)
            for j, btn in enumerate(self.action_buttons[i]):
                btn_x = x + player_container_width - 20 + (j % 2) * 105
                btn_y = y + (j // 2) * 40
                btn.move(btn_x, btn_y)
            self.name_labels[i].move(x, y)

    def post_blinds(self):
        # 스몰 블라인드
        if self.chips[self.sb_index] >= self.small_blind:
            self.chips[self.sb_index] -= self.small_blind
            self.player_bets[self.sb_index] = self.small_blind
            self.max_bet = self.small_blind
        else:
            self.all_in_players[self.sb_index] = True
            self.player_bets[self.sb_index] = self.chips[self.sb_index]
            self.chips[self.sb_index] = 0
            self.max_bet = self.player_bets[self.sb_index]

        # 빅 블라인드
        if self.chips[self.bb_index] >= self.big_blind:
            self.chips[self.bb_index] -= self.big_blind
            self.player_bets[self.bb_index] = self.big_blind
            self.max_bet = self.big_blind
        else:
            self.all_in_players[self.bb_index] = True
            self.player_bets[self.bb_index] = self.chips[self.bb_index]
            self.chips[self.bb_index] = 0
            self.max_bet = self.player_bets[self.bb_index]

        # UI 업데이트
        self.update_ui(self.sb_index)
        self.update_ui(self.bb_index)
        self.update_pot()

    def call(self, idx):
        if idx != self.current_turn or self.folded_players[idx]:
            return

        diff = self.max_bet - self.player_bets[idx]
        if diff == 0:
            self.check(idx)
            return

        if self.chips[idx] >= diff:
            self.chips[idx] -= diff
            self.player_bets[idx] += diff
            self.acted_players.add(idx)
            
            # 모든 칩을 베팅한 경우 올인 상태로 설정
            if self.chips[idx] == 0:
                self.all_in_players[idx] = True
            
            self.update_pot()
            self.update_ui(idx)
            self.advance_turn(f"🎯 P{idx+1} 콜 ({diff})\n베팅: {self.player_bets[idx]} / 칩: {self.chips[idx]}")
        else:
            self.all_in(idx)

    def check(self, idx):
        if idx != self.current_turn or self.folded_players[idx]:
            return

        if self.player_bets[idx] == self.max_bet:
            self.acted_players.add(idx)
            self.advance_turn(f"✅ P{idx+1} 체크\n베팅: {self.player_bets[idx]} / 칩: {self.chips[idx]}")
        else:
            self.update_message("❌ 체크할 수 없습니다.\n\n현재 베팅이 있습니다.\n콜 또는 폴드를 선택하세요.")

    def fold(self, idx):
        if idx != self.current_turn or self.folded_players[idx]:
            return

        self.folded_players[idx] = True
        self.acted_players.add(idx)
        # 이름 라벨에 (폴드) 표시
        self.name_labels[idx].setText(f"Player {idx + 1} (폴드)")
        
        # 폴드한 플레이어의 카드 슬롯 스타일 유지
        for card_label in self.player_labels[idx]:
            card_label.clear()
            card_label.setStyleSheet("""
                background-color: white;
                border: 2px solid #444;
                border-radius: 5px;
            """)
        
        # 한 명만 남았는지 확인
        active_players = [i for i in range(self.num_players) if not self.folded_players[i]]
        if len(active_players) == 1:
            self.complete_round()  # 팟 누적 및 초기화는 여기서만!
            winner_idx = active_players[0]
            pot = self.total_pot
            self.chips[winner_idx] += pot
            self.update_message(f"🏆 P{winner_idx+1} 승리!\n팟: {pot}\n보유칩: {self.chips[winner_idx]}")
            self.update_ui(winner_idx)
            self.disable_all_buttons()
            self.next_stage_button.setText('게임 재시작')
            self.next_stage_button.setEnabled(True)
            self.next_stage_button.clicked.disconnect()
            self.next_stage_button.clicked.connect(self.restart_game)
            return

        self.advance_turn(f"🃏 P{idx+1} 폴드\n베팅: {self.player_bets[idx]} / 칩: {self.chips[idx]}")

    def all_in(self, idx):
        if idx != self.current_turn or self.folded_players[idx]:
            return

        amount = self.chips[idx]
        self.chips[idx] = 0
        self.player_bets[idx] += amount
        self.all_in_players[idx] = True
        self.acted_players.add(idx)
        self.max_bet = max(self.max_bet, self.player_bets[idx])
        
        self.update_ui(idx)
        self.update_pot()

        self.advance_turn(f"🔥 P{idx+1} 올인! ({amount})\n베팅: {self.player_bets[idx]} / 칩: 0")

    def advance_turn(self, message):
        self.update_message(message)
        self.disable_current_player_buttons()

        # 다음 턴 플레이어 찾기
        next_player = self.get_next_active_player()
        
        # 라운드 완료 체크
        if self.is_round_complete():
            # 올인한 플레이어가 있는지 확인
            has_all_in = any(self.all_in_players)
            
            # 올인한 플레이어가 있으면 쇼다운 시작
            if has_all_in and not self.is_showdown:
                self.is_showdown = True
                self.start_showdown_effect()
                self.update_message("🎴 Show Down !!!\n\n모든 플레이어의 카드를 공개해주세요!")
                self.disable_all_buttons()
                self.next_stage_button.setEnabled(True)
                if self.current_round == "river":
                    self.next_stage_button.setText("카드 공개")
                else:
                    self.next_stage_button.setText("다음 카드 공개")
            else:
                # 올인한 플레이어가 없는 경우 다음 단계로 진행
                self.next_stage_button.setEnabled(True)
                self.next_stage_button.setText("다음 카드 공개")
                self.update_message("모든 플레이어가 행동을 마쳤습니다. 다음 단계로 진행하세요.")
            
            self.complete_round()
            return

        # 다음 턴으로 진행
        self.current_turn = next_player
        self.update_ui_for_turn()

    def is_round_complete(self):
        # 1. 모든 플레이어가 행동을 마쳤는지 확인
        all_acted = all(
            i in self.acted_players or 
            self.folded_players[i] or 
            self.all_in_players[i] or 
            self.chips[i] == 0
            for i in range(self.num_players)
        )

        # 2. 모든 활성 플레이어의 베팅이 동일한지 확인
        active_players = [i for i in range(self.num_players) if not self.folded_players[i]]
        if not active_players:
            return True  # 모든 플레이어가 폴드한 경우

        # 3. 마지막 레이즈 이후 모든 플레이어가 행동했는지 확인
        if self.last_raiser != -1:
            # 마지막 레이즈 이후의 플레이어들이 모두 행동했는지 확인
            players_after_raiser = []
            current_idx = (self.last_raiser + 1) % self.num_players
            while current_idx != self.last_raiser:
                if not self.folded_players[current_idx]:
                    players_after_raiser.append(current_idx)
                current_idx = (current_idx + 1) % self.num_players

            # 마지막 레이즈 이후의 모든 플레이어가 행동했는지 확인
            all_after_raiser_acted = all(
                i in self.acted_players or self.all_in_players[i]
                for i in players_after_raiser
            )
            if not all_after_raiser_acted:
                return False

        # 4. 모든 활성 플레이어의 베팅액이 동일한지 확인
        active_bets = [self.player_bets[i] for i in active_players]
        if len(set(active_bets)) > 1:
            return False

        # 5. 모든 조건이 충족되면 라운드 완료
        return all_acted

    def complete_round(self):
        # 현재 라운드의 베팅금을 누적 팟에 추가
        round_pot = sum(self.player_bets)
        self.accumulated_pot += round_pot
        self.total_pot = self.accumulated_pot
        
        # 각 플레이어의 누적 베팅금 업데이트
        for i in range(self.num_players):
            self.player_total_bets[i] += self.player_bets[i]
        
        # 현재 라운드 베팅 초기화
        self.current_round_pot = 0
        self.player_bets = [0] * self.num_players
        self.max_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.acted_players.clear()
        
        # UI 업데이트
        self.update_pot()
        for i in range(self.num_players):
            self.update_ui(i)
        
        # 모든 베팅 버튼 비활성화
        self.disable_all_buttons()

        # River 단계에서 베팅이 완료된 경우
        if self.current_round == "river":
            if self.is_showdown:
                # 쇼다운 상태에서는 바로 승자 결정
                self.determine_winner()
            else:
                # 일반적인 경우 카드 공개 요청
                self.update_message("베팅이 완료되었습니다. 각 플레이어의 카드를 공개해주세요.")
                self.next_stage_button.setText("카드 공개")
                self.next_stage_button.setEnabled(True)
        else:
            # 다음 라운드로 진행할 준비
            self.next_stage_button.setText("다음 카드 공개")
            self.next_stage_button.setEnabled(True)
            # 쇼다운 상태가 아닐 때만 일반 메시지 표시
            if not self.is_showdown:
                self.update_message("모든 플레이어가 행동을 마쳤습니다. 커뮤니티 카드를 공개하세요.")

    def get_next_active_player(self, start_index=None):
        if start_index is None:
            start_index = (self.current_turn + 1) % self.num_players

        # 다음 활성 플레이어 찾기
        for _ in range(self.num_players):
            if (not self.folded_players[start_index] and 
                not self.all_in_players[start_index] and 
                self.chips[start_index] > 0):
                return start_index
            start_index = (start_index + 1) % self.num_players
        return self.current_turn  # 활성 플레이어가 없으면 현재 플레이어 반환
    
    def show_next_stage(self):
        """다음 단계로 진행합니다."""
        # 카드 공개 버튼이 눌린 경우
        if self.next_stage_button.text() == "카드 공개":
            self.determine_winner()
            return

        # 모든 베팅 관련 변수 초기화
        self.max_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.acted_players.clear()

        # 라운드 진행
        self.community_stage += 1

        if self.community_stage == 1:
            self.current_round = "flop"
            # 플랍 카드 인식
            if not self.get_flop_cards():
                return
            if self.is_showdown:
                self.update_message("🎴 쇼다운!\n플랍 공개.\n카드를 모두 오픈!")
            else:
                self.update_message("🎴 플랍: " + " ".join(self.community_cards[:3]))
            self.next_stage_button.setText("다음 카드 공개")
            self.next_stage_button.setEnabled(True)
        elif self.community_stage == 2:
            self.current_round = "turn"
            # 턴 카드 인식
            if not self.get_turn_card():
                return
            if self.is_showdown:
                self.update_message("🎴 쇼다운!\n턴 공개.\n카드를 모두 오픈!")
            else:
                self.update_message("🎴 턴: " + " ".join(self.community_cards[:4]))
            self.next_stage_button.setText("다음 카드 공개")
            self.next_stage_button.setEnabled(True)
        elif self.community_stage == 3:
            self.current_round = "river"
            # 리버 카드 인식
            if not self.get_river_card():
                return
            if self.is_showdown:
                self.update_message("🎴 쇼다운!\n리버 공개.\n카드를 모두 오픈!")
                # 쇼다운 상태에서는 바로 승자 결정
                self.determine_winner()
                return
            else:
                self.update_message("🎴 리버: " + " ".join(self.community_cards))
            self.next_stage_button.setEnabled(False)
            self.start_river_betting()
            return

        # 다음 라운드 시작 위치 설정
        if self.community_stage > 0:  # 플랍 이후
            self.current_turn = self.get_next_active_player(self.sb_index)
        else:  # 프리플랍
            self.current_turn = self.utg_index if self.num_players > 3 else self.sb_index

        # UI 업데이트
        for i in range(self.num_players):
            self.update_ui(i)

        # 쇼다운 상태에서는 베팅 버튼을 비활성화하고 다음 단계 버튼만 활성화
        if self.is_showdown:
            self.disable_all_buttons()
            self.next_stage_button.setEnabled(True)
        else:
            self.update_ui_for_turn()
            self.next_stage_button.setEnabled(False)

    def start_river_betting(self):
        """River 단계의 베팅을 시작합니다."""
        # 쇼다운 상태에서는 베팅을 시작하지 않음
        if self.is_showdown:
            self.update_message("🎴 Show Down !!!\n\n리버 카드가 공개되었습니다.\n모든 플레이어의 카드를 공개해주세요!")
            self.next_stage_button.setText("카드 공개")
            self.next_stage_button.setEnabled(True)
            return

        # 활성 플레이어 확인
        active_players = [i for i in range(self.num_players) if not self.folded_players[i]]
        if not active_players:
            return

        # 첫 번째 활성 플레이어부터 시작
        self.current_turn = active_players[0]
        self.update_ui_for_turn()
        
        # 베팅 관련 변수 초기화
        self.max_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.acted_players.clear()

        # 메시지 업데이트
        self.update_message(f"River 베팅이 시작됩니다. Player {self.current_turn + 1}의 차례입니다.")

    def update_ui_for_turn(self):
        for i in range(self.num_players):
            for btn in self.action_buttons[i]:
                btn.setVisible(False)

        if self.current_turn is None:
            return

        # 쇼다운 상태에서는 베팅 버튼 비활성화
        if hasattr(self, 'is_showdown') and self.is_showdown:
            return

        if (self.folded_players[self.current_turn] or 
            self.all_in_players[self.current_turn] or 
            self.chips[self.current_turn] == 0):
            return

        # 올인한 플레이어가 있는지 확인하고, 현재 플레이어 이전에 올인한 플레이어가 있는지 확인
        can_raise = True
        for i in range(self.num_players):
            if self.all_in_players[i]:
                # 현재 플레이어보다 이전 순서에 올인한 플레이어가 있는지 확인
                if i < self.current_turn:
                    can_raise = False
                    break

        # 현재 플레이어의 총 베팅 가능 금액 계산
        current_total = (self.chips[self.current_turn] + 
                        self.player_bets[self.current_turn] + 
                        self.player_total_bets[self.current_turn])

        # 현재 플레이어보다 적은 총 베팅 가능 금액을 가진 플레이어가 있는지 확인
        can_all_in = True
        for i in range(self.num_players):
            if not self.folded_players[i]:
                # 모든 플레이어의 총 베팅 가능 금액으로 비교
                other_total = (self.chips[i] + 
                             self.player_bets[i] + 
                             self.player_total_bets[i])
                if other_total < current_total:
                    can_all_in = False
                    break

        for j, btn in enumerate(self.action_buttons[self.current_turn]):
            if j == 0:  # 콜
                btn.setVisible(self.max_bet > 0 and self.player_bets[self.current_turn] < self.max_bet)
            elif j == 1:  # 체크
                btn.setVisible(self.max_bet == 0 or self.player_bets[self.current_turn] == self.max_bet)
            elif j == 2:  # 폴드
                btn.setVisible(True)
            elif j == 3:  # 레이즈
                can_raise_amount = (self.chips[self.current_turn] > 
                                  (self.max_bet - self.player_bets[self.current_turn] + self.min_raise))
                btn.setVisible(can_raise and can_raise_amount)
            elif j == 4:  # 올인
                btn.setVisible(self.chips[self.current_turn] > 0 and can_all_in)

    def disable_current_player_buttons(self):
        for btn in self.action_buttons[self.current_turn]:
            btn.setVisible(False)

    def disable_all_buttons(self):
        for buttons in self.action_buttons:
            for btn in buttons:
                btn.setVisible(False)

    def update_game_state(self):
        for i in range(self.num_players):
            # 카드 슬롯 스타일 유지
            for card_label in self.player_labels[i]:
                card_label.setStyleSheet("""
                    background-color: white;
                    border: 2px solid #444;
                    border-radius: 5px;
                """)
            
            # 베팅 정보 업데이트
            self.bet_labels[i].setText(f"베팅: {self.player_bets[i]}")
            self.total_bet_labels[i].setText(f"누적: {self.player_total_bets[i]}")
            self.chip_labels[i].setText(f"칩: {self.chips[i]}")
        self.update_pot()

    def determine_winner(self):
        """승자를 결정합니다."""
        # 쇼다운 시 카드 인식
        if not self.get_player_cards():
            return

        # 활성 플레이어들의 카드 수집
        active_players = [(i, self.player_hands[i]) for i in range(self.num_players) if not self.folded_players[i]]
        if not active_players:
            return

        # 각 플레이어의 족보 점수와 상세 정보 계산
        all_hands = {}
        for player_idx, player_cards in active_players:
            # 현재 라운드에 따라 사용할 커뮤니티 카드 결정
            if self.current_round == "preflop":
                community_cards = []
            elif self.current_round == "flop":
                community_cards = self.community_cards[0:3]
            elif self.current_round == "turn":
                community_cards = self.community_cards[0:4]
            else:  # river
                community_cards = self.community_cards
                
            # 플레이어의 카드와 커뮤니티 카드를 합쳐서 족보 계산
            all_cards = player_cards + community_cards
            best_hand, hand_name, score, values = HandEvaluator.find_best_hand(all_cards)
            all_hands[player_idx] = {
                'score': score,
                'hand_name': hand_name,
                'values': values,
                'best_hand': best_hand
            }

        # HandEvaluator를 사용하여 승자 결정
        winners, hand_name, _ = HandEvaluator.determine_winner(active_players, self.community_cards)

        # 승자에게 팟 지급
        pot = self.total_pot
        pot_per_winner = pot // len(winners)  # 동점자가 있을 경우 팟 분배
        for winner_idx in winners:
            self.chips[winner_idx] += pot_per_winner

        # 모든 플레이어의 족보 정보 표시
        hand_info_text = "🎴 최종 족보 🎴\n"
        hand_info_text += "=" * 30 + "\n"
        
        # 족보 순위에 따라 정렬
        player_hands = [(i, all_hands[i]) for i in range(self.num_players) if not self.folded_players[i]]
        player_hands.sort(key=lambda x: (x[1]['score'], x[1]['values']), reverse=True)  # 높은 족보 순으로 정렬
        
        # 상위 3개 족보만 표시
        for rank, (player_idx, hand_info) in enumerate(player_hands[:3], 1):
            # 족보 이름과 상세 정보
            hand_name = hand_info['hand_name']
            values = hand_info['values']
            best_hand = hand_info['best_hand']
            
            # 상세 정보 추가
            detail_info = self.get_hand_detail(hand_info['score'], values, best_hand)
            
            # 승자 표시
            winner_mark = "👑 " if player_idx in winners else "  "
            # 순위 표시
            rank_mark = f"{rank}위: " if rank <= 3 else ""
            hand_info_text += f"{winner_mark}P{player_idx + 1}: {detail_info}\n"
        
        hand_info_text += "=" * 30 + "\n\n"

        winner_message = hand_info_text
        winner_message += "🏆 승자: " + ", ".join([f"P{w+1}" for w in winners]) + f"\n팟: {pot}"
        self.update_message(winner_message)

        # 베팅액 초기화 (팟 분배 후)
        self.player_bets = [0] * self.num_players
        self.player_total_bets = [0] * self.num_players
        self.total_pot = 0
        self.current_round_pot = 0
        self.accumulated_pot = 0

        # UI 업데이트
        for i in range(self.num_players):
            self.update_ui(i)
            self.bet_labels[i].setText("베팅: 0")
            self.total_bet_labels[i].setText("누적: 0")
        self.update_pot()

        # 다음 단계 버튼을 재시작 버튼으로 변경
        self.next_stage_button.setText("게임 재시작")
        self.next_stage_button.setEnabled(True)
        self.next_stage_button.clicked.disconnect()  # 기존 연결 해제
        self.next_stage_button.clicked.connect(self.restart_game)

        # 쇼다운 상태 초기화
        if hasattr(self, 'is_showdown'):
            self.is_showdown = False

    def restart_game(self):
        """게임을 재시작합니다."""
        # 칩이 0인 플레이어가 있는지 확인
        has_zero_chips = any(chip == 0 for chip in self.chips)
        
        # 칩이 0인 플레이어가 있으면 모든 플레이어에게 3000칩 추가
        if has_zero_chips:
            for i in range(self.num_players):
                self.chips[i] += 3000
        
        # 게임 상태 초기화
        self.player_hands = [[] for _ in range(self.num_players)]  # 빈 리스트로 초기화
        self.community_cards = [None] * 5  # None으로 초기화
        self.player_bets = [0] * self.num_players
        self.player_total_bets = [0] * self.num_players
        self.folded_players = [False] * self.num_players
        self.all_in_players = [False] * self.num_players
        self.community_stage = 0
        self.current_round = "preflop"
        self.max_bet = 0
        self.min_raise = 0
        self.last_raiser = -1
        self.acted_players = set()
        self.total_pot = 0
        self.current_round_pot = 0
        self.accumulated_pot = 0
        
        # 포지션 업데이트 (시계 방향으로 한 칸씩 이동)
        self.sb_index = (self.sb_index + 1) % self.num_players
        self.bb_index = (self.bb_index + 1) % self.num_players
        self.utg_index = (self.utg_index + 1) % self.num_players
        
        # 라운드 초기화
        self.init_round()
        
        # 프리플랍에서의 첫 베팅 순서 설정
        if self.num_players > 3:
            # UTG부터 시작
            self.current_turn = self.utg_index
        else:
            # 3명 이하일 경우 SB부터 시작
            self.current_turn = self.sb_index
        
        # UI 초기화
        self.update_message("")
        self.pot_label.setText("총 팟: 0")
        self.pot_label.setStyleSheet("""
            color: gold;
            font-size: 18px;
            background-color: #222;
            border: 2px solid #555;
            border-radius: 10px;
            padding: 8px;
        """)
        self.pot_label.setFixedSize(340, 70)
        
        # 팟 애니메이션 초기화
        if self.pot_animation_timer:
            self.pot_animation_timer.stop()
            self.pot_animation_timer = None
        
        # 쇼다운 효과 초기화
        if self.showdown_effect_timer:
            self.showdown_effect_timer.stop()
            self.showdown_effect_timer = None
        if self.showdown_overlay:
            self.showdown_overlay.hide()
        if self.showdown_proxy:
            self.showdown_proxy.hide()
        
        # UI 업데이트
        self.update_game_state()
        self.update_ui_for_turn()
        
        # 다음 단계 버튼 초기화
        self.next_stage_button.setText("다음 카드 공개")
        self.next_stage_button.setEnabled(False)
        self.next_stage_button.clicked.disconnect()
        self.next_stage_button.clicked.connect(self.show_next_stage)
        
        # 시작 메시지 표시
        msg = f"🎲 게임 시작!\nSB: P{self.sb_index+1}({self.small_blind}) / BB: P{self.bb_index+1}({self.big_blind})"
        if self.num_players > 3:
            msg += f" / UTG: P{self.utg_index+1}"
        msg += f"\n현재 차례: P{self.current_turn+1}"
        self.update_message(msg)

        # 이름 라벨 복구
        for i in range(self.num_players):
            position = ""
            if i == self.sb_index:
                position = " (SB)"
            elif i == self.bb_index:
                position = " (BB)"
            elif i == self.utg_index and self.num_players > 3:
                position = " (UTG)"
            self.name_labels[i].setText(f"Player {i + 1}{position}")

    def get_hand_detail(self, score, values, best_hand):
        """족보의 상세 정보를 반환합니다."""
        value_names = {14: 'A', 13: 'K', 12: 'Q', 11: 'J'}
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # 값들을 내림차순으로 정렬
        sorted_values = sorted(values, reverse=True)
        
        # 각 값의 이름으로 변환
        value_names_list = [value_names.get(v, str(v)) for v in sorted_values]
        
        if score == 10:  # 로열 스트레이트 플러시
            return f"로열 스트레이트 플러시 (A-K-Q-J-10)"
        elif score == 9:  # 스트레이트 플러시
            return f"스트레이트 플러시 ({'-'.join(value_names_list)})"
        elif score == 8:  # 포카드
            four_value = [v for v, count in value_counts.items() if count == 4][0]
            kicker = [v for v in sorted_values if v != four_value][0]
            return f"포카드 {value_names.get(four_value, str(four_value))} (킥커: {value_names.get(kicker, str(kicker))})"
        elif score == 7:  # 풀하우스
            three_value = [v for v, count in value_counts.items() if count == 3][0]
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            return f"풀하우스 {value_names.get(three_value, str(three_value))} 풀 {value_names.get(pair_value, str(pair_value))}"
        elif score == 6:  # 플러시
            return f"플러시 ({'-'.join(value_names_list)})"
        elif score == 5:  # 스트레이트
            return f"스트레이트 ({'-'.join(value_names_list)})"
        elif score == 4:  # 트리플
            three_value = [v for v, count in value_counts.items() if count == 3][0]
            kickers = [v for v in sorted_values if v != three_value][:2]
            return f"트리플 {value_names.get(three_value, str(three_value))} (킥커: {', '.join(value_names.get(k, str(k)) for k in kickers)})"
        elif score == 3:  # 투페어
            pairs = sorted([v for v, count in value_counts.items() if count == 2], reverse=True)
            kicker = [v for v in sorted_values if v not in pairs][0]
            return f"투페어 {value_names.get(pairs[0], str(pairs[0]))}-{value_names.get(pairs[1], str(pairs[1]))} (킥커: {value_names.get(kicker, str(kicker))})"
        elif score == 2:  # 원페어
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            kickers = [v for v in sorted_values if v != pair_value][:3]
            return f"원페어 {value_names.get(pair_value, str(pair_value))} (킥커: {', '.join(value_names.get(k, str(k)) for k in kickers)})"
        else:  # 하이카드
            return f"하이카드 ({'-'.join(value_names_list)})"

    def find_best_hand(self, cards):
        """7장의 카드 중 가장 높은 5장의 조합을 찾습니다."""
        from itertools import combinations
        
        # 모든 가능한 5장 조합 생성
        all_combinations = list(combinations(cards, 5))
        best_score = -1
        best_hand = None
        best_hand_name = ""

        for hand in all_combinations:
            score, hand_name = self.evaluate_hand(hand)
            if score > best_score:
                best_score = score
                best_hand = hand
                best_hand_name = hand_name

        return best_hand, best_hand_name, best_score

    def evaluate_hand(self, hand):
        """5장의 카드로 구성된 핸드를 평가합니다."""
        # 카드 정렬 (숫자 내림차순)
        sorted_cards = sorted(hand, key=lambda x: self.get_card_value(x), reverse=True)
        
        # 플러시 체크
        is_flush = len(set(card[-1] for card in hand)) == 1
        
        # 스트레이트 체크
        values = [self.get_card_value(card) for card in sorted_cards]
        is_straight = (len(set(values)) == 5 and max(values) - min(values) == 4) or \
                     (set(values) == {14, 2, 3, 4, 5})  # A-5 스트레이트
        
        # 페어 체크
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # 족보 판정
        if is_flush and is_straight:
            if values == [14, 13, 12, 11, 10]:  # 로열 스트레이트 플러시
                return 10, "로열 스트레이트 플러시"
            return 9, "스트레이트 플러시"
        
        # 포카드
        if 4 in value_counts.values():
            return 8, "포카드"
        
        # 풀하우스
        if sorted(value_counts.values()) == [2, 3]:
            return 7, "풀하우스"
        
        # 플러시
        if is_flush:
            return 6, "플러시"
        
        # 스트레이트
        if is_straight:
            return 5, "스트레이트"
        
        # 트리플
        if 3 in value_counts.values():
            return 4, "트리플"
        
        # 투페어
        if list(value_counts.values()).count(2) == 2:
            return 3, "투페어"
        
        # 원페어
        if 2 in value_counts.values():
            return 2, "원페어"
        
        # 하이카드
        return 1, "하이카드"

    def get_card_value(self, card):
        """카드의 숫자 값을 반환합니다."""
        value = card[:-1]  # 마지막 문자(문양) 제외
        if value == 'A':
            return 14
        elif value == 'K':
            return 13
        elif value == 'Q':
            return 12
        elif value == 'J':
            return 11
        else:
            return int(value)

    def update_pot(self):
        self.current_round_pot = sum(self.player_bets)
        self.total_pot = self.current_round_pot + self.accumulated_pot
        self.pot_label.setText(f"총 팟: {self.total_pot}")
        self.apply_pot_glow(self.total_pot)

    def update_ui(self, player_idx):
        # 플레이어 카드 정보 업데이트
        position = ""
        if player_idx == self.sb_index:
            position = " (SB)"
        elif player_idx == self.bb_index:
            position = " (BB)"
        elif player_idx == self.utg_index and self.num_players > 3:
            position = " (UTG)"

        # 카드 슬롯 스타일 유지
        for card_label in self.player_labels[player_idx]:
            card_label.setStyleSheet("""
                background-color: white;
                border: 2px solid #444;
                border-radius: 5px;
            """)

        # 베팅 정보 업데이트
        self.bet_labels[player_idx].setText(f"베팅: {self.player_bets[player_idx]}")
        self.total_bet_labels[player_idx].setText(f"누적: {self.player_total_bets[player_idx]}")
        self.chip_labels[player_idx].setText(f"칩: {self.chips[player_idx]}")

    def apply_pot_glow(self, total):
        # 플레이어 수에 따른 동적 경계점 설정
        base_threshold = 1000 * self.num_players
        high_threshold = base_threshold * 2
        super_threshold = base_threshold * 3

        # 기본 스타일 설정
        base_style = "background-color: #222; border-radius: 10px; padding: 8px;"
        
        # 기존 타이머가 있다면 중지
        if self.pot_animation_timer:
            self.pot_animation_timer.stop()
            self.pot_animation_timer = None

        # 그림자 효과 설정
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 0)
        
        if total >= super_threshold:
            # 슈퍼 하이 팟 효과
            self.pot_label.setStyleSheet(f"""
                {base_style}
                color: #FFD700;
                font-size: 24px;
                font-weight: bold;
                border: 5px solid #FFD700;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #222, stop:0.5 #444, stop:1 #222);
            """)
            self.pot_label.setFixedSize(400, 100)
            shadow.setColor(QColor("#FFD700"))
            self.start_pot_animation(30, 3)  # 매우 빠른 떨림, 강한 강도
        elif total >= high_threshold:
            # 하이 팟 효과
            self.pot_label.setStyleSheet(f"""
                {base_style}
                color: #FFD700;
                font-size: 22px;
                font-weight: bold;
                border: 4px solid #FFD700;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #222, stop:0.5 #333, stop:1 #222);
            """)
            self.pot_label.setFixedSize(380, 90)
            shadow.setColor(QColor("#FFD700"))
            self.start_pot_animation(50, 2)  # 빠른 떨림, 중간 강도
        elif total >= base_threshold:
            # 일반 하이 팟 효과
            self.pot_label.setStyleSheet(f"""
                {base_style}
                color: gold;
                font-size: 20px;
                font-weight: bold;
                border: 3px solid #FFD700;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #222, stop:1 #333);
            """)
            self.pot_label.setFixedSize(360, 80)
            shadow.setColor(QColor("gold"))
            self.start_pot_animation(80, 1)  # 느린 떨림, 약한 강도
        else:
            # 기본 효과
            self.pot_label.setStyleSheet(f"""
                {base_style}
                color: gold;
                font-size: 18px;
                border: 2px solid #555;
            """)
            self.pot_label.setFixedSize(340, 70)
            shadow.setColor(QColor("gold"))

        self.pot_label.setGraphicsEffect(shadow)
        self.update_pot_position()

    def start_pot_animation(self, interval, intensity):
        """팟 라벨의 떨림 애니메이션을 시작합니다."""
        if not self.pot_animation_timer:
            self.pot_animation_timer = QTimer(self)
            self.pot_animation_timer.timeout.connect(self.update_pot_animation)
            self.pot_animation_intensity = intensity
            self.pot_animation_timer.start(interval)

    def update_pot_animation(self):
        """팟 라벨의 떨림 애니메이션을 업데이트합니다."""
        # 랜덤한 방향으로 떨림 효과 생성
        import random
        
        # 떨림 강도에 따라 오프셋 범위 조정
        max_offset = self.pot_animation_intensity
        offset_x = random.randint(-max_offset, max_offset)
        offset_y = random.randint(-max_offset, max_offset)
        
        # 원래 위치에서 오프셋 적용
        new_x = self.original_pot_x + offset_x
        new_y = self.original_pot_y + offset_y
        
        self.pot_label.move(int(new_x), int(new_y))

    def update_showdown_text_positions(self):
        """쇼다운 텍스트의 위치와 크기를 업데이트합니다."""
        if not self.showdown_text:
            return

        width = self.showdown_overlay.width()
        height = self.showdown_overlay.height()
        
        # 대각선 길이 계산
        diagonal = (width ** 2 + height ** 2) ** 0.5
        
        # 폰트 크기를 대각선 길이에 비례하여 계산 (약간 작게 조정)
        font_size = int(diagonal / 10)  # 8에서 10으로 변경하여 여유 공간 확보
        
        # 텍스트 크기 설정 (여유 공간 추가)
        self.showdown_text.setStyleSheet(f"""
            color: rgba(255, 0, 0, {self.showdown_alpha});
            font-size: {font_size}px;
            font-weight: bold;
            font-family: 'Arial Black';
            background: transparent;
        """)
        
        # 텍스트 크기 조정 (여유 공간 추가)
        self.showdown_text.setFixedSize(width + 100, height + 100)  # 여유 공간 추가
        
        # 대각선 각도 계산 (필드 크기에 맞춰 동적으로 조정)
        angle = math.degrees(math.atan2(height, width))
        
        # 텍스트를 중앙에 배치하고 계산된 각도로 회전
        self.showdown_proxy.setPos(width/2 - self.showdown_text.width()/2, 
                                 height/2 - self.showdown_text.height()/2)
        self.showdown_proxy.setRotation(angle)
        self.showdown_proxy.setTransformOriginPoint(self.showdown_text.width()/2, 
                                                  self.showdown_text.height()/2)

    def start_showdown_effect(self):
        """쇼다운 효과를 시작합니다."""
        if not self.showdown_effect_timer:
            self.showdown_effect_timer = QTimer(self)
            self.showdown_effect_timer.timeout.connect(self.update_showdown_effect)
            self.showdown_alpha = 0
            self.showdown_overlay.show()
            self.showdown_proxy.show()
            self.update_showdown_text_positions()
            self.showdown_effect_timer.start(50)
            # 쇼다운 효과 시작 시 모든 베팅 버튼 비활성화
            self.disable_all_buttons()

    def update_showdown_effect(self):
        """쇼다운 효과를 업데이트합니다."""
        self.showdown_alpha = min(0.3, self.showdown_alpha + 0.02)  # 최대 30% 투명도
        self.showdown_text.setStyleSheet(f"""
            color: rgba(255, 0, 0, {self.showdown_alpha});
            font-weight: bold;
            font-family: 'Arial Black';
            background: transparent;
        """)
        self.showdown_overlay.setStyleSheet(f"background-color: rgba(255, 215, 0, {self.showdown_alpha});")
        self.update_showdown_text_positions()
        
        if self.showdown_alpha >= 0.3:
            self.showdown_effect_timer.stop()
            self.showdown_effect_timer = None

    def raise_bet(self, idx):
        if idx != self.current_turn or self.folded_players[idx]:
            return

        current_call = self.max_bet - self.player_bets[idx]
        min_raise_amount = 100
        min_total = current_call + min_raise_amount

        all_in_players = [i for i in range(self.num_players) if self.all_in_players[i]]
        if all_in_players:
            max_all_in = max(self.player_bets[i] for i in all_in_players)
            min_total = max(min_total, max_all_in)

        active_players = [i for i in range(self.num_players) 
                         if not self.folded_players[i]]
        if active_players:
            # 현재 플레이어의 총 베팅 가능 금액
            current_player_total = self.chips[idx] + self.player_bets[idx]
            
            # 다른 플레이어들의 총 베팅 가능 금액 계산
            other_players_totals = []
            for i in active_players:
                if i != idx:  # 현재 플레이어 제외
                    if self.all_in_players[i]:
                        # 올인한 플레이어는 현재 베팅 금액만 고려
                        other_players_totals.append(self.player_bets[i])
                    else:
                        # 일반 플레이어는 보유 칩 + 현재 베팅 금액 고려
                        total = self.chips[i] + self.player_bets[i]
                        other_players_totals.append(total)
            
            if other_players_totals:
                # 다른 플레이어들의 총 베팅 가능 금액 중 최소값
                min_other_total = min(other_players_totals)
                # 현재 플레이어의 총 베팅 가능 금액과 다른 플레이어들의 최소값 중 작은 값에서
                # 현재 플레이어의 베팅액을 뺀 값이 최대 레이즈 금액
                max_total = min(current_player_total, min_other_total) - self.player_bets[idx]
            else:
                max_total = self.chips[idx]
        else:
            max_total = self.chips[idx]

        if min_total > max_total:
            self.update_message("❌ 최소 레이즈 금액이 보유 칩보다 많습니다.")
            return

        dialog = RaiseDialog(self, min_total, max_total, self.max_bet)
        
        # 플레이어의 이름 UI 위치를 기준으로 다이얼로그 위치 설정
        name_label = self.name_labels[idx]
        
        # 플레이어의 위치에 따라 다이얼로그 위치 계산
        if idx < 3:  # 위쪽 플레이어들 (0, 1, 2)
            dialog_x = name_label.x() + name_label.width() + 10
            dialog_y = name_label.y()
        else:  # 아래쪽 플레이어들 (3, 4)
            dialog_x = name_label.x() + name_label.width() + 10
            dialog_y = name_label.y() - 40
        
        # 화면 밖으로 나가지 않도록 조정
        window_geometry = self.geometry()
        if dialog_x + dialog.width() > window_geometry.width():
            dialog_x = window_geometry.width() - dialog.width() - 10
        if dialog_x < 0:
            dialog_x = 10
        if dialog_y + dialog.height() > window_geometry.height():
            dialog_y = window_geometry.height() - dialog.height() - 10
        if dialog_y < 0:
            dialog_y = 10
        
        # 전역 좌표로 변환
        global_pos = self.mapToGlobal(QPoint(dialog_x, dialog_y))
        dialog.set_position(global_pos.x(), global_pos.y())
        
        if dialog.exec_() == QDialog.Accepted:
            amount = dialog.get_value()
            if amount < min_total:
                self.update_message(f"❌ 최소 레이즈 금액은 {min_total}입니다.")
                return

            self.chips[idx] -= amount
            self.player_bets[idx] += amount
            self.max_bet = max(self.max_bet, self.player_bets[idx])
            self.last_raiser = idx
            self.min_raise = amount - current_call
            self.acted_players.add(idx)
            
            if self.chips[idx] == 0:
                self.all_in_players[idx] = True
                self.update_message(f"🔥 P{idx+1} 올인! ({amount}칩)\n베팅: {self.player_bets[idx]} / 칩: 0")
            else:
                self.update_message(f"🎯 P{idx+1}가 {amount}칩을 레이즈합니다.")
            
            self.update_ui(idx)
            self.update_pot()
            self.advance_turn(self.rotated_message_label.text())

    def update_message(self, text):
        """메시지를 업데이트하는 함수"""
        self.rotated_message_label.setText(text)

    def get_flop_cards(self):
        """플랍 카드 3장을 인식하고 저장합니다."""
        max_retries = 3
        retry_count = 0
        self.update_message("플랍 카드 인식 중...")
        
        while retry_count < max_retries:
            try:
                # 좌표가 유효한지 확인하고 필요시 재추출
                if self.card_detector.should_re_extract_coordinates():
                    if not self.card_detector.extract_card_coordinates():
                        retry_count += 1
                        if retry_count < max_retries:
                            self.update_message("카드 좌표 추출 실패.\n카드 배치를 확인해주세요.\n다시 시도중... ({retry_count}/{max_retries})")
                            QTimer.singleShot(1000, lambda: None)
                        else:
                            self.update_message("카드 좌표 추출에 실패했습니다.")
                            return False
                        continue
                
                # 플랍 카드만 인식
                flop_cards = self.card_detector.detect_flop_cards()
                if flop_cards:
                    # Unknown이 아닌 카드만 저장
                    valid_cards = [card for card in flop_cards if card != "Unknown"]
                    if len(valid_cards) == 3:
                        self.community_cards[0:3] = valid_cards
                        self.update_message("플랍 카드 인식 완료")
                        return True
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            self.update_message(f"플랍 카드 인식 실패: {len(valid_cards)}/3장 인식됨\n카드 배치나 조명의 문제일 수 있습니다.\n다시 시도중... ({retry_count}/{max_retries})")
                            QTimer.singleShot(1000, lambda: None)
                        else:
                            self.update_message(f"플랍 카드 인식 실패: {len(valid_cards)}/3장 인식됨")
                            return False
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.update_message("플랍 카드 인식에 실패했습니다.\n카드 배치나 조명의 문제일 수 있습니다.\n다시 시도중... ({retry_count}/{max_retries})")
                        QTimer.singleShot(1000, lambda: None)
                    else:
                        self.update_message("플랍 카드 인식에 실패했습니다.")
                        return False
            except Exception as e:
                print(f"Error in get_flop_cards: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    self.update_message(f"카드 인식 중 오류 발생: {str(e)}\n다시 시도중... ({retry_count}/{max_retries})")
                    QTimer.singleShot(1000, lambda: None)
                else:
                    self.update_message("카드 인식 중 오류가 발생했습니다.")
                    return False
        return False

    def get_turn_card(self):
        """턴 카드 1장을 인식하고 저장합니다."""
        max_retries = 3
        retry_count = 0
        self.update_message("턴 카드 인식 중...")
        
        while retry_count < max_retries:
            try:
                # 좌표가 유효한지 확인하고 필요시 재추출
                if self.card_detector.should_re_extract_coordinates():
                    if not self.card_detector.extract_card_coordinates():
                        retry_count += 1
                        if retry_count < max_retries:
                            self.update_message("카드 좌표 추출 실패.\n카드 배치를 확인해주세요.\n다시 시도중... ({retry_count}/{max_retries})")
                            QTimer.singleShot(1000, lambda: None)
                        else:
                            self.update_message("카드 좌표 추출에 실패했습니다.")
                            return False
                        continue
                
                # 턴 카드만 인식
                turn_card = self.card_detector.detect_turn_card()
                if turn_card and turn_card != "Unknown":
                    self.community_cards[3] = turn_card
                    self.update_message("턴 카드 인식 완료")
                    return True
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.update_message("턴 카드 인식 실패: 카드를 인식할 수 없습니다.\n카드 배치나 조명의 문제일 수 있습니다.\n다시 시도중... ({retry_count}/{max_retries})")
                        QTimer.singleShot(1000, lambda: None)
                    else:
                        self.update_message("턴 카드 인식 실패: 카드를 인식할 수 없습니다.")
                        return False
            except Exception as e:
                print(f"Error in get_turn_card: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    self.update_message(f"카드 인식 중 오류 발생: {str(e)}\n다시 시도중... ({retry_count}/{max_retries})")
                    QTimer.singleShot(1000, lambda: None)
                else:
                    self.update_message("카드 인식 중 오류가 발생했습니다.")
                    return False
        return False

    def get_river_card(self):
        """리버 카드 1장을 인식하고 저장합니다."""
        max_retries = 3
        retry_count = 0
        self.update_message("리버 카드 인식 중...")
        
        while retry_count < max_retries:
            try:
                # 좌표가 유효한지 확인하고 필요시 재추출
                if self.card_detector.should_re_extract_coordinates():
                    if not self.card_detector.extract_card_coordinates():
                        retry_count += 1
                        if retry_count < max_retries:
                            self.update_message("카드 좌표 추출 실패.\n카드 배치를 확인해주세요.\n다시 시도중... ({retry_count}/{max_retries})")
                            QTimer.singleShot(1000, lambda: None)
                        else:
                            self.update_message("카드 좌표 추출에 실패했습니다.")
                            return False
                        continue
                
                # 리버 카드만 인식
                river_card = self.card_detector.detect_river_card()
                if river_card and river_card != "Unknown":
                    self.community_cards[4] = river_card
                    self.update_message("리버 카드 인식 완료")
                    return True
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.update_message("리버 카드 인식 실패: 카드를 인식할 수 없습니다.\n카드 배치나 조명의 문제일 수 있습니다.\n다시 시도중... ({retry_count}/{max_retries})")
                        QTimer.singleShot(1000, lambda: None)
                    else:
                        self.update_message("리버 카드 인식 실패: 카드를 인식할 수 없습니다.")
                        return False
            except Exception as e:
                print(f"Error in get_river_card: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    self.update_message(f"카드 인식 중 오류 발생: {str(e)}\n다시 시도중... ({retry_count}/{max_retries})")
                    QTimer.singleShot(1000, lambda: None)
                else:
                    self.update_message("카드 인식 중 오류가 발생했습니다.")
                    return False
        return False

    def get_player_cards(self):
        """쇼다운 시 플레이어 카드를 인식하고 저장합니다."""
        max_retries = 3
        retry_count = 0
        self.update_message("플레이어 카드 인식 중...")
        
        while retry_count < max_retries:
            try:
                # 좌표가 유효한지 확인하고 필요시 재추출
                if self.card_detector.should_re_extract_coordinates():
                    if not self.card_detector.extract_card_coordinates():
                        retry_count += 1
                        if retry_count < max_retries:
                            self.update_message("카드 좌표 추출 실패.\n카드 배치를 확인해주세요.\n다시 시도중... ({retry_count}/{max_retries})")
                            QTimer.singleShot(1000, lambda: None)
                        else:
                            self.update_message("카드 좌표 추출에 실패했습니다.")
                            return False
                        continue
                
                # 각 플레이어의 카드 인식
                all_cards_valid = True
                for i in range(self.num_players):
                    if not self.folded_players[i]:
                        player_cards = self.card_detector.detect_player_cards(i + 1)
                        if player_cards:
                            # Unknown이 아닌 카드만 저장
                            valid_cards = [card for card in player_cards if card != "Unknown"]
                            if len(valid_cards) == 2:
                                self.player_hands[i] = valid_cards
                            else:
                                all_cards_valid = False
                                break
                        else:
                            all_cards_valid = False
                            break
                
                if all_cards_valid:
                    self.update_message("플레이어 카드 인식 완료")
                    return True
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.update_message("플레이어 카드 인식 실패: 일부 카드를 인식할 수 없습니다.\n카드 배치나 조명의 문제일 수 있습니다.\n다시 시도중... ({retry_count}/{max_retries})")
                        QTimer.singleShot(1000, lambda: None)
                    else:
                        self.update_message("플레이어 카드 인식 실패: 일부 카드를 인식할 수 없습니다.")
                        return False
            except Exception as e:
                print(f"Error in get_player_cards: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    self.update_message(f"카드 인식 중 오류 발생: {str(e)}\n다시 시도중... ({retry_count}/{max_retries})")
                    QTimer.singleShot(1000, lambda: None)
                else:
                    self.update_message("카드 인식 중 오류가 발생했습니다.")
                    return False
        return False

    def init_round(self):
        """라운드를 초기화하고 게임 상태를 설정합니다."""
        # 게임 상태 초기화
        self.player_bets = [0] * self.num_players
        self.player_total_bets = [0] * self.num_players
        self.folded_players = [False] * self.num_players
        self.all_in_players = [False] * self.num_players
        self.max_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.acted_players.clear()
        self.community_stage = 0
        self.current_round = "preflop"
        self.total_pot = 0
        self.current_round_pot = 0

        # 블라인드 포스팅
        self.post_blinds()
        
        # UI 업데이트 - 모든 카드 슬롯에 빈 슬롯 표시
        for i in range(self.num_players):
            for card_label in self.player_labels[i]:
                card_label.setStyleSheet("""
                    background-color: white;
                    border: 2px solid #444;
                    border-radius: 5px;
                """)
        
        self.update_game_state()
        self.update_ui_for_turn()

        # 시작 메시지 표시
        msg = f"🎲 게임 시작!\nSB: P{self.sb_index+1}({self.small_blind}) / BB: P{self.bb_index+1}({self.big_blind})"
        if self.num_players > 3:
            msg += f" / UTG: P{self.utg_index+1}"
        msg += f"\n현재 차례: P{self.current_turn+1}"
        self.update_message(msg)

    def show_camera_view(self):
        """카메라로 캡처한 화면을 3초간 보여줍니다."""
        try:
            # 카메라로 이미지 캡처
            image = self.card_detector.picam2.capture_array()
            
            # 이미지 크기 조정 (게임 화면에 맞게)
            height, width = image.shape[:2]
            aspect_ratio = width / height
            max_width = self.width() // 2
            max_height = self.height() // 2
            
            if width > max_width:
                width = max_width
                height = int(width / aspect_ratio)
            if height > max_height:
                height = max_height
                width = int(height * aspect_ratio)
                
            image = cv2.resize(image, (width, height))
            
            # OpenCV BGR -> RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # QImage로 변환
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 라벨 크기 설정 및 이미지 표시
            self.camera_view_label.setFixedSize(width, height)
            self.camera_view_label.setPixmap(QPixmap.fromImage(qt_image))
            
            # 라벨을 화면 중앙에 위치
            x = (self.width() - width) // 2
            y = (self.height() - height) // 2
            self.camera_view_label.move(x, y)
            
            # 라벨 표시
            self.camera_view_label.show()
            
            # 3초 후 라벨 숨기기
            QTimer.singleShot(3000, self.camera_view_label.hide)
            
        except Exception as e:
            self.update_message(f"카메라 화면 캡처 실패: {str(e)}")