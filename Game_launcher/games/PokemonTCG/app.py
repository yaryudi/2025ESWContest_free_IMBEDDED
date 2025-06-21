import sys
import os
import random
import math
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QStyleOptionButton, QDialog, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QFontDatabase, QTransform, QPen
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer, QPointF, QSize, QPoint, QByteArray, QBuffer, QIODevice, QRect
from energy_management_dialog import EnergyManagementDialog
from hp_management_dialog import HPManagementDialog
from status_management_dialog import StatusManagementDialog
from evolution_management_dialog import EvolutionManagementDialog
from position_management_dialog import PositionManagementDialog
from tool_management_dialog import ToolManagementDialog
from random_number_dialog import RandomNumberDialog
from stadium_management_dialog import StadiumManagementDialog

BG_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'board_bg.png')
STADIUM_IMAGE_DIR = os.path.join(os.path.dirname(__file__), '스타디움_이미지')
FONT_PATH = os.path.join(os.path.dirname(__file__), 'font', 'PFstardust_font', 'PF스타더스트 3.0 Bold.ttf')
GB_FONT_PATH = os.path.join(os.path.dirname(__file__), 'font', 'Pokemon_GB_font', 'PokemonGb-RAeo.ttf')
COIN_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'coin_images')

# 레어도별 가중치 설정 (높을수록 더 자주 등장)
RARITY_WEIGHTS = {
    'Common': 1.0,           # 일반
    'Uncommon': 0.7,         # 비일반
    'Rare': 0.4,            # 레어
    'Rare_Secret': 0.2,      # 시크릿 레어
    'Rare_Prism_Star': 0.15, # 프리즘 스타 레어
    'Hyper_Rare': 0.1,       # 하이퍼 레어
    'ACE_SPEC_Rare': 0.05,   # 에이스 스펙 레어
    'Promo': 0.3            # 프로모션
}

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

class CardSlot(QLabel):
    def __init__(self, color, parent=None, rotation=0, font_family=None):
        super().__init__(parent)
        self.color = color
        self.card_name = ""
        self.rotation = rotation
        self.hp = 0  # 체력 값 추가
        self.status_dict = {}  # 상태이상 정보(예: {'poison': True, 'burn': False, 'sleep': False, 'paralysis': False, 'confusion': False})
        # hp_label의 부모를 PokemonBoard로 설정
        self.hp_label = QLabel(parent)  # 부모를 parent로!
        self.hp_label.setStyleSheet("""
            background-color: white;
            border: 2px solid black;
            border-radius: 4px;
            padding: 2px;
            font-weight: bold;
            color: black;
        """)
        self.hp_label.setAlignment(Qt.AlignCenter)
        self.hp_label.setFixedSize(45, 25)  # 3자리 숫자를 표시할 수 있는 크기
        self.hp_label.hide()  # 초기에는 숨김
        
        # 상태이상 텍스트를 표시할 라벨
        self.status_label = QLabel(parent)
        self.status_label.setStyleSheet("""
            background-color: white;
            border: 2px solid black;
            border-radius: 4px;
            padding: 2px;
            font-weight: bold;
            color: black;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(45, 25)  # 한글 2글자를 표시할 수 있는 크기
        self.status_label.hide()  # 초기에는 숨김
        
        if font_family:
            self.setFont(QFont(font_family, 10))
            self.hp_label.setFont(QFont(font_family, 10))
            self.status_label.setFont(QFont(font_family, 10))
        self.setStyleSheet(f"background-color: rgb{color}; border: 6px solid black; border-radius: 8px;")
        self.setParent(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        
        self.tool_label = QLabel(parent)
        self.tool_label.setFixedSize(28, 28)
        self.tool_label.hide()

    def set_hp(self, hp_value):
        LABEL_W, LABEL_H = 45, 25
        self.hp = hp_value
        if self.rotation == 0:
            self.hp_label.hide()
            return
        if not hp_value or hp_value <= 0:
            self.hp_label.hide()
            return
        text = str(hp_value)
        # 회전된 텍스트를 QPixmap으로 만들어서 hp_label에 세팅
        if self.rotation == 90 or self.rotation == -90:
            pix = QPixmap(LABEL_H, LABEL_W)  # 25x45
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)
            font = self.hp_label.font()
            font.setPointSize(10)  # 폰트 크기 설정
            painter.setFont(font)
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawRect(0, 0, pix.width()-1, pix.height()-1)
            painter.setPen(QColor(0, 0, 0))
            painter.translate(pix.width()//2, pix.height()//2)
            painter.rotate(self.rotation)
            painter.translate(-LABEL_W//2, -LABEL_H//2)
            painter.drawText(QRect(0, 0, LABEL_W, LABEL_H), Qt.AlignCenter, text)
            painter.end()
            self.hp_label.setPixmap(pix)
            self.hp_label.setFixedSize(LABEL_H, LABEL_W)
        else:
            pix = QPixmap(LABEL_W, LABEL_H)
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)
            font = self.hp_label.font()
            font.setPointSize(10)  # 폰트 크기 설정
            painter.setFont(font)
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawRect(0, 0, pix.width()-1, pix.height()-1)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(QRect(0, 0, LABEL_W, LABEL_H), Qt.AlignCenter, text)
            painter.end()
            self.hp_label.setPixmap(pix)
            self.hp_label.setFixedSize(LABEL_W, LABEL_H)
        self.hp_label.show()
        self.update_hp_label_position()  # 위치 갱신 추가

    def update_hp_label_position(self):
        if self.rotation == 0 or not self.hp or self.hp <= 0:
            self.hp_label.hide()
            return
        self.hp_label.show()
        LABEL_W, LABEL_H = 45, 25
        slot_geom = self.geometry()
        abs_x = slot_geom.x()
        abs_y = slot_geom.y()
        if self.rotation == 90:  # 플레이어1: 슬롯 오른쪽 옆면(세로)
            x = abs_x + slot_geom.width()
            y = abs_y
        elif self.rotation == -90:  # 플레이어2: 슬롯 왼쪽 옆면(세로)
            x = abs_x - LABEL_H
            y = abs_y + (slot_geom.height() - LABEL_W)
        else:
            x = abs_x
            y = abs_y - LABEL_H
        self.hp_label.move(x, y)
        # 상태이상 라벨 위치도 함께 업데이트
        self.update_status_label_position()

    def clear_hp(self):
        """체력 표시 초기화"""
        self.hp = 0
        self.hp_label.hide()

    def set_card(self, card_name):
        """카드 이름을 설정하고 표시"""
        self.card_name = card_name
        self.setText(card_name)
        self.setStyleSheet(f"""
            background-color: rgb{self.color};
            border: 6px solid black;
            border-radius: 8px;
            color: black;
            font-weight: bold;
            padding: 5px;
        """)

    def clear_card(self):
        """카드 정보 초기화"""
        self.card_name = ""
        self.setText("")
        self.setStyleSheet(f"background-color: rgb{self.color}; border: 6px solid black; border-radius: 8px;")

    def paintEvent(self, event):
        if self.rotation == 0:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.rotation)
        painter.translate(-self.width() / 2, -self.height() / 2)
        # 오라 그리기 부분 제거됨
        painter.setFont(self.font())
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())
        painter.restore()

    def set_status_text(self, status_text):
        # 상태별 배경색 지정
        bg_color = "white"
        if status_text == "수면":
            bg_color = "#b3e5fc"  # 연한 하늘색
        elif status_text == "마비":
            bg_color = "#ffe082"  # 더 진한 연노랑(머스타드 느낌)
        elif status_text == "혼란":
            bg_color = "white"    # 혼란은 흰색(혹은 다른 색상 지정 가능)

        self.status_label.setStyleSheet(f"""
            background-color: {bg_color};
            border: 2px solid black;
            border-radius: 4px;
            padding: 2px;
            font-weight: bold;
            color: black;
        """)
        if not status_text:
            self.status_label.hide()
            self.update_status_label_position()  # 상태이상 해제 시에도 위치 갱신
            return
        # 회전된 텍스트를 QPixmap으로 만들어서 status_label에 세팅
        LABEL_W, LABEL_H = 45, 25
        pix = QPixmap(LABEL_H, LABEL_W)  # 25x45
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = self.status_label.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QColor(bg_color))  # 배경색을 상태별로 지정
        painter.drawRect(0, 0, pix.width()-1, pix.height()-1)
        painter.setPen(QColor(0, 0, 0))
        painter.translate(pix.width()//2, pix.height()//2)
        painter.rotate(self.rotation)
        painter.translate(-LABEL_W//2, -LABEL_H//2)
        painter.drawText(QRect(0, 0, LABEL_W, LABEL_H), Qt.AlignCenter, status_text)
        painter.end()
        self.status_label.setPixmap(pix)
        self.status_label.setFixedSize(LABEL_H, LABEL_W)
        self.status_label.show()
        self.update_status_label_position()  # 상태이상 표시 시에도 위치 갱신

    def update_status_label_position(self):
        if not self.status_label.isVisible():
            return
            
        LABEL_W, LABEL_H = 45, 25
        slot_geom = self.geometry()
        abs_x = slot_geom.x()
        abs_y = slot_geom.y()
        
        if self.rotation == 90:  # 플레이어1: 슬롯 오른쪽 옆면(세로)
            x = abs_x + slot_geom.width()
            y = abs_y + LABEL_W  # hp_label 아래에 위치
        elif self.rotation == -90:  # 플레이어2: 슬롯 왼쪽 옆면(세로)
            x = abs_x - LABEL_H
            y = abs_y + (slot_geom.height() - LABEL_W * 2)  # hp_label 아래에 위치
        else:
            return
            
        self.status_label.move(x, y)

    def set_tool_icon(self, icon_path):
        if icon_path:
            pixmap = QPixmap(icon_path).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            transform = QTransform().rotate(self.rotation)
            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
            self.tool_label.setPixmap(rotated_pixmap)
            self.tool_label.setFixedSize(25, 25)
            self.tool_label.show()
            self.update_tool_label_position()
        else:
            self.tool_label.hide()

    def update_tool_label_position(self):
        slot_geom = self.geometry()
        abs_x = slot_geom.x()
        abs_y = slot_geom.y()
        if self.rotation == 90:  # 플레이어1
            x = abs_x + slot_geom.width()
            y = abs_y + slot_geom.height() - self.tool_label.height()
        elif self.rotation == -90:  # 플레이어2
            x = abs_x - self.tool_label.width()
            y = abs_y
        else:
            x = abs_x + slot_geom.width()
            y = abs_y
        self.tool_label.move(x, y)

class PokemonBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pokemon TCG Board UI")
        self.setFixedSize(1280, 720)
        
        # 코인토스 관련 상태
        self.coin_toss_state = "waiting"  # waiting, choosing, tossing, finished
        self.first_player_choice = None
        self.second_player_choice = None
        self.coin_result = None
        self.coin_toss_animation_timer = QTimer()
        self.coin_toss_animation_timer.timeout.connect(self.update_coin_toss_animation)
        self.coin_toss_animation_frame = 0
        
        # 코인 애니메이션 관련 변수
        self.coin_heads_image = None
        self.coin_tails_image = None
        self.coin_current_image = None
        self.coin_position = QPointF(150, 100)  # 메시지 창 내부 중앙 위치로 변경
        self.coin_rotation = 0
        self.coin_scale = 1.0
        self.coin_opacity = 1.0
        self.load_coin_images()
        
        # 폰트 로드
        font_id = QFontDatabase.addApplicationFont(FONT_PATH)
        if font_id != -1:
            self.pokemon_font = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            self.pokemon_font = 'Arial'
            print("PF스타더스트 폰트 로드 실패, Arial 폰트로 대체됩니다.")
        gb_font_id = QFontDatabase.addApplicationFont(GB_FONT_PATH)
        if gb_font_id != -1:
            self.gb_font = QFontDatabase.applicationFontFamilies(gb_font_id)[0]
        else:
            self.gb_font = 'Arial'
            print("포켓몬GB 폰트 로드 실패, Arial 폰트로 대체됩니다.")
        self.setFont(QFont(self.pokemon_font, 10))
        
        # 턴 상태 관리
        self.current_turn = 1  # 1: 플레이어1 턴, 2: 플레이어2 턴
        
        # 배경 이미지와 투명도 관련 변수
        self._opacity = 0.85  # 기본 투명도 (0.0 ~ 1.0)
        self.bg_pixmap = QPixmap(BG_IMAGE_PATH).scaled(1280, 720, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.next_bg_pixmap = None
        
        # 페이드 애니메이션 설정
        self.fade_animation = QPropertyAnimation(self, b"opacity")
        self.fade_animation.setDuration(800)  # 800ms
        self.fade_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 색상 정의
        self.dark_blue = (10, 60, 90)
        self.dark_green = (30, 100, 30)
        self.light_blue = (160, 200, 240)
        self.white = (255, 255, 255)
        self.orange = (240, 180, 140)
        self.light_prize_blue = (180, 220, 255)  # 연한 프라이즈 파란색
        self.stadium_purple = (220, 200, 255)    # 스타디움 보라색
        
        # 컨트롤 박스 색상
        control_color = (220, 220, 220)
        self.control_left = CardSlot(control_color, self, rotation=90, font_family=self.pokemon_font)
        self.control_right = CardSlot(control_color, self, rotation=-90, font_family=self.pokemon_font)
        
        # 스타디움 변경 버튼
        self.stadium_button = QPushButton("스타디움 배경 변경", self)
        self.stadium_button.clicked.connect(self.change_stadium_background)
        self.stadium_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: black;
                border: 2px solid #2c3e50;
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c3e50;
            }
        """)
        
        # 프라이즈(왼쪽 위, 오른쪽 아래)
        self.prize_left = [CardSlot(self.light_prize_blue, self, rotation=90, font_family=self.pokemon_font) for _ in range(6)]
        self.prize_right = [CardSlot(self.light_prize_blue, self, rotation=-90, font_family=self.pokemon_font) for _ in range(6)]
        
        # 트래쉬/덱(좌하/우상)
        self.trash_left = CardSlot(self.dark_green, self, rotation=90, font_family=self.pokemon_font)
        self.deck_left = CardSlot(self.light_blue, self, rotation=90, font_family=self.pokemon_font)
        self.deck_right = CardSlot(self.light_blue, self, rotation=-90, font_family=self.pokemon_font)
        self.trash_right = CardSlot(self.dark_green, self, rotation=-90, font_family=self.pokemon_font)
        
        # 중앙 슬롯(흰색)
        self.center_white = [CardSlot(self.white, self, font_family=self.pokemon_font) for _ in range(13)]
        # 벤치와 액티브 포켓몬 슬롯에 회전 적용
        for i in range(5):  # 왼쪽 벤치
            self.center_white[i].rotation = 90
        for i in range(5, 10):  # 오른쪽 벤치
            self.center_white[i].rotation = -90
        self.center_white[10].rotation = 90  # 왼쪽 액티브
        self.center_white[11].rotation = -90  # 오른쪽 액티브
        # 스타디움은 회전하지 않음
        
        # 메시지 창 추가
        self.message_window = QLabel(self)
        self.message_window.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(0, 0, 0, 0.85);
                color: #FFD700;
                border: 3px solid #FFD700;
                border-radius: 15px;
                padding: 15px;
                font-family: '{self.gb_font}';
            }}
        """)
        self.message_window.setAlignment(Qt.AlignCenter)
        self.message_window.setFont(QFont(self.gb_font, 10, QFont.Bold))
        
        # 테스트 버튼 추가
        self.setup_test_button()
        
        # 초기 카드 배치
        self.update_card_display()

        self.slot_energy_state = {}  # {(player, slot_type): {energy_type: count}}
        self.energy_labels = {}      # {(player, slot_type): [QLabel, ...]}
        self.slot_hp_state = {}  # {(player, slot_type): {'base': 300, 'bonus': 0, 'damage': 0}}
        self.slot_status_state = {}  # {(player, slot_type): {'poison': bool, 'burn': bool}}
        self.slot_tool_state = {}  # {(player, slot_type): bool}

    def setup_test_button(self):
        """테스트용 버튼 설정"""
        # 공통 버튼 스타일
        button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: black;
                border: 2px solid #2c3e50;
                border-radius: 8px;
                padding: 1px;
                font-size: 11px;
                font-weight: bold;
                min-width: 47px;
                min-height: 47px;
                max-width: 47px;
                max-height: 47px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c3e50;
            }
        """
        
        # 코인토스 버튼 스타일
        coin_toss_button_style = """
            QPushButton {
                background-color: #f1c40f;
                color: black;
                border: 2px solid #2c3e50;
                border-radius: 8px;
                padding: 1px;
                font-size: 11px;
                font-weight: bold;
                min-width: 47px;
                min-height: 47px;
                max-width: 47px;
                max-height: 47px;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """
        
        # 플레이어1 컨트롤 박스 (왼쪽)
        left_layout = QGridLayout()
        left_layout.setSpacing(2)  # 버튼 간격 줄임
        left_layout.setContentsMargins(2, 2, 2, 2)  # 여백 줄임
        
        # 코인토스 버튼 추가 (플레이어1 시점 - 90도 회전)
        self.coin_heads_button1 = CustomRotatedButton("앞면", rotation=90, font_family=self.pokemon_font)
        self.coin_heads_button1.setStyleSheet(coin_toss_button_style)
        self.coin_heads_button1.clicked.connect(lambda: self.choose_coin_side(1, "heads"))
        self.coin_tails_button1 = CustomRotatedButton("뒷면", rotation=90, font_family=self.pokemon_font)
        self.coin_tails_button1.setStyleSheet(coin_toss_button_style)
        self.coin_tails_button1.clicked.connect(lambda: self.choose_coin_side(1, "tails"))
        
        left_layout.addWidget(self.coin_heads_button1, 0, 0)
        left_layout.addWidget(self.coin_tails_button1, 1, 0)  # 세로로 배치
        
        # 에너지 관리 버튼
        self.energy_button1 = CustomRotatedButton("에너지\n관리", rotation=90, font_family=self.pokemon_font)
        self.energy_button1.setStyleSheet(button_style)
        self.energy_button1.clicked.connect(lambda: self.manage_energy(1))
        left_layout.addWidget(self.energy_button1, 2, 0)
        # 체력 관리 버튼
        self.hp_button1 = CustomRotatedButton("체력\n관리", rotation=90, font_family=self.pokemon_font)
        self.hp_button1.setStyleSheet(button_style)
        self.hp_button1.clicked.connect(lambda: self.manage_hp(1))
        left_layout.addWidget(self.hp_button1, 2, 1)
        # 상태이상 관리 버튼
        self.status_button1 = CustomRotatedButton("상태이상\n관리", rotation=90, font_family=self.pokemon_font)
        self.status_button1.setStyleSheet(button_style)
        self.status_button1.clicked.connect(lambda: self.manage_status(1))
        left_layout.addWidget(self.status_button1, 3, 0)
        # 진/퇴화 관리 버튼
        self.evolution_button1 = CustomRotatedButton("진화\n관리", rotation=90, font_family=self.pokemon_font)
        self.evolution_button1.setStyleSheet(button_style)
        self.evolution_button1.clicked.connect(lambda: self.manage_evolution(1))
        left_layout.addWidget(self.evolution_button1, 3, 1)
        # 포지션 관리 버튼
        self.position_button1 = CustomRotatedButton("포지션\n관리", rotation=90, font_family=self.pokemon_font)
        self.position_button1.setStyleSheet(button_style)
        self.position_button1.clicked.connect(lambda: self.manage_position(1))
        left_layout.addWidget(self.position_button1, 4, 0)
        # 트레이너 관리 버튼
        self.trainer_button1 = CustomRotatedButton("스타디움\n도구\n관리", rotation=90, font_family=self.pokemon_font)
        self.trainer_button1.setStyleSheet(button_style)
        self.trainer_button1.clicked.connect(lambda: self.manage_trainer(1))
        left_layout.addWidget(self.trainer_button1, 4, 1)
        # 포켓몬 체크 버튼
        self.pokemon_check_button1 = CustomRotatedButton("포켓몬\n체크", rotation=90, font_family=self.pokemon_font)
        self.pokemon_check_button1.setStyleSheet(button_style)
        self.pokemon_check_button1.clicked.connect(lambda: self.check_pokemon(1))
        left_layout.addWidget(self.pokemon_check_button1, 5, 1)
        # 턴 종료 버튼 (특별 스타일)
        self.end_turn_button1 = CustomRotatedButton("턴 종료", rotation=90, font_family=self.pokemon_font)
        self.end_turn_button1.setStyleSheet(button_style + """
            QPushButton {
                background-color: #e74c3c;
                font-size: 11px;
                min-height: 47px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.end_turn_button1.clicked.connect(lambda: self.end_turn(1))
        left_layout.addWidget(self.end_turn_button1, 5, 0)
        self.control_left.setLayout(left_layout)
        # 플레이어2 컨트롤 박스 (오른쪽)
        right_layout = QGridLayout()
        right_layout.setSpacing(2)  # 버튼 간격 줄임
        right_layout.setContentsMargins(2, 2, 2, 2)  # 여백 줄임
        
        # 코인토스 버튼 추가 (플레이어2 시점 - -90도 회전)
        self.coin_heads_button2 = CustomRotatedButton("앞면", rotation=-90, font_family=self.pokemon_font)
        self.coin_heads_button2.setStyleSheet(coin_toss_button_style)
        self.coin_heads_button2.clicked.connect(lambda: self.choose_coin_side(2, "heads"))
        self.coin_tails_button2 = CustomRotatedButton("뒷면", rotation=-90, font_family=self.pokemon_font)
        self.coin_tails_button2.setStyleSheet(coin_toss_button_style)
        self.coin_tails_button2.clicked.connect(lambda: self.choose_coin_side(2, "tails"))
        
        right_layout.addWidget(self.coin_tails_button2, 0, 0)  # 뒷면을 위에 배치
        right_layout.addWidget(self.coin_heads_button2, 1, 0)  # 앞면을 아래에 배치
        
        # 에너지 관리 버튼
        self.energy_button2 = CustomRotatedButton("에너지\n관리", rotation=-90, font_family=self.pokemon_font)
        self.energy_button2.setStyleSheet(button_style)
        self.energy_button2.clicked.connect(lambda: self.manage_energy(2))
        right_layout.addWidget(self.energy_button2, 5, 1)
        # 체력 관리 버튼
        self.hp_button2 = CustomRotatedButton("체력\n관리", rotation=-90, font_family=self.pokemon_font)
        self.hp_button2.setStyleSheet(button_style)
        self.hp_button2.clicked.connect(lambda: self.manage_hp(2))
        right_layout.addWidget(self.hp_button2, 5, 0)
        # 상태이상 관리 버튼
        self.status_button2 = CustomRotatedButton("상태이상\n관리", rotation=-90, font_family=self.pokemon_font)
        self.status_button2.setStyleSheet(button_style)
        self.status_button2.clicked.connect(lambda: self.manage_status(2))
        right_layout.addWidget(self.status_button2, 4, 1)
        # 진/퇴화 관리 버튼
        self.evolution_button2 = CustomRotatedButton("진화\n관리", rotation=-90, font_family=self.pokemon_font)
        self.evolution_button2.setStyleSheet(button_style)
        self.evolution_button2.clicked.connect(lambda: self.manage_evolution(2))
        right_layout.addWidget(self.evolution_button2, 4, 0)
        # 포지션 관리 버튼
        self.position_button2 = CustomRotatedButton("포지션\n관리", rotation=-90, font_family=self.pokemon_font)
        self.position_button2.setStyleSheet(button_style)
        self.position_button2.clicked.connect(lambda: self.manage_position(2))
        right_layout.addWidget(self.position_button2, 3, 1)
        # 트레이너 관리 버튼
        self.trainer_button2 = CustomRotatedButton("스타디움\n도구\n관리", rotation=-90, font_family=self.pokemon_font)
        self.trainer_button2.setStyleSheet(button_style)
        self.trainer_button2.clicked.connect(lambda: self.manage_trainer(2))
        right_layout.addWidget(self.trainer_button2, 3, 0)
        # 턴 종료 버튼 (특별 스타일)
        self.end_turn_button2 = CustomRotatedButton("턴 종료", rotation=-90, font_family=self.pokemon_font)
        self.end_turn_button2.setStyleSheet(button_style + """
            QPushButton {
                background-color: #e74c3c;
                font-size: 11px;
                min-height: 47px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.end_turn_button2.clicked.connect(lambda: self.end_turn(2))
        right_layout.addWidget(self.end_turn_button2, 2, 1)
        # 포켓몬 체크 버튼
        self.pokemon_check_button2 = CustomRotatedButton("포켓몬\n체크", rotation=-90, font_family=self.pokemon_font)
        self.pokemon_check_button2.setStyleSheet(button_style)
        self.pokemon_check_button2.clicked.connect(lambda: self.check_pokemon(2))
        right_layout.addWidget(self.pokemon_check_button2, 2, 0)
        self.control_right.setLayout(right_layout)
        # 초기 턴 상태에 따라 버튼 표시/숨김
        self.update_turn_buttons()

    def update_turn_buttons(self):
        """턴 상태에 따라 버튼 표시/숨김"""
        # 코인토스 버튼 표시/숨김
        is_coin_toss = self.coin_toss_state == "waiting"
        self.coin_heads_button1.setVisible(is_coin_toss)
        self.coin_tails_button1.setVisible(is_coin_toss)
        self.coin_heads_button2.setVisible(is_coin_toss)
        self.coin_tails_button2.setVisible(is_coin_toss)
        
        # 기존 버튼들 표시/숨김
        is_game_started = self.coin_toss_state == "finished"
        self.energy_button1.setVisible(is_game_started and self.current_turn == 1)
        self.hp_button1.setVisible(is_game_started and self.current_turn == 1)
        self.status_button1.setVisible(is_game_started and self.current_turn == 1)
        self.evolution_button1.setVisible(is_game_started and self.current_turn == 1)
        self.position_button1.setVisible(is_game_started and self.current_turn == 1)
        self.trainer_button1.setVisible(is_game_started and self.current_turn == 1)
        self.pokemon_check_button1.setVisible(is_game_started and self.current_turn == 1)
        self.end_turn_button1.setVisible(is_game_started and self.current_turn == 1)
        
        self.energy_button2.setVisible(is_game_started and self.current_turn == 2)
        self.hp_button2.setVisible(is_game_started and self.current_turn == 2)
        self.status_button2.setVisible(is_game_started and self.current_turn == 2)
        self.evolution_button2.setVisible(is_game_started and self.current_turn == 2)
        self.position_button2.setVisible(is_game_started and self.current_turn == 2)
        self.trainer_button2.setVisible(is_game_started and self.current_turn == 2)
        self.pokemon_check_button2.setVisible(is_game_started and self.current_turn == 2)
        self.end_turn_button2.setVisible(is_game_started and self.current_turn == 2)
        
        # 메시지 창 업데이트
        if is_game_started:
            turn_text = "◀ Player 1's Turn!" if self.current_turn == 1 else "Player 2's Turn! ▶"
            self.message_window.setText(f"{turn_text}")
        else:
            self.message_window.setText("코인토스로 선공을 정하세요!\n\n한 플레이어가 먼저 선택하면\n다른 플레이어의 면은\n자동으로 선택됩니다.")

    def end_turn(self, player):
        """턴 종료 처리"""
        if player == self.current_turn:
            self.current_turn = 2 if player == 1 else 1
            self.update_turn_buttons()
            print(f"플레이어{player}의 턴이 종료되었습니다.")

    def paintEvent(self, event):
        painter = QPainter(self)
        # 1. 배경 이미지 (항상 맨 아래)
        painter.setOpacity(self._opacity)
        painter.drawPixmap(self.rect(), self.bg_pixmap)
        # 2. 오라(상태이상) 효과 (배경 위, 슬롯 아래)
        for slot in getattr(self, 'center_white', []):
            if hasattr(slot, 'status_dict') and slot.status_dict:
                slot_rect = slot.geometry()
                cx = slot_rect.center().x()
                cy = slot_rect.center().y()
                w = slot_rect.width()
                h = slot_rect.height()
                d = math.sqrt(w**2 + h**2)  # 대각선 길이
                # 1. 화상 오라 먼저 (대각선의 1.15배)
                if slot.status_dict.get('burn'):
                    painter.setBrush(QColor(231, 76, 60, 180))
                    painter.setPen(Qt.NoPen)
                    burn_d = d * 1.15
                    painter.drawEllipse(int(cx - burn_d/2), int(cy - burn_d/2), int(burn_d), int(burn_d))
                # 2. 독 오라 (대각선 길이)
                if slot.status_dict.get('poison'):
                    painter.setBrush(QColor(108, 52, 131, 180))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(int(cx - d/2), int(cy - d/2), int(d), int(d))
        # 3. 나머지(코인, 메시지창 등 기존 코드)
        if self.coin_toss_state == "tossing" and self.coin_current_image:
            msg_rect = self.message_window.geometry()
            coin_x = int(msg_rect.x() + self.coin_position.x())
            coin_y = int(msg_rect.y() + self.coin_position.y())
            shadow_painter = QPainter(self)
            shadow_painter.setOpacity(self.coin_opacity * 0.3)
            shadow_painter.setBrush(QColor(0, 0, 0, 100))
            shadow_painter.setPen(Qt.NoPen)
            shadow_painter.drawEllipse(coin_x - 60, coin_y - 60, 120, 120)
            painter.setOpacity(self.coin_opacity)
            coin_size = int(100 * self.coin_scale)
            coin_rect = self.coin_current_image.rect()
            coin_rect.setSize(QSize(coin_size, coin_size))
            coin_rect.moveCenter(QPoint(coin_x, coin_y))
            transform = QTransform()
            transform.translate(coin_x, coin_y)
            transform.rotate(self.coin_rotation)
            transform.translate(-coin_x, -coin_y)
            painter.setTransform(transform)
            painter.drawPixmap(coin_rect, self.coin_current_image)
            self.message_window.setVisible(False)
        elif self.coin_toss_state == "finished":
            self.message_window.setVisible(True)

    @pyqtProperty(float)
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = value
        self.update()

    def change_stadium_background(self):
        """스타디움 배경 이미지를 레어도를 고려하여 랜덤으로 변경"""
        try:
            # 레어도별 이미지 파일 목록 수집
            rarity_images = {rarity: [] for rarity in RARITY_WEIGHTS.keys()}
            
            # 각 레어도 폴더에서 bg_로 시작하는 이미지 파일 수집
            for rarity in rarity_images.keys():
                rarity_dir = os.path.join(STADIUM_IMAGE_DIR, rarity)
                if os.path.exists(rarity_dir):
                    images = [f for f in os.listdir(rarity_dir) if f.startswith('bg_')]
                    rarity_images[rarity] = [(os.path.join(rarity_dir, img), rarity) for img in images]
            
            # 모든 이미지와 레어도 정보를 하나의 리스트로 합치기
            all_images = []
            for images in rarity_images.values():
                all_images.extend(images)
            
            if all_images:
                # 레어도 가중치를 적용하여 이미지 선택
                weights = [RARITY_WEIGHTS[rarity] for _, rarity in all_images]
                selected_image, selected_rarity = random.choices(all_images, weights=weights, k=1)[0]
                
                # 새로운 배경 이미지 준비
                self.next_bg_pixmap = QPixmap(selected_image).scaled(
                    1280, 720,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )
                
                # 페이드 아웃 애니메이션 설정
                self.fade_animation.setStartValue(0.85)
                self.fade_animation.setEndValue(0.0)
                self.fade_animation.finished.connect(self.apply_new_background)
                self.fade_animation.start()
                
                print(f"배경 이미지가 {os.path.basename(selected_image)} ({selected_rarity})로 변경됩니다.")
            else:
                print("스타디움 배경 이미지를 찾을 수 없습니다.")
        except Exception as e:
            print(f"배경 이미지 변경 중 오류 발생: {e}")

    def apply_new_background(self):
        """새로운 배경 이미지 적용 및 페이드 인"""
        if self.next_bg_pixmap:
            self.bg_pixmap = self.next_bg_pixmap
            self.next_bg_pixmap = None
            
            # 페이드 인 애니메이션 설정
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(0.85)
            self.fade_animation.finished.disconnect()
            self.fade_animation.start()

    def update_card_display(self):
        """카드 표시 업데이트"""
        # 덱 카드 수 표시
        self.deck_left.set_card("덱")
        self.deck_right.set_card("덱")
        
        # 트래시 카드 수 표시
        self.trash_left.set_card("트래시")
        self.trash_right.set_card("트래시")
        
        # 프라이즈 카드 개별 표시
        for i, slot in enumerate(self.prize_left):
            slot.set_card(f"프라이즈 {i+1}")
        for i, slot in enumerate(self.prize_right):
            slot.set_card(f"프라이즈 {6-i}")  # 플레이어2 기준으로 역순으로 표시
            
        # 벤치 슬롯 텍스트 표시 및 체력 초기화
        for i in range(5):
            self.center_white[i].set_card(f"벤치 {i+1}")      # 왼쪽 벤치
            self.center_white[i].clear_hp()                   # set_hp(0) 대신 clear_hp()
            self.center_white[5+i].set_card(f"벤치 {5-i}")    # 오른쪽 벤치 (역순)
            self.center_white[5+i].clear_hp()
            
        # 액티브 칸 텍스트 표시 (플레이어 시선에 맞춰 회전) 및 체력 초기화
        self.center_white[10].set_card("배틀필드")  # 왼쪽 액티브 (90도 회전)
        self.center_white[10].clear_hp()
        self.center_white[11].set_card("배틀필드")  # 오른쪽 액티브 (-90도 회전)
        self.center_white[11].clear_hp()

    def resizeEvent(self, event):
        # 픽셀 기준 고정 배치
        slot_w, slot_h = 120, 90
        gap = 18
        # 프라이즈(왼쪽 위, 2x3) - 간격 없이 붙임
        for i in range(2):
            for j in range(3):
                idx = i*3 + j
                x = i*slot_w + gap  # gap은 바깥쪽 여백만 적용
                y = j*slot_h + gap
                self.prize_left[idx].setGeometry(x, y, slot_w, slot_h)
        # 프라이즈(오른쪽 아래, 2x3) - 간격 없이 붙임
        for i in range(2):
            for j in range(3):
                idx = i*3 + j
                x = 1280 - 2*slot_w - gap + i*slot_w
                y = 720 - 3*slot_h - gap + j*slot_h
                self.prize_right[idx].setGeometry(x, y, slot_w, slot_h)
        # 트래쉬/덱(좌하)
        self.trash_left.setGeometry(gap, 720-slot_h-gap, slot_w, slot_h)
        self.deck_left.setGeometry(gap+slot_w+gap, 720-slot_h-gap, slot_w, slot_h)
        # 트래쉬/덱(우상)
        self.deck_right.setGeometry(1280-gap-slot_w*2-gap, gap, slot_w, slot_h)
        self.trash_right.setGeometry(1280-gap-slot_w, gap, slot_w, slot_h)
        # 액티브 칸(정사각형, 창의 가운데, 좌우 대칭, 간격 더 넓힘)
        active_size = 120
        active_gap = 60
        center_y = 720//2 - active_size//2
        left_active_x = 1280//2 - active_size - active_gap//2
        right_active_x = 1280//2 + active_gap//2
        self.center_white[10].setGeometry(left_active_x, center_y, active_size, active_size)  # 왼쪽 액티브
        self.center_white[11].setGeometry(right_active_x, center_y, active_size, active_size)  # 오른쪽 액티브
        # 벤치카드(왼쪽/오른쪽 세로 5개)
        bench_count = 5
        bench_gap = 18
        center_line = 720 // 2
        for i in range(bench_count):
            # 가운데 슬롯의 중심이 창의 중심선에 오도록
            y_center = center_line + (i-2)*(slot_h+bench_gap)
            y = y_center - slot_h//2
            self.center_white[i].setGeometry(left_active_x-175, y, slot_w, slot_h)  # 왼쪽 벤치
            self.center_white[5+i].setGeometry(right_active_x+175, y, slot_w, slot_h)  # 오른쪽 벤치
        # 스타디움 칸(중앙 하단)
        self.center_white[12].setGeometry(580, center_line+100, slot_w, slot_h)
        self.center_white[12].setStyleSheet(f"background-color: rgb{self.stadium_purple}; border: 6px solid black; border-radius: 8px;")
        
        # 메시지 창 배치 (상단 중앙)
        msg_width = 300
        msg_height = 200
        self.message_window.setGeometry(
            (1280 - msg_width) // 2,  # 중앙 정렬
            20,  # 상단에서 20px 떨어진 위치
            msg_width,
            msg_height
        )
        
        # 코인 위치 업데이트 (메시지 창 중앙)
        self.coin_position = QPointF(msg_width // 2, msg_height // 2)
        
        # 컨트롤 박스 배치(왼쪽/오른쪽, 2x3 크기)
        # 왼쪽: 프라이즈 아래, 덱/트래쉬 위 (2x3 크기)
        control_left_x = gap
        control_left_y = gap + 3*slot_h + 20  # 프라이즈 3개 아래 + 여유
        self.control_left.setGeometry(control_left_x, control_left_y, slot_w*2, slot_h*3)
        # 오른쪽: 프라이즈 위, 덱/트래쉬 아래 (2x3 크기)
        control_right_x = 1280 - 3*slot_w - gap + slot_w  # 프라이즈 오른쪽 끝 x
        control_right_y = 720 - 6*slot_h - gap - 20  # 프라이즈 3개 + 컨트롤 박스 1개 위 + 여유
        self.control_right.setGeometry(control_right_x, control_right_y, slot_w*2, slot_h*3)
        
        # 스타디움 변경 버튼 배치 (하단 중앙)
        button_width = 200
        button_height = 40
        self.stadium_button.setGeometry(
            (1280 - button_width) // 2,  # 중앙 정렬
            720 - button_height - 20,    # 하단에서 20px 떨어진 위치
            button_width,
            button_height
        )

        # 각 슬롯의 hp_label 위치 재조정
        for slot in self.center_white:
            slot.update_hp_label_position()

    # 버튼 클릭 이벤트 핸들러들 (나중에 구현)
    def manage_energy(self, player):
        msg = f"플레이어 {player}\n에너지 관리 버튼을 눌렀습니다."
        self.message_window.setText(msg)
        QApplication.processEvents()
        dialog = EnergyManagementDialog(self, current_player=player, slot_energy_state=self.slot_energy_state)
        dialog.energy_selected.connect(self.handle_energy_selection)
        dialog.exec_()
    
    def handle_energy_selection(self, player_number, slot_type, energy_dict):
        msg = f"플레이어 {player_number}<br>{self.get_slot_kor(slot_type)} 슬롯이<br>"
        energy_dir = os.path.join(os.path.dirname(__file__), '에너지_이미지')
        energy_msgs = []
        for energy_type, count in energy_dict.items():
            if count > 0:
                energy_path = os.path.join(energy_dir, f'{energy_type}_energy.jpg')
                energy_msgs.append(f'<img src="{energy_path}" width="24" height="24">에너지 {count}개')
        msg += '<br>'.join(energy_msgs) + "로 설정되었습니다."
        self.message_window.setText(msg)
        print(msg)
        # 슬롯별 에너지 상태 저장
        self.slot_energy_state[(player_number, slot_type)] = dict(energy_dict)
        # 기존 에너지 이미지 제거
        key = (player_number, slot_type)
        if key in self.energy_labels:
            for label in self.energy_labels[key]:
                label.deleteLater()
            self.energy_labels[key] = []
        else:
            self.energy_labels[key] = []
        # 에너지 이미지 생성 및 배치 (슬롯 기준)
        slot = None
        if player_number == 1:
            if slot_type == "active":
                slot = self.center_white[10]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[bench_index]
        else:
            if slot_type == "active":
                slot = self.center_white[11]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                # 버튼 1→슬롯5, 2→4, ... 5→1로 매핑
                slot = self.center_white[5 + (4 - bench_index)]
        if slot:
            slot_rect = slot.geometry()
            img_size = 24
            if slot_type == "active":
                # 배틀필드: 1줄, 가운데 정렬
                total = sum(energy_dict.values())
                total_height = total * img_size
                if player_number == 1:
                    x = slot_rect.x() - img_size - 2
                    y0 = slot_rect.y() + (slot_rect.height() - total_height) // 2
                    n = 0
                    for energy_type, count in energy_dict.items():
                        energy_path = os.path.join(energy_dir, f'{energy_type}_energy.jpg')
                        for i in range(count):
                            y = y0 + n * img_size
                            energy_label = QLabel(self)
                            pixmap = QPixmap(energy_path).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            transform = QTransform().rotate(90)
                            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
                            energy_label.setPixmap(rotated_pixmap)
                            energy_label.setGeometry(x, y, img_size, img_size)
                            energy_label.show()
                            self.energy_labels[key].append(energy_label)
                            n += 1
                else:
                    x = slot_rect.x() + slot_rect.width() + 2
                    y0 = slot_rect.y() + slot_rect.height() - ((slot_rect.height() - total_height) // 2) - img_size
                    n = 0
                    for energy_type, count in energy_dict.items():
                        energy_path = os.path.join(energy_dir, f'{energy_type}_energy.jpg')
                        for i in range(count):
                            y = y0 - n * img_size
                            energy_label = QLabel(self)
                            pixmap = QPixmap(energy_path).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            transform = QTransform().rotate(-90)
                            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
                            energy_label.setPixmap(rotated_pixmap)
                            energy_label.setGeometry(x, y, img_size, img_size)
                            energy_label.show()
                            self.energy_labels[key].append(energy_label)
                            n += 1
            else:
                # 벤치: 2줄, 4개씩, 새로운 줄이 아래에 생김(순서 반전 없음)
                max_per_col = 4
                n = 0
                if player_number == 1:
                    x0 = slot_rect.x() - img_size - 2
                    y0 = slot_rect.y()
                    for energy_type, count in energy_dict.items():
                        for i in range(count):
                            col = n // max_per_col
                            row = n % max_per_col
                            x = x0 - col * (img_size + 2)
                            y = y0 + row * img_size
                            energy_label = QLabel(self)
                            pixmap = QPixmap(os.path.join(energy_dir, f'{energy_type}_energy.jpg')).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            transform = QTransform().rotate(90)
                            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
                            energy_label.setPixmap(rotated_pixmap)
                            energy_label.setGeometry(x, y, img_size, img_size)
                            energy_label.show()
                            self.energy_labels[key].append(energy_label)
                            n += 1
                else:
                    x0 = slot_rect.x() + slot_rect.width() + 2
                    y0 = slot_rect.y() + slot_rect.height() - img_size
                    for energy_type, count in energy_dict.items():
                        for i in range(count):
                            col = n // max_per_col
                            row = n % max_per_col
                            x = x0 + col * (img_size + 2)
                            y = y0 - row * img_size
                            energy_label = QLabel(self)
                            pixmap = QPixmap(os.path.join(energy_dir, f'{energy_type}_energy.jpg')).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            transform = QTransform().rotate(-90)
                            rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
                            energy_label.setPixmap(rotated_pixmap)
                            energy_label.setGeometry(x, y, img_size, img_size)
                            energy_label.show()
                            self.energy_labels[key].append(energy_label)
                            n += 1

    def manage_hp(self, player):
        msg = f"플레이어 {player}\n체력 관리 버튼을 눌렀습니다."
        self.message_window.setText(msg)
        QApplication.processEvents()
        dialog = HPManagementDialog(self, current_player=player)
        dialog.hp_updated.connect(self.handle_hp_update)
        dialog.exec_()
    
    def clear_slot_state(self, player_number, slot_type):
        key = (player_number, slot_type)
        # 체력
        if key in self.slot_hp_state:
            del self.slot_hp_state[key]
        # 상태이상
        if key in self.slot_status_state:
            del self.slot_status_state[key]
        # 에너지
        if key in self.slot_energy_state:
            del self.slot_energy_state[key]
        # 도구
        if key in self.slot_tool_state:
            del self.slot_tool_state[key]
        # 카드 슬롯 자체도 초기화
        slot = None
        if player_number == 1:
            if slot_type == "active":
                slot = self.center_white[10]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[bench_index]
        else:
            if slot_type == "active":
                slot = self.center_white[11]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[5 + (4 - bench_index)]
        if slot:
            slot.clear_hp()
            slot.set_status_text("")
            slot.set_tool_icon(None)
            slot.status_dict = {}  # 오라 효과도 완전히 제거
            slot.set_card(self.get_slot_kor(slot_type))  # 기본 슬롯명 복구
            # 에너지 이미지도 삭제
            if key in self.energy_labels:
                for label in self.energy_labels[key]:
                    label.deleteLater()
                self.energy_labels[key] = []
    
    def handle_hp_update(self, player_number, slot_type, bonus_hp, damage):
        """체력 업데이트 처리"""
        # 체력 상태 저장
        self.slot_hp_state[(player_number, slot_type)] = {
            'base': 300,  # 기본 체력
            'bonus': bonus_hp,
            'damage': damage
        }
        # 현재 체력 계산
        current_hp = 300 + bonus_hp - damage
        # 해당 슬롯의 체력 표시 업데이트
        slot = None
        if player_number == 1:
            if slot_type == "active":
                slot = self.center_white[10]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[bench_index]
        else:
            if slot_type == "active":
                slot = self.center_white[11]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[5 + (4 - bench_index)]
        if slot:
            slot.set_hp(current_hp)
            slot.update_hp_label_position()  # 위치 갱신 추가
        # 메시지 창 업데이트 및 KO 처리
        if current_hp <= 0:
            self.clear_slot_state(player_number, slot_type)
            self.message_window.setText(f"플레이어 {player_number}의 {self.get_slot_kor(slot_type)} 슬롯\n 포켓몬이 기절했습니다!")
        else:
            self.message_window.setText(
                f"플레이어 {player_number}\n"
                f"{self.get_slot_kor(slot_type)} 슬롯의\n"
                f"현재 체력이 {current_hp}이 되었습니다.\n"
                f"추가 체력: {bonus_hp}\n누적 피해: {damage}"
            )

    def manage_status(self, player):
        msg = f"플레이어 {player}\n상태이상 관리 버튼을 눌렀습니다."
        self.message_window.setText(msg)
        QApplication.processEvents()
        dialog = StatusManagementDialog(self, current_player=player, slot_status_state=self.slot_status_state)
        dialog.status_selected.connect(self.handle_status_apply)
        dialog.status_removed.connect(self.handle_status_remove)
        dialog.exec_()

    def handle_status_apply(self, player_number, slot_type, status):
        key = (player_number, slot_type)
        if key not in self.slot_status_state:
            self.slot_status_state[key] = {}
        # 수면/마비/혼란 중 하나만 허용
        status_group = {"sleep", "paralysis", "confusion"}
        if status in status_group:
            # 기존 셋 중 하나라도 True면 모두 False로
            for s in status_group:
                self.slot_status_state[key][s] = False
                self.slot_status_state[key][status] = True
        else:
            self.slot_status_state[key][status] = True

        slot = None
        if player_number == 1 and slot_type == "active":
            slot = self.center_white[10]
        elif player_number == 2 and slot_type == "active":
            slot = self.center_white[11]
        if slot:
            slot.status_dict = self.slot_status_state[key]
            # 수면/마비/혼란만 표시
            status_text = ""
            for s in ["sleep", "paralysis", "confusion"]:
                if self.slot_status_state[key].get(s):
                    status_text = self.get_status_kor(s)
                    break
            slot.set_status_text(status_text)
            slot.update()
        self.message_window.setText(f"플레이어 {player_number}의 {self.get_slot_kor(slot_type)} 슬롯\n {self.get_status_kor(status)} 상태이상이 적용되었습니다.")

    def handle_status_remove(self, player_number, slot_type, status):
        key = (player_number, slot_type)
        if key in self.slot_status_state and status in self.slot_status_state[key]:
            self.slot_status_state[key][status] = False
        slot = None
        if player_number == 1 and slot_type == "active":
            slot = self.center_white[10]
        elif player_number == 2 and slot_type == "active":
            slot = self.center_white[11]
        if slot:
            slot.status_dict = self.slot_status_state[key]
            # 수면/마비/혼란만 표시
            status_text = ""
            for s in ["sleep", "paralysis", "confusion"]:
                if self.slot_status_state[key].get(s):
                    status_text = self.get_status_kor(s)
                    break
            slot.set_status_text(status_text)
            slot.update()
        self.message_window.setText(f"플레이어 {player_number}의 {self.get_slot_kor(slot_type)} 슬롯\n{self.get_status_kor(status)} 상태이상이 해제되었습니다.")

    def get_status_kor(self, status):
        return {"poison": "독", "burn": "화상", "sleep": "수면", "paralysis": "마비", "confusion": "혼란"}.get(status, status)

    def manage_evolution(self, player):
        msg = f"플레이어 {player}\n진화 관리 버튼을 눌렀습니다."
        self.message_window.setText(msg)
        QApplication.processEvents()
        dialog = EvolutionManagementDialog(self, current_player=player)
        dialog.slot_selected.connect(lambda p, s: self.handle_evolution_slot_selection(p, s))
        dialog.evolution_completed.connect(self.handle_evolution_completed)
        dialog.exec_()
    
    def handle_evolution_slot_selection(self, player_number, slot_type):
        msg = f"플레이어 {player_number}\n{self.get_slot_kor(slot_type)} 슬롯\n(진화)이 선택되었습니다."
        self.message_window.setText(msg)
        print(msg)
        # TODO: 진화 관리 로직 구현

    def handle_evolution_completed(self, player_number, slot_type):
        slot_kor = self.get_slot_kor(slot_type)
        self.message_window.setText(f"플레이어 {player_number}의 {slot_kor} 슬롯\n포켓몬이 진화했습니다!")

    def manage_position(self, player):
        msg = f"플레이어 {player}\n포지션 관리 버튼을 눌렀습니다."
        self.message_window.setText(msg)
        QApplication.processEvents()
        dialog = PositionManagementDialog(self, current_player=player)
        dialog.position_swapped.connect(self.handle_position_swap)
        dialog.exec_()

    def handle_position_swap(self, player_number, slot1_type, slot2_type):
        slot1 = None
        slot2 = None
        if player_number == 1:
            if slot1_type == "active":
                slot1 = self.center_white[10]
            elif slot1_type.startswith("bench_"):
                bench_index = int(slot1_type.split('_')[1])
                slot1 = self.center_white[bench_index]
            if slot2_type == "active":
                slot2 = self.center_white[10]
            elif slot2_type.startswith("bench_"):
                bench_index = int(slot2_type.split('_')[1])
                slot2 = self.center_white[bench_index]
        else:
            if slot1_type == "active":
                slot1 = self.center_white[11]
            elif slot1_type.startswith("bench_"):
                bench_index = int(slot1_type.split('_')[1])
                slot1 = self.center_white[5 + (4 - bench_index)]
            if slot2_type == "active":
                slot2 = self.center_white[11]
            elif slot2_type.startswith("bench_"):
                bench_index = int(slot2_type.split('_')[1])
                slot2 = self.center_white[5 + (4 - bench_index)]
        if slot1 and slot2:
            slot1_card = slot1.card_name
            slot2_card = slot2.card_name
            slot1.card_name, slot2.card_name = slot2_card, slot1_card
            slot1.setText(slot1.text())
            slot2.setText(slot2.text())
            slot1_hp = slot1.hp
            slot2_hp = slot2.hp
            slot1.set_hp(slot2_hp)
            slot2.set_hp(slot1_hp)
            key1 = (player_number, slot1_type)
            key2 = (player_number, slot2_type)
            # 항상 (bool, icon_path) 튜플로 변환
            def to_tuple(val):
                if isinstance(val, tuple):
                    return val
                elif isinstance(val, bool):
                    return (val, None)
                return (False, None)
            tool1 = to_tuple(self.slot_tool_state.get(key1, (False, None)))
            tool2 = to_tuple(self.slot_tool_state.get(key2, (False, None)))
            self.slot_tool_state[key1], self.slot_tool_state[key2] = tool2, tool1
            slot1.set_tool_icon(tool2[1] if tool2[0] else None)
            slot2.set_tool_icon(tool1[1] if tool1[0] else None)
            slot1.status_dict = {}
            slot2.status_dict = {}
            slot1.set_status_text("")
            slot2.set_status_text("")
            if key1 in self.slot_status_state:
                del self.slot_status_state[key1]
            if key2 in self.slot_status_state:
                del self.slot_status_state[key2]
            if key1 in self.slot_energy_state or key2 in self.slot_energy_state:
                self.slot_energy_state[key1], self.slot_energy_state[key2] = \
                    self.slot_energy_state.get(key2, {}), self.slot_energy_state.get(key1, {})
            if key1 in self.slot_hp_state or key2 in self.slot_hp_state:
                self.slot_hp_state[key1], self.slot_hp_state[key2] = \
                    self.slot_hp_state.get(key2, {}), self.slot_hp_state.get(key1, {})
            if key1 in self.energy_labels or key2 in self.energy_labels:
                self.energy_labels[key1], self.energy_labels[key2] = \
                    self.energy_labels.get(key2, []), self.energy_labels.get(key1, [])
                self.handle_energy_selection(player_number, slot1_type, self.slot_energy_state.get(key1, {}))
                self.handle_energy_selection(player_number, slot2_type, self.slot_energy_state.get(key2, {}))
            self.message_window.setText(
                f"플레이어 {player_number}\n"
                f"{self.get_slot_kor(slot1_type)}와 {self.get_slot_kor(slot2_type)}의\n\n"
                f"포지션이 교체되었습니다.\n"
                f"상태이상이 초기화됩니다."
            )

    def manage_trainer(self, player):
        """스타디움/도구 관리 버튼 클릭 처리"""
        # 모든 버튼 숨기기
        self.energy_button1.setVisible(False)
        self.hp_button1.setVisible(False)
        self.status_button1.setVisible(False)
        self.evolution_button1.setVisible(False)
        self.position_button1.setVisible(False)
        self.trainer_button1.setVisible(False)
        self.pokemon_check_button1.setVisible(False)
        self.end_turn_button1.setVisible(False)
        
        self.energy_button2.setVisible(False)
        self.hp_button2.setVisible(False)
        self.status_button2.setVisible(False)
        self.evolution_button2.setVisible(False)
        self.position_button2.setVisible(False)
        self.trainer_button2.setVisible(False)
        self.pokemon_check_button2.setVisible(False)
        self.end_turn_button2.setVisible(False)
        
        # 스타디움, 도구, 이전으로 버튼 표시
        if player == 1:
            # 스타디움 버튼 추가
            self.stadium_button1 = CustomRotatedButton("스타디움", rotation=90, font_family=self.pokemon_font)
            self.stadium_button1.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
                QPushButton:pressed {
                    background-color: #6c3483;
                }
            """)
            self.stadium_button1.clicked.connect(lambda: self.show_stadium_dialog(1))
            self.control_left.layout().addWidget(self.stadium_button1, 0, 0)
            
            # 도구 버튼 추가
            self.tool_button1 = CustomRotatedButton("도구", rotation=90, font_family=self.pokemon_font)
            self.tool_button1.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1f618d;
                }
            """)
            self.tool_button1.clicked.connect(lambda: self.show_tool_dialog(1))
            self.control_left.layout().addWidget(self.tool_button1, 1, 0)
            
            # 이전으로 버튼 추가
            self.back_button1 = CustomRotatedButton("이전으로", rotation=90, font_family=self.pokemon_font)
            self.back_button1.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            self.back_button1.clicked.connect(lambda: self.back_to_normal_buttons(1))
            self.control_left.layout().addWidget(self.back_button1, 2, 0)
        else:
            # 이전으로 버튼 추가
            self.back_button2 = CustomRotatedButton("이전으로", rotation=-90, font_family=self.pokemon_font)
            self.back_button2.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            self.back_button2.clicked.connect(lambda: self.back_to_normal_buttons(2))
            self.control_right.layout().addWidget(self.back_button2, 0, 0)
            
            # 도구 버튼 추가
            self.tool_button2 = CustomRotatedButton("도구", rotation=-90, font_family=self.pokemon_font)
            self.tool_button2.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1f618d;
                }
            """)
            self.tool_button2.clicked.connect(lambda: self.show_tool_dialog(2))
            self.control_right.layout().addWidget(self.tool_button2, 1, 0)
            
            # 스타디움 버튼 추가
            self.stadium_button2 = CustomRotatedButton("스타디움", rotation=-90, font_family=self.pokemon_font)
            self.stadium_button2.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
                QPushButton:pressed {
                    background-color: #6c3483;
                }
            """)
            self.stadium_button2.clicked.connect(lambda: self.show_stadium_dialog(2))
            self.control_right.layout().addWidget(self.stadium_button2, 2, 0)
        
        self.message_window.setText(f"플레이어 {player}\n스타디움/도구 관리 모드입니다.")

    def show_tool_dialog(self, player):
        dialog = ToolManagementDialog(self, current_player=player, slot_tool_state=self.slot_tool_state)
        dialog.slot_selected.connect(self.handle_tool_slot_selection)
        dialog.exec_()

    def handle_tool_slot_selection(self, player_number, slot_type):
        key = (player_number, slot_type)
        slot = None
        if player_number == 1:
            if slot_type == "active":
                slot = self.center_white[10]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[bench_index]
        else:
            if slot_type == "active":
                slot = self.center_white[11]
            elif slot_type.startswith("bench_"):
                bench_index = int(slot_type.split('_')[1])
                slot = self.center_white[5 + (4 - bench_index)]
        icon_path = os.path.join('tool_image', 'pokemon_bag_icon.png')
        # 항상 (bool, icon_path) 튜플로 저장
        def to_tuple(val):
            if isinstance(val, tuple):
                return val
            elif isinstance(val, bool):
                return (val, None)
            return (False, None)
        current = to_tuple(self.slot_tool_state.get(key, (False, None)))
        if current[0]:
            # 이미 장착된 경우: 해제
            self.slot_tool_state[key] = (False, None)
            if slot:
                slot.set_tool_icon(None)
            self.message_window.setText(f"플레이어 {player_number}\n{self.get_slot_kor(slot_type)} 슬롯\n도구가 해제되었습니다.")
        else:
            # 장착
            self.slot_tool_state[key] = (True, icon_path)
            if slot:
                slot.set_tool_icon(icon_path)
            self.message_window.setText(f"플레이어 {player_number}\n{self.get_slot_kor(slot_type)} 슬롯\n도구가 장착되었습니다.")

    def back_to_normal_buttons(self, player):
        """일반 버튼으로 돌아가기"""
        try:
            # 추가된 버튼들 제거
            if player == 1:
                # 포켓몬 체크 버튼들
                if hasattr(self, 'coin_toss_button1') and self.coin_toss_button1:
                    self.control_left.layout().removeWidget(self.coin_toss_button1)
                    self.coin_toss_button1.deleteLater()
                    self.coin_toss_button1 = None
                if hasattr(self, 'random_number_button1') and self.random_number_button1:
                    self.control_left.layout().removeWidget(self.random_number_button1)
                    self.random_number_button1.deleteLater()
                    self.random_number_button1 = None
                if hasattr(self, 'back_button1') and self.back_button1:
                    self.control_left.layout().removeWidget(self.back_button1)
                    self.back_button1.deleteLater()
                    self.back_button1 = None
                # 스타디움/도구 버튼들
                if hasattr(self, 'stadium_button1') and self.stadium_button1:
                    self.control_left.layout().removeWidget(self.stadium_button1)
                    self.stadium_button1.deleteLater()
                    self.stadium_button1 = None
                if hasattr(self, 'tool_button1') and self.tool_button1:
                    self.control_left.layout().removeWidget(self.tool_button1)
                    self.tool_button1.deleteLater()
                    self.tool_button1 = None
            else:
                # 포켓몬 체크 버튼들
                if hasattr(self, 'coin_toss_button2') and self.coin_toss_button2:
                    self.control_right.layout().removeWidget(self.coin_toss_button2)
                    self.coin_toss_button2.deleteLater()
                    self.coin_toss_button2 = None
                if hasattr(self, 'random_number_button2') and self.random_number_button2:
                    self.control_right.layout().removeWidget(self.random_number_button2)
                    self.random_number_button2.deleteLater()
                    self.random_number_button2 = None
                if hasattr(self, 'back_button2') and self.back_button2:
                    self.control_right.layout().removeWidget(self.back_button2)
                    self.back_button2.deleteLater()
                    self.back_button2 = None
                # 스타디움/도구 버튼들
                if hasattr(self, 'stadium_button2') and self.stadium_button2:
                    self.control_right.layout().removeWidget(self.stadium_button2)
                    self.stadium_button2.deleteLater()
                    self.stadium_button2 = None
                if hasattr(self, 'tool_button2') and self.tool_button2:
                    self.control_right.layout().removeWidget(self.tool_button2)
                    self.tool_button2.deleteLater()
                    self.tool_button2 = None
            
            # 일반 버튼들 다시 표시
            self.update_turn_buttons()
            
            # 메시지 창 업데이트
            if self.coin_toss_state == "finished":
                turn_text = "◀ Player 1's Turn!" if self.current_turn == 1 else "Player 2's Turn! ▶"
                self.message_window.setText(f"{turn_text}")
            else:
                self.message_window.setText("코인토스로 선공을 정하세요!\n\n한 플레이어가 먼저 선택하면\n다른 플레이어의 면은\n자동으로 선택됩니다.")
        except Exception as e:
            print(f"버튼 제거 중 오류 발생: {e}")
            # 오류 발생 시에도 일반 버튼들은 표시
            self.update_turn_buttons()

    def get_slot_kor(self, slot_type):
        if slot_type == "active":
            return "배틀필드"
        elif slot_type.startswith("bench_"):
            return f"벤치 {int(slot_type.split('_')[1])+1}"
        else:
            return slot_type

    def choose_coin_side(self, player, choice):
        """플레이어의 코인 선택 처리"""
        if self.coin_toss_state != "waiting":
            return
            
        if player == 1:
            self.first_player_choice = choice
            self.second_player_choice = "tails" if choice == "heads" else "heads"
        else:
            self.second_player_choice = choice
            self.first_player_choice = "tails" if choice == "heads" else "heads"
            
        # 코인토스 시작
        self.coin_toss_state = "tossing"
        self.coin_result = random.choice(["heads", "tails"])
        self.coin_toss_animation_frame = 0
        self.coin_rotation = 0
        self.coin_scale = 1.0
        self.coin_opacity = 0.0
        self.coin_current_image = self.coin_heads_image
        self.coin_toss_animation_timer.start(100)  # 100ms 간격으로 애니메이션 업데이트
        
        # 코인토스 버튼 숨기기
        self.coin_heads_button1.setVisible(False)
        self.coin_tails_button1.setVisible(False)
        self.coin_heads_button2.setVisible(False)
        self.coin_tails_button2.setVisible(False)
        
        # 메시지 창 업데이트
        self.message_window.setText("코인토스 중...")
        
        # 코인토스 중에 컨트롤 박스 버튼 숨기기
        if player == 1:
            if hasattr(self, 'coin_toss_button1'):
                self.coin_toss_button1.setVisible(False)
            if hasattr(self, 'random_number_button1'):
                self.random_number_button1.setVisible(False)
            if hasattr(self, 'back_button1'):
                self.back_button1.setVisible(False)
        else:
            if hasattr(self, 'coin_toss_button2'):
                self.coin_toss_button2.setVisible(False)
            if hasattr(self, 'random_number_button2'):
                self.random_number_button2.setVisible(False)
            if hasattr(self, 'back_button2'):
                self.back_button2.setVisible(False)

    def update_coin_toss_animation(self):
        """코인토스 애니메이션 업데이트"""
        self.coin_toss_animation_frame += 1
        
        # 포켓몬 체크 모드인지 확인
        is_pokemon_check = (hasattr(self, 'coin_toss_button1') and self.coin_toss_button1) or \
                          (hasattr(self, 'coin_toss_button2') and self.coin_toss_button2)
        
        # 포켓몬 체크 모드일 때는 1.5초(15프레임), 아닐 때는 6초(60프레임)
        max_frames = 15 if is_pokemon_check else 60
        
        if self.coin_toss_animation_frame >= max_frames:  # 애니메이션 종료
            self.coin_toss_animation_timer.stop()
            self.coin_toss_state = "finished"
            self.show_coin_toss_result()
        else:
            # 코인 회전 애니메이션 효과
            self.coin_rotation = self.coin_toss_animation_frame * (360 / max_frames)  # 360도 / 프레임 수
            
            # 3D 회전 효과를 위한 각도 계산 (0-180도)
            angle = self.coin_rotation % 180
            
            # 코인 앞/뒷면 전환 (90도마다)
            if angle < 90:
                self.coin_current_image = self.coin_heads_image
            else:
                self.coin_current_image = self.coin_tails_image
            
            # 3D 효과를 위한 크기 변화 (90도에서 가장 작아지고, 0도와 180도에서 가장 커짐)
            scale_factor = abs(math.cos(math.radians(angle)))  # 0-1 사이의 값
            self.coin_scale = 1.0 + scale_factor * 0.4  # 1.0 ~ 1.4 사이 변화
            
            # 3D 효과를 위한 투명도 변화 (90도에서 가장 투명하고, 0도와 180도에서 가장 불투명)
            self.coin_opacity = 10 + scale_factor * 0.15  # 0.85 ~ 1.0 사이 변화 (투명도 범위 조정)
            
            # 전체 애니메이션 진행도에 따른 추가 효과
            progress = self.coin_toss_animation_frame / max_frames
            
            # 시작과 끝에서의 페이드 인/아웃 효과
            if progress < 0.2:  # 시작 20%
                fade_factor = progress * 5  # 0 -> 1
                self.coin_opacity = 10 + (self.coin_opacity - 0.85) * fade_factor  # 최소 투명도 0.85 유지
            elif progress > 0.8:  # 마지막 20%
                fade_factor = (1.0 - progress) * 5  # 1 -> 0
                self.coin_opacity = 10 + (self.coin_opacity - 0.85) * fade_factor  # 최소 투명도 0.85 유지
            
            self.update()  # 화면 갱신

    def show_coin_toss_result(self):
        """코인토스 결과 표시"""
        # 포켓몬 체크 모드인지 확인
        is_pokemon_check = (hasattr(self, 'coin_toss_button1') and self.coin_toss_button1) or \
                          (hasattr(self, 'coin_toss_button2') and self.coin_toss_button2)
        
        if is_pokemon_check:
            # 결과 이미지 설정
            result_image = self.coin_heads_image if self.coin_result == "heads" else self.coin_tails_image
            result_image = result_image.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 이미지를 HTML 형식으로 변환
            image_data = result_image.toImage()
            image_bytes = QByteArray()
            buffer = QBuffer(image_bytes)
            buffer.open(QIODevice.WriteOnly)
            image_data.save(buffer, "PNG")
            image_base64 = image_bytes.toBase64().data().decode()
            
            # 결과 텍스트를 한글로 변환
            result_text = "앞면" if self.coin_result == "heads" else "뒷면"
            
            # HTML 형식으로 결과 표시
            result_html = f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{image_base64}' style='margin-bottom: 10px;'><br>
                <span style='font-size: 14px;'>결과: {result_text}</span>
            </div>
            """
            
            # 메시지 창 표시
            self.message_window.setVisible(True)
            self.message_window.setText(result_html)
            
            # 2초 후 원래 메인 컨트롤 박스로 복귀
            QTimer.singleShot(2000, lambda: self.back_to_main_control_box(1 if hasattr(self, 'coin_toss_button1') else 2))
            
            # 코인토스 상태를 finished로 설정하여 메인 컨트롤 박스로 돌아가도록 함
            self.coin_toss_state = "finished"
            self.coin_current_image = None
        else:
            # 기존의 선공 결정 로직
            first_player_won = (self.first_player_choice == self.coin_result)
            winner = 1 if first_player_won else 2
            
            # 결과 이미지 설정
            result_image = self.coin_heads_image if self.coin_result == "heads" else self.coin_tails_image
            result_image = result_image.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 이미지를 HTML 형식으로 변환
            image_data = result_image.toImage()
            image_bytes = QByteArray()
            buffer = QBuffer(image_bytes)
            buffer.open(QIODevice.WriteOnly)
            image_data.save(buffer, "PNG")
            image_base64 = image_bytes.toBase64().data().decode()
            
            # 결과 텍스트를 한글로 변환
            result_text = "앞면" if self.coin_result == "heads" else "뒷면"
            
            # HTML 형식으로 결과 표시
            result_html = f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{image_base64}' style='margin-bottom: 10px;'><br>
                <span style='font-size: 14px;'>결과: {result_text}</span><br>
                <span style='font-size: 14px;'>플레이어{winner} 선공!</span>
            </div>
            """
            
            # 메시지 창 표시
            self.message_window.setVisible(True)
            self.message_window.setText(result_html)
            
            # 3초 후 게임 시작
            QTimer.singleShot(3000, lambda: self.start_game(winner))

    def back_to_main_control_box(self, player):
        """포켓몬 체크 등에서 메인 컨트롤 박스로 복귀"""
        try:
            # 추가된 버튼들 제거
            if player == 1:
                # 포켓몬 체크 버튼들
                if hasattr(self, 'coin_toss_button1') and self.coin_toss_button1:
                    self.control_left.layout().removeWidget(self.coin_toss_button1)
                    self.coin_toss_button1.deleteLater()
                    self.coin_toss_button1 = None
                if hasattr(self, 'random_number_button1') and self.random_number_button1:
                    self.control_left.layout().removeWidget(self.random_number_button1)
                    self.random_number_button1.deleteLater()
                    self.random_number_button1 = None
                if hasattr(self, 'back_button1') and self.back_button1:
                    self.control_left.layout().removeWidget(self.back_button1)
                    self.back_button1.deleteLater()
                    self.back_button1 = None
            else:
                # 포켓몬 체크 버튼들
                if hasattr(self, 'coin_toss_button2') and self.coin_toss_button2:
                    self.control_right.layout().removeWidget(self.coin_toss_button2)
                    self.coin_toss_button2.deleteLater()
                    self.coin_toss_button2 = None
                if hasattr(self, 'random_number_button2') and self.random_number_button2:
                    self.control_right.layout().removeWidget(self.random_number_button2)
                    self.random_number_button2.deleteLater()
                    self.random_number_button2 = None
                if hasattr(self, 'back_button2') and self.back_button2:
                    self.control_right.layout().removeWidget(self.back_button2)
                    self.back_button2.deleteLater()
                    self.back_button2 = None
            
            # 일반 버튼들 다시 표시
            self.update_turn_buttons()
            
            # 메시지 창 업데이트
            turn_text = "◀ Player 1's Turn!" if self.current_turn == 1 else "Player 2's Turn! ▶"
            self.message_window.setText(f"{turn_text}")
        except Exception as e:
            print(f"버튼 제거 중 오류 발생: {e}")
            # 오류 발생 시에도 일반 버튼들은 표시
            self.update_turn_buttons()

    def start_game(self, winner):
        """게임 시작"""
        self.current_turn = winner
        self.update_turn_buttons()
        self.message_window.setText(f"플레이어{winner}의 턴입니다!\n\n게임을 시작합니다!")
        self.coin_current_image = None  # 코인 이미지 숨기기

    def load_coin_images(self):
        """코인 이미지 로드 및 예외 처리"""
        try:
            heads_path = os.path.join(COIN_IMAGES_DIR, 'heads.png')
            tails_path = os.path.join(COIN_IMAGES_DIR, 'tails.png')
            
            if not os.path.exists(heads_path) or not os.path.exists(tails_path):
                print("코인 이미지 파일이 없습니다. 기본 이미지를 생성합니다.")
                # 기본 코인 이미지 생성 (원형)
                self.coin_heads_image = QPixmap(200, 200)
                self.coin_heads_image.fill(Qt.transparent)
                painter = QPainter(self.coin_heads_image)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setBrush(QColor(255, 215, 0))  # 금색
                painter.setPen(QPen(Qt.black, 2))
                painter.drawEllipse(10, 10, 180, 180)
                painter.setPen(QPen(Qt.black, 3))
                painter.setFont(QFont('Arial', 40, QFont.Bold))
                painter.drawText(self.coin_heads_image.rect(), Qt.AlignCenter, "H")
                painter.end()
                
                self.coin_tails_image = QPixmap(200, 200)
                self.coin_tails_image.fill(Qt.transparent)
                painter = QPainter(self.coin_tails_image)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setBrush(QColor(255, 215, 0))  # 금색
                painter.setPen(QPen(Qt.black, 2))
                painter.drawEllipse(10, 10, 180, 180)
                painter.setPen(QPen(Qt.black, 3))
                painter.setFont(QFont('Arial', 40, QFont.Bold))
                painter.drawText(self.coin_tails_image.rect(), Qt.AlignCenter, "T")
                painter.end()
            else:
                self.coin_heads_image = QPixmap(heads_path)
                self.coin_tails_image = QPixmap(tails_path)
            
            self.coin_current_image = self.coin_heads_image
        except Exception as e:
            print(f"코인 이미지 로드 중 오류 발생: {e}")
            # 오류 발생 시 기본 이미지 생성
            self.coin_heads_image = QPixmap(200, 200)
            self.coin_tails_image = QPixmap(200, 200)
            self.coin_current_image = self.coin_heads_image

    def check_pokemon(self, player):
        """포켓몬 체크 버튼 클릭 처리"""
        # 모든 버튼 숨기기
        self.energy_button1.setVisible(False)
        self.hp_button1.setVisible(False)
        self.status_button1.setVisible(False)
        self.evolution_button1.setVisible(False)
        self.position_button1.setVisible(False)
        self.trainer_button1.setVisible(False)
        self.pokemon_check_button1.setVisible(False)
        self.end_turn_button1.setVisible(False)
        
        self.energy_button2.setVisible(False)
        self.hp_button2.setVisible(False)
        self.status_button2.setVisible(False)
        self.evolution_button2.setVisible(False)
        self.position_button2.setVisible(False)
        self.trainer_button2.setVisible(False)
        self.pokemon_check_button2.setVisible(False)
        self.end_turn_button2.setVisible(False)
        
        # 코인토스 버튼과 숫자 랜덤 버튼, 이전으로 버튼 표시
        if player == 1:
            # 코인토스 버튼 추가
            self.coin_toss_button1 = CustomRotatedButton("코인\n토스", rotation=90, font_family=self.pokemon_font)
            self.coin_toss_button1.setStyleSheet("""
                QPushButton {
                    background-color: #f1c40f;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #f39c12;
                }
                QPushButton:pressed {
                    background-color: #d35400;
                }
            """)
            self.coin_toss_button1.clicked.connect(lambda: self.pokemon_check_coin_toss(1))
            self.control_left.layout().addWidget(self.coin_toss_button1, 0, 0)
            
            # 숫자 랜덤 버튼 추가
            self.random_number_button1 = CustomRotatedButton("숫자\n랜덤", rotation=90, font_family=self.pokemon_font)
            self.random_number_button1.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
                QPushButton:pressed {
                    background-color: #6c3483;
                }
            """)
            self.random_number_button1.clicked.connect(lambda: self.generate_random_number(1))
            self.control_left.layout().addWidget(self.random_number_button1, 1, 0)
            
            # 이전으로 버튼 추가
            self.back_button1 = CustomRotatedButton("이전으로", rotation=90, font_family=self.pokemon_font)
            self.back_button1.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            self.back_button1.clicked.connect(lambda: self.back_to_normal_buttons(1))
            self.control_left.layout().addWidget(self.back_button1, 2, 0)
        else:
            # 이전으로 버튼 추가
            self.back_button2 = CustomRotatedButton("이전으로", rotation=-90, font_family=self.pokemon_font)
            self.back_button2.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            self.back_button2.clicked.connect(lambda: self.back_to_normal_buttons(2))
            self.control_right.layout().addWidget(self.back_button2, 0, 0)
            
            # 숫자 랜덤 버튼 추가
            self.random_number_button2 = CustomRotatedButton("숫자\n랜덤", rotation=-90, font_family=self.pokemon_font)
            self.random_number_button2.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #8e44ad;
                }
                QPushButton:pressed {
                    background-color: #6c3483;
                }
            """)
            self.random_number_button2.clicked.connect(lambda: self.generate_random_number(2))
            self.control_right.layout().addWidget(self.random_number_button2, 1, 0)
            
            # 코인토스 버튼 추가
            self.coin_toss_button2 = CustomRotatedButton("코인\n토스", rotation=-90, font_family=self.pokemon_font)
            self.coin_toss_button2.setStyleSheet("""
                QPushButton {
                    background-color: #f1c40f;
                    color: black;
                    border: 2px solid #2c3e50;
                    border-radius: 8px;
                    padding: 1px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 47px;
                    min-height: 47px;
                    max-width: 47px;
                    max-height: 47px;
                }
                QPushButton:hover {
                    background-color: #f39c12;
                }
                QPushButton:pressed {
                    background-color: #d35400;
                }
            """)
            self.coin_toss_button2.clicked.connect(lambda: self.pokemon_check_coin_toss(2))
            self.control_right.layout().addWidget(self.coin_toss_button2, 2, 0)
        
        self.message_window.setText(f"플레이어 {player}\n포켓몬 체크 모드입니다.")

    def pokemon_check_coin_toss(self, player):
        """포켓몬 체크용 코인토스"""
        # 코인토스 시작
        self.coin_toss_state = "tossing"
        self.coin_result = random.choice(["heads", "tails"])
        self.coin_toss_animation_frame = 0
        self.coin_rotation = 0
        self.coin_scale = 1.0
        self.coin_opacity = 0.0
        self.coin_current_image = self.coin_heads_image
        self.coin_toss_animation_timer.start(100)  # 100ms 간격으로 애니메이션 업데이트
        
        # 메시지 창 업데이트
        self.message_window.setText("코인토스 중...")
        
        # 코인토스 중에 컨트롤 박스 버튼 숨기기
        if player == 1:
            if hasattr(self, 'coin_toss_button1'):
                self.coin_toss_button1.setVisible(False)
            if hasattr(self, 'random_number_button1'):
                self.random_number_button1.setVisible(False)
            if hasattr(self, 'back_button1'):
                self.back_button1.setVisible(False)
        else:
            if hasattr(self, 'coin_toss_button2'):
                self.coin_toss_button2.setVisible(False)
            if hasattr(self, 'random_number_button2'):
                self.random_number_button2.setVisible(False)
            if hasattr(self, 'back_button2'):
                self.back_button2.setVisible(False)

    def generate_random_number(self, player):
        """랜덤 숫자 생성"""
        dialog = RandomNumberDialog(self, current_player=player)
        if dialog.exec_() == QDialog.Accepted:
            max_number = dialog.get_selected_number()
            number = random.randint(1, max_number)
            self.message_window.setText(f"플레이어 {player}\n랜덤 숫자: {number}")

    def show_stadium_dialog(self, player):
        dialog = StadiumManagementDialog(self, current_player=player)
        dialog.stadium_completed.connect(self.handle_stadium_completed)
        dialog.exec_()

    def handle_stadium_completed(self, player_number):
        self.message_window.setText("임시 스타디움\n\n카드가 발동되었습니다!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    board = PokemonBoard()
    board.show()
    sys.exit(app.exec_()) 