import sys
import os
GAMES_JSON_PATH = os.path.join("data", "game", "games.json")
from enum import IntEnum
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem, QMessageBox, QPushButton, QLabel, QSpinBox, QDialog, QGraphicsScene, QGraphicsTextItem, QGraphicsProxyWidget, QGroupBox, QLineEdit, QComboBox, QTextEdit, QLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from ui.mainwindow import Ui_MainWindow
from core.game import GameManager
from core.character_manager import CharacterManager
from .spell_dialog import SpellSelectionDialog

class Page(IntEnum):
    MAIN       = 0
    GAME_LIST  = 1
    CHAR_LIST  = 2
    MAKE_CHAR  = 3
    PLAY_MAP   = 4
    FIGHT_PAGE = 5

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # OOP 구조: GameManager 인스턴스
        self.manager = GameManager(GAMES_JSON_PATH)
        self.current_game = None
        self.current_game_id = None  # 선택된 게임의 시간값
        
        # CharacterManager 인스턴스
        self.char_manager = CharacterManager()

        # 시작 화면을 MAIN으로
        self.switch_page(Page.MAIN)

        # 시그널 연결
        self.ui.p1_to_p2_btn.clicked.connect(lambda: self.switch_page(Page.GAME_LIST))
        self.ui.p2_to_p1_btn.clicked.connect(lambda: self.switch_page(Page.MAIN))
        self.ui.p3_to_p2_btn.clicked.connect(lambda: self.switch_page(Page.GAME_LIST))
        self.ui.p4_goback_btn.clicked.connect(lambda: self.switch_page(Page.CHAR_LIST))
        
        # 디버그 메뉴 액션 연결
        self.ui.action_toggle_overlay.triggered.connect(self.toggle_overlay)

        self.ui.p2_make_game_btn.clicked.connect(self.on_new_game)
        self.ui.p2_del_game_btn.clicked.connect(self.on_delete_game)
        self.ui.p2_game_list.itemDoubleClicked.connect(self.on_select_game)
        self.ui.p3_play_btn.clicked.connect(self.play_game)
        
        # 캐릭터 관련 버튼 연결
        self.ui.p3_make_char_btn.clicked.connect(lambda: self.switch_page(Page.MAKE_CHAR))
        self.ui.p4_goon_btn.clicked.connect(self.create_character)
        self.ui.spell_button.clicked.connect(self.show_spell_selection)
        self.ui.p3_del_char_btn.clicked.connect(self.on_delete_character)

        # p3/p4 텍스트 초기화
        self.ui.p3_text_gamename.clear()
        self.ui.p4_text_gamename.clear()

        # 게임 리스트 표시
        self.refresh_game_list()
        
        # 캐릭터 리스트 초기화
        self.refresh_character_list()

        # 직업 버튼 초기화
        self.init_class_buttons()
        
        # 종족 콤보박스 초기화
        self.init_race_combos()
        
        # 배경 버튼 초기화
        self.init_background_buttons()
        
        # 능력치 입력 초기화
        self.init_ability_inputs()

        # 선택된 스펠 저장
        self.selected_spells = []

        # 상좌(좌상단) 버튼 3개를 QGraphicsView에 180도 회전시켜 추가
        self.ui.graphics_topleft_buttons.setMinimumSize(220, 180)  # 컨테이너 크기 줄임
        scene_topleft_btns = QGraphicsScene()
        for i, label in enumerate(["행동1", "행동2", "행동3"]):
            btn = QPushButton(label)
            btn.setMinimumSize(200, 50)  # 버튼 크기 줄임
            btn.setMaximumSize(200, 50)
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setRotation(180)
            proxy.setPos(0, i * 60)  # 버튼 간격 줄임
            scene_topleft_btns.addItem(proxy)
        self.ui.graphics_topleft_buttons.setScene(scene_topleft_btns)
        # 상우(우상단) 버튼 3개를 QGraphicsView에 180도 회전시켜 추가
        self.ui.graphics_topright_buttons.setMinimumSize(220, 180)  # 컨테이너 크기 줄임
        scene_topright_btns = QGraphicsScene()
        for i, label in enumerate(["행동1", "행동2", "행동3"]):
            btn = QPushButton(label)
            btn.setMinimumSize(200, 50)  # 버튼 크기 줄임
            btn.setMaximumSize(200, 50)
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setRotation(180)
            proxy.setPos(0, i * 60)  # 버튼 간격 줄임
            scene_topright_btns.addItem(proxy)
        self.ui.graphics_topright_buttons.setScene(scene_topright_btns)

        # 상좌(좌상단) 텍스트 QGraphicsView에 180도 회전 텍스트 추가
        scene_topleft_text = QGraphicsScene()
        text_item_topleft = QGraphicsTextItem("플레이어1")
        text_item_topleft.setRotation(180)
        rect = text_item_topleft.boundingRect()
        text_item_topleft.setTransformOriginPoint(rect.center())
        scene_topleft_text.addItem(text_item_topleft)
        self.ui.graphics_topleft_text.setScene(scene_topleft_text)
        self.ui.graphics_topleft_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_topleft_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 상우(우상단) 텍스트 QGraphicsView에 180도 회전 텍스트 추가
        scene_topright_text = QGraphicsScene()
        text_item_topright = QGraphicsTextItem("플레이어2")
        text_item_topright.setRotation(180)
        rect2 = text_item_topright.boundingRect()
        text_item_topright.setTransformOriginPoint(rect2.center())
        scene_topright_text.addItem(text_item_topright)
        self.ui.graphics_topright_text.setScene(scene_topright_text)
        self.ui.graphics_topright_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_topright_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 스크롤바 항상 숨김 설정
        self.ui.graphics_topleft_buttons.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_topleft_buttons.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_topright_buttons.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_topright_buttons.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 좌하(하좌) 텍스트 QGraphicsView에 텍스트 추가 (회전 없음)
        scene_bottomleft_text = QGraphicsScene()
        text_item_bottomleft = QGraphicsTextItem("플레이어3")
        scene_bottomleft_text.addItem(text_item_bottomleft)
        self.ui.graphics_bottomleft_text.setScene(scene_bottomleft_text)
        self.ui.graphics_bottomleft_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_bottomleft_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 좌하(하좌) 버튼 QGraphicsView에 버튼 3개 추가 (회전 없음)
        self.ui.graphics_bottomleft_buttons.setMinimumSize(220, 180)  # 컨테이너 크기 줄임
        scene_bottomleft_btns = QGraphicsScene()
        for i, label in enumerate(["행동1", "행동2", "행동3"]):
            btn = QPushButton(label)
            btn.setMinimumSize(200, 50)  # 버튼 크기 줄임
            btn.setMaximumSize(200, 50)
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setPos(0, i * 60)  # 버튼 간격 줄임
            scene_bottomleft_btns.addItem(proxy)
        self.ui.graphics_bottomleft_buttons.setScene(scene_bottomleft_btns)
        self.ui.graphics_bottomleft_buttons.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_bottomleft_buttons.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 우하(하우) 텍스트 QGraphicsView에 텍스트 추가 (회전 없음)
        scene_bottomright_text = QGraphicsScene()
        text_item_bottomright = QGraphicsTextItem("플레이어4")
        scene_bottomright_text.addItem(text_item_bottomright)
        self.ui.graphics_bottomright_text.setScene(scene_bottomright_text)
        self.ui.graphics_bottomright_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_bottomright_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 우하(하우) 버튼 QGraphicsView에 버튼 3개 추가 (회전 없음)
        self.ui.graphics_bottomright_buttons.setMinimumSize(220, 180)  # 컨테이너 크기 줄임
        scene_bottomright_btns = QGraphicsScene()
        for i, label in enumerate(["행동1", "행동2", "행동3"]):
            btn = QPushButton(label)
            btn.setMinimumSize(200, 50)  # 버튼 크기 줄임
            btn.setMaximumSize(200, 50)
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setPos(0, i * 60)  # 버튼 간격 줄임
            scene_bottomright_btns.addItem(proxy)
        self.ui.graphics_bottomright_buttons.setScene(scene_bottomright_btns)
        self.ui.graphics_bottomright_buttons.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.graphics_bottomright_buttons.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.webview_map = None  # 웹맵 뷰 변수 추가
        self.setup_playmap_webview()  # 웹맵 세팅 함수 호출
        # 오버레이 항상 보이게 설정
        self.ui.frame_overlay.setVisible(True)
        
        # 창 크기를 화면에 맞게 조정
        self.resize(1280, 800)
        
        # 전체 화면용 스타일시트 적용
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                font-size: 16px;
                padding: 12px;
                min-height: 50px;
                background-color: #4a4a4a;
                color: white;
                border: 2px solid #666;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QTextEdit, QTextBrowser {
                font-size: 14px;
                padding: 8px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QListWidget {
                font-size: 14px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555;
            }
            QListWidget::item:selected {
                background-color: #5a5a5a;
            }
            QVBoxLayout, QHBoxLayout {
                margin: 10px;
                spacing: 10px;
            }
            
            /* 캐릭터 생성 페이지 전용 스타일 */
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-size: 12px;
                color: white;
            }
            QLineEdit, QComboBox {
                font-size: 12px;
                padding: 5px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                min-height: 25px;
            }
            QSpinBox {
                font-size: 12px;
                padding: 3px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                min-height: 25px;
                min-width: 60px;
            }
            
            /* 캐릭터 생성 페이지 하단 버튼들 */
            #spell_button, #p4_goback_btn, #p4_goon_btn {
                font-size: 14px;
                padding: 8px;
                min-height: 35px;
                max-height: 35px;
            }
        """)
        
        # 레이아웃 여백 조정
        self.ui.centralwidget.layout().setContentsMargins(20, 20, 20, 20)
        self.ui.centralwidget.layout().setSpacing(15)
        
        # 전체 화면으로 설정
        self.showFullScreen()

    def keyPressEvent(self, event):
        """키보드 이벤트 처리"""
        if event.key() == Qt.Key_Escape:
            # ESC 키로 전체 화면 해제
            if self.isFullScreen():
                self.showNormal()
                print("전체 화면 해제")
            else:
                self.showFullScreen()
                print("전체 화면으로 전환")
        super().keyPressEvent(event)

    def switch_page(self, page: Page):
        self.ui.stack_page.setCurrentIndex(page.value)
        if page == Page.CHAR_LIST:
            self.refresh_character_list()
        if page == Page.MAKE_CHAR:
            self.reset_makechar_page()

    def reset_makechar_page(self):
        # 이름 입력 초기화
        self.ui.name_input.clear()
        # 직업 버튼 초기화
        for i in range(self.ui.class_grid.count()):
            button = self.ui.class_grid.itemAt(i).widget()
            button.setChecked(False)
        # 종족/서브종족 초기화
        if self.ui.race_combo.count() > 0:
            self.ui.race_combo.setCurrentIndex(0)
        if self.ui.subrace_combo.count() > 0:
            self.ui.subrace_combo.setCurrentIndex(0)
        # 배경 버튼 초기화
        for i in range(self.ui.background_grid.count()):
            button = self.ui.background_grid.itemAt(i).widget()
            button.setChecked(False)
        # 능력치 초기화
        for i in range(0, self.ui.ability_layout.count(), 2):
            spin = self.ui.ability_layout.itemAt(i + 1).widget()
            spin.setValue(10)
        # 스펠 선택 초기화
        self.selected_spells = []
        self.ui.spell_button.setEnabled(True)
        
        # Character creation page - make it even smaller
        if hasattr(self, 'ui') and hasattr(self.ui, 'makechar'):
            # Make buttons even smaller
            for button in self.ui.makechar.findChildren(QPushButton):
                button.setFixedSize(80, 25)  # Even smaller buttons
                font = button.font()
                font.setPointSize(8)  # Smaller font
                button.setFont(font)
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #4a4a4a;
                        color: white;
                        border: 1px solid #666;
                        border-radius: 3px;
                        padding: 2px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #5a5a5a;
                    }
                    QPushButton:pressed {
                        background-color: #3a3a3a;
                    }
                """)
            
            # Make text areas even smaller
            for text_edit in self.ui.makechar.findChildren(QTextEdit):
                text_edit.setFixedHeight(60)  # Even smaller height
                font = text_edit.font()
                font.setPointSize(9)  # Smaller font
                text_edit.setFont(font)
                text_edit.setStyleSheet("""
                    QTextEdit {
                        background-color: #2a2a2a;
                        color: white;
                        border: 1px solid #555;
                        border-radius: 3px;
                        padding: 3px;
                    }
                """)
            
            # Make labels smaller
            for label in self.ui.makechar.findChildren(QLabel):
                font = label.font()
                font.setPointSize(9)  # Smaller font
                label.setFont(font)
                label.setStyleSheet("""
                    QLabel {
                        color: white;
                        background-color: transparent;
                        padding: 1px;
                    }
                """)
            
            # Adjust layout spacing
            for layout in self.ui.makechar.findChildren(QLayout):
                if hasattr(layout, 'setSpacing'):
                    layout.setSpacing(2)  # Even smaller spacing
                if hasattr(layout, 'setContentsMargins'):
                    layout.setContentsMargins(5, 5, 5, 5)  # Even smaller margins

    def refresh_game_list(self):
        self.ui.p2_game_list.clear()
        for game in self.manager.games:
            item = QListWidgetItem(game.display_text)
            self.ui.p2_game_list.addItem(item)

    def refresh_character_list(self):
        self.ui.p3_char_list.clear()
        if self.current_game_id:
            characters = self.char_manager.list_characters_by_game(self.current_game_id)
            for char in characters:
                self.ui.p3_char_list.addItem(QListWidgetItem(char))

    def on_new_game(self):
        # games.json에서 중복 시간값 체크
        import json
        from datetime import datetime
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(GAMES_JSON_PATH), exist_ok=True)
        
        # games.json 파일이 존재하지 않으면 빈 딕셔너리로 초기화
        try:
            with open(GAMES_JSON_PATH, "r", encoding="utf-8") as f:
                games = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            games = {}
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if now in games:
            QMessageBox.warning(self, "경고", "동일한 시간의 게임이 이미 존재합니다. 잠시 후 다시 시도하세요.")
            return
        
        try:
            game = self.manager.add_game()
            self.ui.p2_game_list.addItem(QListWidgetItem(game.display_text))
        except Exception as e:
            QMessageBox.critical(self, "오류", f"게임 생성 중 오류가 발생했습니다: {str(e)}")

    def on_delete_game(self):
        row = self.ui.p2_game_list.currentRow()
        if row >= 0:
            self.manager.delete_game(row)
            self.refresh_game_list()

    def on_select_game(self, item: QListWidgetItem):
        row = self.ui.p2_game_list.row(item)
        self.current_game = self.manager.games[row]
        self.current_game_id = self.current_game.created_at  # Game 객체의 created_at 사용
        text = self.current_game.display_text
        self.ui.p3_text_gamename.setText(text)
        self.ui.p4_text_gamename.setText(text)
        self.switch_page(Page.CHAR_LIST)
        
    def init_class_buttons(self):
        """직업 선택 버튼 초기화 (UI와 동일하게 12종)"""
        classes = [
            "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk",
            "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"
        ]
        layout = self.ui.class_grid
        layout.setSpacing(10)  # 간격 줄임
        layout.setContentsMargins(10, 10, 10, 10)  # 여백 줄임
        for i, class_name in enumerate(classes):
            button = QPushButton(class_name)
            button.setMinimumSize(150, 40)  # 버튼 크기 줄임
            button.setMaximumSize(150, 40)  # 버튼 크기 줄임
            button.setCheckable(True)
            button.clicked.connect(lambda checked, c=class_name: self.select_class(c))
            layout.addWidget(button, i // 6, i % 6)
        # 스펠 버튼 항상 활성화
        self.ui.spell_button.setEnabled(True)
            
    def init_race_combos(self):
        """종족 콤보박스 초기화 (UI와 동일하게 확장)"""
        races = [
            "Human", "Elf", "Dwarf", "Halfling", "Gnome", "Half-Elf",
            "Half-Orc", "Tiefling", "Dragonborn"
        ]
        self.ui.race_combo.clear()
        self.ui.race_combo.addItems(races)
        self.ui.race_combo.currentTextChanged.connect(self.update_subraces)
        
    def update_subraces(self, race):
        """선택된 종족에 따른 서브종족 업데이트 (UI와 동일하게 확장)"""
        self.ui.subrace_combo.clear()
        subraces = {
            "Human": ["None", "Variant Human"],
            "Elf": ["High Elf", "Wood Elf", "Dark Elf (Drow)"],
            "Dwarf": ["Hill Dwarf", "Mountain Dwarf"],
            "Halfling": ["Lightfoot", "Stout"],
            "Gnome": ["Rock Gnome", "Forest Gnome"],
            "Half-Elf": ["None"],
            "Half-Orc": ["None"],
            "Tiefling": ["None"],
            "Dragonborn": ["None"]
        }
        self.ui.subrace_combo.addItems(subraces.get(race, ["None"]))
        
    def init_background_buttons(self):
        """배경 선택 버튼 초기화 (UI와 동일하게 확장)"""
        backgrounds = [
            'Acolyte', 'Criminal', 'Folk Hero', 'Noble', 'Sage',
            'Soldier', 'Entertainer', 'Guild Artisan', 'Hermit',
            'Outlander', 'Urchin', 'Charlatan', 'Haunted One'
        ]
        layout = self.ui.background_grid
        layout.setSpacing(10)  # 간격 줄임
        layout.setContentsMargins(10, 10, 10, 10)  # 여백 줄임
        for i, bg in enumerate(backgrounds):
            button = QPushButton(bg)
            button.setMinimumSize(150, 40)  # 버튼 크기 줄임
            button.setMaximumSize(150, 40)  # 버튼 크기 줄임
            button.setCheckable(True)
            button.clicked.connect(lambda checked, b=bg: self.select_background(b))
            layout.addWidget(button, i // 6, i % 6)
            
    def init_ability_inputs(self):
        """능력치 입력 필드 초기화 (UI와 동일하게 6종 영어)"""
        abilities = ["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"]
        layout = self.ui.ability_layout
        for i, ability in enumerate(abilities):
            label = QLabel(ability)
            spin = QSpinBox()
            spin.setRange(3, 18)
            spin.setValue(10)
            layout.addWidget(label, i // 2, (i % 2) * 2)
            layout.addWidget(spin, i // 2, (i % 2) * 2 + 1)
            
    def select_class(self, class_name):
        """직업 선택 처리"""
        # 다른 직업 버튼 해제
        for i in range(self.ui.class_grid.count()):
            button = self.ui.class_grid.itemAt(i).widget()
            if button.text() != class_name:
                button.setChecked(False)
        # 스펠 선택 초기화
        self.selected_spells = []
        # 스펠 버튼 항상 활성화로 변경 (별도 처리)
        # self.ui.spell_button.setEnabled(True)
        
    def select_background(self, background):
        """배경 선택 처리"""
        # 다른 배경 버튼 해제
        for i in range(self.ui.background_grid.count()):
            button = self.ui.background_grid.itemAt(i).widget()
            if button.text() != background:
                button.setChecked(False)
                
    def show_spell_selection(self):
        """스펠 선택 다이얼로그 표시"""
        # 선택된 직업 찾기
        selected_class = None
        for i in range(self.ui.class_grid.count()):
            button = self.ui.class_grid.itemAt(i).widget()
            if button.isChecked():
                selected_class = button.text()
                break
        # 마법 직업 목록
        spell_classes = [
            "Bard", "Cleric", "Druid", "Paladin", "Ranger",
            "Sorcerer", "Warlock", "Wizard",
            "바드", "성직자", "드루이드", "팔라딘", "레인저", "소서러", "워락", "마법사"
        ]
        if not selected_class:
            QMessageBox.warning(self, "경고", "직업을 먼저 선택하세요.")
            return
        if selected_class not in spell_classes:
            QMessageBox.warning(self, "경고", "마법을 쓸 수 없는 직업입니다.")
            return
        dialog = SpellSelectionDialog(self, selected_class)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_spells = dialog.get_selected_spells()
            
    def create_character(self):
        """캐릭터 생성"""
        # 입력값 수집
        name = self.ui.name_input.text()
        if not name:
            QMessageBox.warning(self, "경고", "캐릭터 이름을 입력하세요.")
            return
            
        # 선택된 직업 찾기
        selected_class = None
        for i in range(self.ui.class_grid.count()):
            button = self.ui.class_grid.itemAt(i).widget()
            if button.isChecked():
                selected_class = button.text()
                break
                
        if not selected_class:
            QMessageBox.warning(self, "경고", "직업을 선택하세요.")
            return
            
        # 종족 정보
        race = self.ui.race_combo.currentText()
        subrace = self.ui.subrace_combo.currentText()
        
        # 선택된 배경 찾기
        selected_background = None
        for i in range(self.ui.background_grid.count()):
            button = self.ui.background_grid.itemAt(i).widget()
            if button.isChecked():
                selected_background = button.text()
                break
                
        if not selected_background:
            QMessageBox.warning(self, "경고", "배경을 선택하세요.")
            return
            
        # 능력치 수집 (영문 키)
        abilities = {}
        for i in range(0, self.ui.ability_layout.count(), 2):
            label = self.ui.ability_layout.itemAt(i).widget()
            spin = self.ui.ability_layout.itemAt(i + 1).widget()
            abilities[label.text()] = spin.value()
            
        # 캐릭터 생성
        if not self.current_game_id:
            QMessageBox.warning(self, "경고", "게임을 먼저 선택하세요.")
            return
        try:
            self.char_manager.create_character(
                name=name,
                character_class=selected_class,
                race=race,
                subrace=subrace,
                background=selected_background,
                abilities=abilities,
                spells=self.selected_spells,
                game_id=self.current_game_id
            )
            QMessageBox.information(self, "성공", "캐릭터가 생성되었습니다.")
            self.switch_page(Page.CHAR_LIST)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"캐릭터 생성 중 오류가 발생했습니다: {str(e)}")

    def on_delete_character(self):
        # 캐릭터 삭제 버튼 클릭 시
        selected = self.ui.p3_char_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "경고", "삭제할 캐릭터를 선택하세요.")
            return
        name = selected.text()
        if not self.current_game_id:
            QMessageBox.warning(self, "경고", "게임이 선택되지 않았습니다.")
            return
        self.char_manager.delete_character(name, self.current_game_id)
        self.refresh_character_list()

    def play_game(self):
        self.switch_page(Page.PLAY_MAP)

    def toggle_overlay(self):
        """오버레이 토글 기능"""
        if self.ui.stack_page.currentIndex() == Page.PLAY_MAP:
            is_visible = not self.ui.frame_overlay.isVisible()
            self.ui.frame_overlay.setVisible(is_visible)
            if is_visible:
                # 오버레이를 우측 하단 모서리에 배치
                parent = self.ui.frame_overlay.parent()
                if parent:
                    x = parent.width() - self.ui.frame_overlay.width() - 10
                    y = parent.height() - self.ui.frame_overlay.height() - 10
                    self.ui.frame_overlay.move(x, y)

    def setup_playmap_webview(self):
        # PLAY_MAP 페이지의 QGraphicsView를 QWebEngineView로 대체
        playmap_widget = self.ui.playmap
        grid_layout = self.ui.gridLayout  # QGridLayout
        
        # 기존에 추가된 웹뷰가 있다면 제거
        if hasattr(self, 'webview_map') and self.webview_map:
            grid_layout.removeWidget(self.webview_map)
            self.webview_map.deleteLater()

        # QWebEngineView 생성 및 추가
        self.webview_map = QWebEngineView(playmap_widget)
        
        # 파일 경로 확인
        # __file__을 기준으로 상대경로 계산하여 안정성 확보
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(gui_dir) # 25Embedded-trpg 폴더
        html_path = os.path.join(base_dir, "map", "index.html")
        
        if os.path.exists(html_path):
            self.webview_map.load(QUrl.fromLocalFile(html_path))
        else:
            # 파일을 찾지 못한 경우 에러 메시지 표시
            error_html = f"""
            <html>
                <body>
                    <h1>File Not Found</h1>
                    <p>Could not find the map file at:</p>
                    <p><b>{html_path}</b></p>
                    <p>Please check if the file exists and the path is correct.</p>
                </body>
            </html>
            """
            self.webview_map.setHtml(error_html)
            
        self.webview_map.setMinimumSize(800, 600)
        # 0,0 위치에 웹뷰를 추가하고, 1행 1열을 차지하도록 함
        grid_layout.addWidget(self.webview_map, 0, 0, 1, 1)
        
        # 기존 QGraphicsView 숨기기
        self.ui.graphicsView_map.setVisible(False)
        
        # 오버레이 프레임(frame_overlay)은 그대로 유지하고 최상단으로 올림
        self.ui.frame_overlay.raise_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
