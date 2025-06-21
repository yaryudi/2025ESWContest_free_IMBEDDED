import sys
import os
import subprocess
import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QScrollArea, QMainWindow, QSizePolicy, QFrame, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QCursor, QPainter, QColor, QLinearGradient, QFont, QTextDocument, QTextOption, QTextCursor, QTextBlockFormat
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal

class GameLaunchThread(QThread):
    """게임 실행을 위한 별도 스레드"""
    game_started = pyqtSignal()
    game_failed = pyqtSignal(str)
    
    def __init__(self, script_path, game_dir):
        super().__init__()
        self.script_path = script_path
        self.game_dir = game_dir
        
    def run(self):
        try:
            # 게임 실행
            process = subprocess.Popen([sys.executable, self.script_path], cwd=self.game_dir)
            self.game_started.emit()
        except Exception as e:
            self.game_failed.emit(str(e))

class LoadingDialog(QFrame):
    """로딩 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedSize(300, 150)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 로딩 텍스트
        self.loading_label = QLabel("게임을 시작하는 중...")
        self.loading_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 10px;
                text-align: center;
                background-color: #1a1a1a;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 8px;
            }
        """)
        self.progress_bar.setRange(0, 0)  # 무한 로딩
        
        layout.addWidget(self.loading_label)
        layout.addWidget(self.progress_bar)
        
        # 애니메이션 타이머
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_loading_text)
        self.dots_count = 0
        
    def start_animation(self):
        self.animation_timer.start(500)  # 0.5초마다 업데이트
        
    def stop_animation(self):
        self.animation_timer.stop()
        
    def update_loading_text(self):
        self.dots_count = (self.dots_count + 1) % 4
        dots = "." * self.dots_count
        self.loading_label.setText(f"게임을 시작하는 중{dots}")

class GameLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game Launcher")
        self.setGeometry(100, 100, 1280, 800)  # 1280x800 해상도에 맞게 조정
        self.setStyleSheet("background-color: #0f0f0f; color: white;")
        self.games = []
        self.current_game = None
        
        # 중복 실행 방지 변수들
        self.running_games = {}  # 실행 중인 게임들을 추적
        self.loading_dialog = None
        self.launch_thread = None
        
        # 실행 중인 게임 확인 타이머
        self.game_check_timer = QTimer(self)
        self.game_check_timer.timeout.connect(self.check_running_games)
        self.game_check_timer.start(5000)  # 5초마다 확인
        
        self.initUI()
        
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

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 컨테이너 (그라데이션 배경)
        main_container = QFrame()
        main_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:0.5 #2d2d2d, stop:1 #1a1a1a);
                border-radius: 0px;
            }
        """)

        # 왼쪽 게임 리스트 (앨범 스타일)
        self.game_list_widget = QWidget()
        self.game_list_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                margin: 10px;
            }
        """)
        
        self.game_buttons_layout = QVBoxLayout(self.game_list_widget)
        self.game_buttons_layout.setAlignment(Qt.AlignTop)
        self.game_buttons_layout.setSpacing(8)
        self.game_buttons_layout.setContentsMargins(15, 20, 15, 20)

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.game_buttons_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setFixedWidth(325)  # 400에서 2/3 정도로 줄임 (400 * 2/3 ≈ 267)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: rgba(255, 255, 255, 0.1);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(255, 255, 255, 0.5);
            }
        """)

        # 오른쪽 메인 콘텐츠 영역
        content_container = QWidget()
        content_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 20px;
                margin: 10px;
            }
        """)

        # 썸네일 (큰 앨범 아트 스타일)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                margin: 20px;
            }
        """)
        self.thumbnail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 게임 제목
        self.game_title_label = QLabel()
        self.game_title_label.setAlignment(Qt.AlignCenter)
        self.game_title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 28px;
                font-weight: bold;
                margin: 10px;
            }
        """)

        # 설명창 (더 깔끔한 디자인)
        self.description_box = QTextEdit()
        self.description_box.setReadOnly(True)
        self.description_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.description_box.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.08);
                color: #e0e0e0;
                border: none;
                border-radius: 15px;
                font-size: 16px;
                padding: 15px;
                margin: 10px;
            }
        """)
        
        # 줄 간격 조정을 위한 문서 설정
        doc = self.description_box.document()
        doc.setDocumentMargin(0)
        
        # 폰트 설정으로 줄 간격 조정
        font = self.description_box.font()
        font.setPointSize(16)
        font.setStyleStrategy(QFont.PreferAntialias)
        self.description_box.setFont(font)

        # 플레이 버튼 (유튜브 뮤직 스타일) - 1280x800 해상도에 맞게 조정
        self.play_button = QPushButton("▶ 재생")
        self.play_button.setFixedHeight(100)
        self.play_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.play_button.clicked.connect(self.launch_current_game)
        self.play_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff0000, stop:1 #cc0000);
                color: white;
                font-size: 20px;
                font-weight: bold;
                border: none;
                border-radius: 25px;
                margin: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff3333, stop:1 #dd0000);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #cc0000, stop:1 #aa0000);
            }
        """)

        # 메인 콘텐츠 레이아웃
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(10)
        content_layout.addWidget(self.thumbnail_label, stretch=3)
        content_layout.addWidget(self.game_title_label)
        content_layout.addWidget(self.description_box, stretch=2)
        content_layout.addWidget(self.play_button)

        # 전체 배치
        main_layout = QHBoxLayout(main_container)
        main_layout.addWidget(scroll_area, stretch=1)
        main_layout.addWidget(content_container, stretch=3)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().addWidget(main_container, 1)

        self.load_games()

    def load_games(self):
        games_root = "games"
        if not os.path.exists(games_root):
            return

        for game_dir in os.listdir(games_root):
            game_path = os.path.join(games_root, game_dir)
            if os.path.isdir(game_path):
                script_path = os.path.join(game_path, "app.py")
                if not os.path.exists(script_path):
                    continue

                thumb_files = [f for f in os.listdir(game_path) if f.lower().endswith((".jpg", ".png"))]
                thumbnail = os.path.join(game_path, thumb_files[0]) if thumb_files else ""

                desc_path = os.path.join(game_path, "description.txt")
                description = ""
                if os.path.exists(desc_path):
                    with open(desc_path, "r", encoding="utf-8") as f:
                        description = f.read().strip()

                game_data = {
                    "name": game_dir,
                    "path": script_path,
                    "thumbnail": thumbnail,
                    "description": description
                }
                self.games.append(game_data)

                # 앨범 스타일 게임 카드 생성
                self.create_game_card(game_data)

        if self.games:
            self.update_game_display(self.games[0])

    def create_game_card(self, game_data):
        # 카드 컨테이너
        card_widget = QFrame()
        card_widget.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.08);
                border-radius: 15px;
                margin: 4px;
            }
            QFrame:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
        """)
        card_widget.setFixedHeight(90)  # 높이를 100에서 90으로 조정 (설명 제거로 인해)

        # 카드 레이아웃
        card_layout = QHBoxLayout(card_widget)
        card_layout.setContentsMargins(15, 12, 15, 12)
        card_layout.setSpacing(15)

        # 썸네일
        thumb_label = QLabel()
        if game_data["thumbnail"]:
            thumb_pixmap = QPixmap(game_data["thumbnail"]).scaled(
                70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            thumb_label.setPixmap(thumb_pixmap)
        else:
            thumb_label.setText("🎮")
            thumb_label.setStyleSheet("font-size: 28px; color: #888;")
        thumb_label.setFixedSize(70, 70)
        thumb_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 3px;
            }
        """)

        # 게임 제목만 표시
        title_label = QLabel(game_data["name"])
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;  /* 16px에서 18px로 증가 */
                font-weight: bold;
            }
        """)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        card_layout.addWidget(thumb_label)
        card_layout.addWidget(title_label, stretch=1)

        # 클릭 이벤트
        card_widget.mousePressEvent = lambda event, g=game_data: self.update_game_display(g)

        self.game_buttons_layout.addWidget(card_widget)

    def update_game_display(self, game):
        self.current_game = game
        
        # 게임 제목 업데이트
        self.game_title_label.setText(game["name"])
        
        # 썸네일 업데이트
        pixmap = QPixmap(game["thumbnail"])
        if pixmap.isNull():
            self.thumbnail_label.setText("🎮\nNo Image")
            self.thumbnail_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    margin: 20px;
                    font-size: 48px;
                    color: #888;
                }
            """)
        else:
            # 1280x800 화면에 맞게 썸네일 표시 (이미지 잘림 방지)
            max_width = int(1280 * 0.35)  # 전체 너비의 35%
            max_height = int(800 * 0.5)   # 전체 높이의 50%
            
            # 이미지 비율을 유지하면서 최대 크기 계산
            original_size = pixmap.size()
            scale_factor = min(max_width / original_size.width(), max_height / original_size.height())
            
            # 너무 작아지지 않도록 최소 크기 설정
            scale_factor = max(scale_factor, 0.3)
            
            width_val = int(original_size.width() * scale_factor)
            height_val = int(original_size.height() * scale_factor)
            size = QSize(width_val, height_val)
            
            scaled = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            final = QPixmap(size)
            final.fill(QColor(0, 0, 0, 0))

            painter = QPainter(final)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 둥근 모서리로 그리기
            painter.setBrush(QColor(255, 255, 255, 25))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, size.width(), size.height(), 20, 20)
            
            # 이미지 그리기
            x = (size.width() - scaled.width()) // 2
            y = (size.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()

            self.thumbnail_label.setPixmap(final)
            self.thumbnail_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border-radius: 20px;
                    margin: 20px;
                }
            """)

        # 설명 업데이트
        self.description_box.setText(game.get("description", "게임 설명이 없습니다."))
        
        # 줄 간격 200%로 설정
        cursor = self.description_box.textCursor()
        cursor.select(QTextCursor.Document)
        block_format = QTextBlockFormat()
        block_format.setLineHeight(128, QTextBlockFormat.ProportionalHeight)
        cursor.setBlockFormat(block_format)
        cursor.clearSelection()  # 선택 효과 제거
        self.description_box.setTextCursor(cursor)
        
        # 버튼 텍스트 업데이트
        self.play_button.setText(f"▶ {game['name']} 시작")

    def launch_current_game(self):
        if not self.current_game:
            return
            
        game_name = self.current_game["name"]
        
        # 이미 실행 중인지 확인
        if game_name in self.running_games:
            return  # 메시지 없이 그냥 무시
            
        # 다른 게임이 실행 중인지 확인
        if self.running_games:
            running_game = list(self.running_games.keys())[0]
            QMessageBox.warning(self, "게임 실행 중", f"{running_game}이(가) 실행 중입니다.\n다른 게임을 실행하려면 먼저 현재 게임을 종료해주세요.")
            return
            
        # 로딩 다이얼로그 표시
        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.start_animation()
        
        # 중앙에 배치
        dialog_x = (self.width() - self.loading_dialog.width()) // 2
        dialog_y = (self.height() - self.loading_dialog.height()) // 2
        self.loading_dialog.move(dialog_x, dialog_y)
        self.loading_dialog.show()
        
        # 게임 실행 스레드 시작
        script_path = os.path.abspath(self.current_game["path"])
        game_dir = os.path.dirname(script_path)
        
        self.launch_thread = GameLaunchThread(script_path, game_dir)
        self.launch_thread.game_started.connect(lambda: self.on_game_started(game_name))
        self.launch_thread.game_failed.connect(self.on_game_failed)
        self.launch_thread.start()

    def on_game_started(self, game_name):
        """게임이 성공적으로 시작되었을 때"""
        # 실행 중인 게임 목록에 추가
        self.running_games[game_name] = True
        
        # 로딩 다이얼로그 숨기기
        if self.loading_dialog:
            self.loading_dialog.stop_animation()
            self.loading_dialog.hide()
            self.loading_dialog = None
            
    def on_game_failed(self, error):
        """게임 시작에 실패했을 때"""
        # 로딩 다이얼로그 숨기기
        if self.loading_dialog:
            self.loading_dialog.stop_animation()
            self.loading_dialog.hide()
            self.loading_dialog = None
            
        # 오류 메시지 표시
        QMessageBox.critical(self, "게임 시작 실패", f"게임을 시작하는 중 오류가 발생했습니다.\n{error}")

    def check_running_games(self):
        """실행 중인 게임들을 확인하고 종료된 게임들을 제거"""
        for game_name in list(self.running_games.keys()):
            # psutil을 사용하여 해당 게임 프로세스가 실행 중인지 확인
            # 실제로는 더 정확한 프로세스 매칭이 필요할 수 있음
            game_found = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any(game_name.lower() in arg.lower() for arg in cmdline):
                            game_found = True
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not game_found:
                del self.running_games[game_name]
                print(f"{game_name}이(가) 종료되었습니다.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_game:
            self.update_game_display(self.current_game)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 키보드 단축키 안내 출력
    print("\n=== 게임 런처 키보드 단축키 ===")
    print("ESC: 전체 화면 켜기/끄기")
    print("==============================\n")
    
    launcher = GameLauncher()
    launcher.show()
    sys.exit(app.exec_())
