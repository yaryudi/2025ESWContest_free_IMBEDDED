"""
카메라 초기화 로딩 다이얼로그
카메라 연결 및 초기화 과정을 사용자에게 보여주는 로딩 창입니다.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class CameraLoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("카메라 초기화")
        self.setFixedSize(450, 250)  # 크기를 늘려서 텍스트가 잘리지 않도록 함
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog)
        self.setModal(True)  # 모달 다이얼로그로 설정
        
        # 전체 창 스타일 설정
        self.setStyleSheet("""
            QDialog {
                background-color: #2C3E50;
                border: 3px solid #34495E;
                border-radius: 15px;
            }
        """)
        
        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.setSpacing(15)  # 간격을 줄여서 공간 확보
        layout.setContentsMargins(25, 25, 25, 25)  # 여백을 줄여서 공간 확보
        
        # 제목 라벨
        self.title_label = QLabel("카메라 초기화 중...")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setStyleSheet("""
            QLabel {
                color: #3498DB;
                background-color: transparent;
            }
        """)
        layout.addWidget(self.title_label)
        
        # 진행률 표시 라벨
        self.progress_label = QLabel("시도 중... 1/3")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setFont(QFont("Arial", 14))
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #ECF0F1;
                background-color: transparent;
            }
        """)
        layout.addWidget(self.progress_label)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #34495E;
                border-radius: 8px;
                text-align: center;
                background-color: #34495E;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 상태 메시지 라벨
        self.status_label = QLabel("카메라 연결을 시도하고 있습니다...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setWordWrap(True)  # 텍스트 래핑 활성화
        self.status_label.setMinimumHeight(40)  # 최소 높이 설정
        self.status_label.setStyleSheet("""
            QLabel {
                color: #BDC3C7;
                background-color: transparent;
                padding: 5px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # 버튼 레이아웃 (처음에는 숨김)
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)
        self.button_layout.setContentsMargins(0, 10, 0, 0)  # 상단 여백 추가
        
        # 재시도 버튼
        self.retry_button = QPushButton("재시도")
        self.retry_button.setFixedSize(100, 40)
        self.retry_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.retry_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #21618C;
            }
        """)
        self.retry_button.hide()
        self.button_layout.addWidget(self.retry_button)
        
        # 종료 버튼
        self.exit_button = QPushButton("종료")
        self.exit_button.setFixedSize(100, 40)
        self.exit_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)
        self.exit_button.hide()
        self.button_layout.addWidget(self.exit_button)
        
        layout.addLayout(self.button_layout)
        
        self.setLayout(layout)
        
        # 중앙에 위치시키기
        self.center_on_screen()
        
    def center_on_screen(self):
        """화면 중앙에 다이얼로그 위치"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            # 부모가 없는 경우 화면 중앙에 위치
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().availableGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def update_progress(self, attempt, max_attempts, status_message=""):
        """진행률 업데이트"""
        self.progress_label.setText(f"시도 중... {attempt}/{max_attempts}")
        self.progress_bar.setValue(attempt)
        
        if status_message:
            self.status_label.setText(status_message)
        
        # UI 업데이트 강제 실행
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
    
    def show_success(self):
        """성공 메시지 표시"""
        self.title_label.setText("카메라 초기화 완료!")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #27AE60;
                background-color: transparent;
            }
        """)
        self.progress_label.setText("연결 성공!")
        self.status_label.setText("카메라가 성공적으로 연결되었습니다.")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        # UI 업데이트 강제 실행
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
    
    def reset_for_retry(self):
        """재시도를 위해 다이얼로그를 초기 상태로 리셋"""
        self.title_label.setText("카메라 초기화 중...")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #3498DB;
                background-color: transparent;
            }
        """)
        self.progress_label.setText("시도 중... 1/3")
        self.progress_bar.setValue(0)
        self.status_label.setText("카메라 연결을 시도하고 있습니다...")
        
        # 버튼 숨기기
        self.retry_button.hide()
        self.exit_button.hide()
        
        # UI 업데이트 강제 실행
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
    
    def show_error(self, error_message="카메라 연결 실패: 연결 상태를 다시 확인해주세요."):
        """에러 메시지 표시 및 재시도/종료 버튼 표시"""
        self.title_label.setText("카메라 초기화 실패")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #E74C3C;
                background-color: transparent;
            }
        """)
        self.progress_label.setText("연결 실패")
        self.status_label.setText(error_message)
        
        # 재시도/종료 버튼 표시
        self.retry_button.show()
        self.exit_button.show()
        
        # UI 업데이트 강제 실행
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

