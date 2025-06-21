import sys
import time
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QPixmap
import serial
import numpy as np
import pyautogui

# pyautogui 설정
pyautogui.FAILSAFE = False

class CalibrationPoint(QFrame):
    """캘리브레이션 포인트를 표시하는 위젯"""
    def __init__(self, position, parent=None):
        super().__init__(parent)
        self.position = position  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        self.is_active = False
        self.is_completed = False
        self.setFixedSize(200, 150)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #cccccc;
                border-radius: 10px;
                background-color: #f0f0f0;
            }
        """)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 포인트 위치에 따라 원 그리기
        if self.position == 'top-left':
            x, y = 30, 30
        elif self.position == 'top-right':
            x, y = self.width() - 30, 30
        elif self.position == 'bottom-left':
            x, y = 30, self.height() - 30
        else:  # bottom-right
            x, y = self.width() - 30, self.height() - 30
            
        # 상태에 따른 색상 설정
        if self.is_completed:
            color = QColor(0, 255, 0)  # 녹색 (완료)
        elif self.is_active:
            color = QColor(255, 165, 0)  # 주황색 (활성)
        else:
            color = QColor(128, 128, 128)  # 회색 (비활성)
            
        painter.setPen(QPen(color, 4))
        painter.setBrush(color)
        painter.drawEllipse(x-10, y-10, 20, 20)
        
        # 텍스트 표시
        painter.setPen(QPen(Qt.black, 2))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        
        text = self.position.replace('-', ' ').title()
        painter.drawText(self.rect(), Qt.AlignCenter, text)

class CalibrationThread(QThread):
    """터치 감지를 위한 별도 스레드"""
    touch_detected = pyqtSignal(tuple)  # (row, col) 좌표
    
    def __init__(self, ser, offset):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.running = True
        self.NUM_ROWS = 40
        self.NUM_COLS = 30
        self.FRAME_SIZE = self.NUM_ROWS * self.NUM_COLS
        self.TOUCH_THRESHOLD = 30
        self.last_touch_time = 0
        self.cool_down_time = 3.0  # 3초 쿨다운
        
    def run(self):
        while self.running:
            try:
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)
                    
                    # 터치 감지
                    peak = self.find_peak(filtered)
                    if peak:
                        r, c, v = peak
                        if v >= self.TOUCH_THRESHOLD:
                            current_time = time.time()
                            # 3초 쿨다운 체크
                            if current_time - self.last_touch_time >= self.cool_down_time:
                                self.touch_detected.emit((r, c))
                                self.last_touch_time = current_time
                            
            except Exception as e:
                print(f"터치 감지 오류: {e}")
                
            time.sleep(0.01)  # 10ms 간격
    
    def read_frame(self):
        raw = self.ser.read(self.FRAME_SIZE)
        if len(raw) != self.FRAME_SIZE:
            return None
        
        data = np.frombuffer(raw, dtype=np.uint8)
        frame = np.zeros((self.NUM_ROWS, self.NUM_COLS), dtype=np.uint8)
        
        for row in range(self.NUM_ROWS):
            for mux_ch in range(8):
                for dev in range(4):
                    col = dev * 8 + mux_ch
                    if col >= self.NUM_COLS:
                        continue
                    idx = row * self.NUM_COLS + mux_ch * 4 + dev
                    if idx < len(data):
                        if col == 15:
                            frame[row, 23] = data[idx]
                        elif col == 7:
                            frame[row, 16] = data[idx]
                        else:
                            frame[row, col] = data[idx]
        
        return frame
    
    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask
    
    def find_peak(self, arr):
        candidates = sorted(
            ((v, r, c) for (r, c), v in np.ndenumerate(arr)),
            key=lambda x: x[0], reverse=True
        )
        for value, r, c in candidates:
            if value < self.TOUCH_THRESHOLD:
                continue
            if np.max(arr[r, :]) > value or np.max(arr[:, c]) > value:
                continue
            return r, c, value
        return None
    
    def stop(self):
        self.running = False

class MouseControlThread(QThread):
    """마우스 제어를 위한 별도 스레드"""
    def __init__(self, ser, offset, calibration_matrix):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.calibration_matrix = calibration_matrix
        self.running = True
        self.NUM_ROWS = 40
        self.NUM_COLS = 30
        self.FRAME_SIZE = self.NUM_ROWS * self.NUM_COLS
        self.TOUCH_THRESHOLD = 30
        self.SCREEN_W = 1980
        self.SCREEN_H = 1080
        
    def run(self):
        while self.running:
            try:
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)
                    
                    # 터치 감지
                    peak = self.find_peak(filtered)
                    if peak:
                        r, c, v = peak
                        if v >= self.TOUCH_THRESHOLD:
                            # 캘리브레이션된 좌표로 변환
                            screen_coords = self.map_touch_to_screen(r, c)
                            if screen_coords:
                                x_px, y_px = screen_coords
                                pyautogui.moveTo(x_px, y_px)
                            
            except Exception as e:
                print(f"마우스 제어 오류: {e}")
                
            time.sleep(0.01)  # 10ms 간격
    
    def read_frame(self):
        raw = self.ser.read(self.FRAME_SIZE)
        if len(raw) != self.FRAME_SIZE:
            return None
        
        data = np.frombuffer(raw, dtype=np.uint8)
        frame = np.zeros((self.NUM_ROWS, self.NUM_COLS), dtype=np.uint8)
        
        for row in range(self.NUM_ROWS):
            for mux_ch in range(8):
                for dev in range(4):
                    col = dev * 8 + mux_ch
                    if col >= self.NUM_COLS:
                        continue
                    idx = row * self.NUM_COLS + mux_ch * 4 + dev
                    if idx < len(data):
                        if col == 15:
                            frame[row, 23] = data[idx]
                        elif col == 7:
                            frame[row, 16] = data[idx]
                        else:
                            frame[row, col] = data[idx]
        
        return frame
    
    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask
    
    def find_peak(self, arr):
        candidates = sorted(
            ((v, r, c) for (r, c), v in np.ndenumerate(arr)),
            key=lambda x: x[0], reverse=True
        )
        for value, r, c in candidates:
            if value < self.TOUCH_THRESHOLD:
                continue
            if np.max(arr[r, :]) > value or np.max(arr[:, c]) > value:
                continue
            return r, c, value
        return None
    
    def map_touch_to_screen(self, r, c):
        """터치 좌표를 화면 좌표로 변환"""
        if self.calibration_matrix is None:
            return None
        
        x_matrix, y_matrix = self.calibration_matrix
        screen_x = int(np.polyval(x_matrix, r))
        screen_y = int(np.polyval(y_matrix, c))
        
        # 화면 범위 제한
        screen_x = np.clip(screen_x, 0, self.SCREEN_W-1)
        screen_y = np.clip(screen_y, 0, self.SCREEN_H-1)
        
        return screen_x, screen_y
    
    def stop(self):
        self.running = False

class CalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ser = None
        self.calibration_thread = None
        self.mouse_control_thread = None  # 마우스 제어 스레드 추가
        self.current_step = 0
        self.calibration_data = {
            'touch_points': [],
            'screen_points': [],
            'matrix': None
        }
        self.SCREEN_W = 1280
        self.SCREEN_H = 800
        self.last_touch_time = time.time()
        self.last_touch_coords = None  # 마지막 터치 좌표 저장
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('터치패드 캘리브레이션')
        
        # 전체화면 설정
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)  # 커서 숨기기
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 제목
        title_label = QLabel('터치패드 캘리브레이션')
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 상태 메시지
        self.status_label = QLabel('10초 동안 터치 신호가 없으면 캘리브레이션이 시작됩니다...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 10px;
                background-color: #e8f4fd;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # 타이머
        self.timer_label = QLabel('10초')
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ff6600;
                padding: 10px;
            }
        """)
        main_layout.addWidget(self.timer_label)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # 캘리브레이션 포인트들
        points_layout = QHBoxLayout()
        
        self.top_left = CalibrationPoint('top-left')
        self.top_right = CalibrationPoint('top-right')
        self.bottom_left = CalibrationPoint('bottom-left')
        self.bottom_right = CalibrationPoint('bottom-right')
        
        points_layout.addWidget(self.top_left)
        points_layout.addWidget(self.top_right)
        points_layout.addWidget(self.bottom_left)
        points_layout.addWidget(self.bottom_right)
        
        main_layout.addLayout(points_layout)
        
        # 시작 버튼 (초기에는 숨김)
        self.start_button = QPushButton('캘리브레이션 시작')
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_button.clicked.connect(self.start_calibration)
        self.start_button.hide()  # 초기에는 숨김
        main_layout.addWidget(self.start_button)
        
        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.countdown = 10
        
        # 터치 감지 타이머
        self.touch_timer = QTimer()
        self.touch_timer.timeout.connect(self.on_touch_timeout)
        self.touch_timer.setSingleShot(True)
        
        # 10초 대기 타이머
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(self.on_wait_timeout)
        self.wait_timer.setSingleShot(True)
        
        # 초기 10초 대기 시작
        self.start_wait_period()
        
    def start_wait_period(self):
        """10초 대기 기간 시작"""
        self.countdown = 10
        self.timer.start(1000)  # 1초마다
        self.wait_timer.start(10000)  # 10초 후 타임아웃
        self.status_label.setText('10초 동안 터치 신호가 없으면 캘리브레이션이 시작됩니다...')
        
        # 10초 대기 중에도 터치 감지를 위해 시리얼 연결 및 스레드 시작
        try:
            self.ser = serial.Serial('COM9', 115200, timeout=1)
            self.offset = self.calibrate_offset()
            self.calibration_thread = CalibrationThread(self.ser, self.offset)
            self.calibration_thread.touch_detected.connect(self.on_touch_detected)
            self.calibration_thread.start()
        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}')
        
    def on_wait_timeout(self):
        """10초 대기 타임아웃 시 호출"""
        self.timer.stop()
        self.timer_label.setText('캘리브레이션 시작!')
        self.status_label.setText('캘리브레이션을 시작합니다...')
        self.start_button.show()
        self.start_button.click()
        
    def start_calibration(self):
        try:
            # 이미 시리얼이 연결되어 있으면 재사용
            if self.ser is None:
                self.ser = serial.Serial('COM9', 115200, timeout=1)
                self.offset = self.calibrate_offset()
                self.calibration_thread = CalibrationThread(self.ser, self.offset)
                self.calibration_thread.touch_detected.connect(self.on_touch_detected)
                self.calibration_thread.start()
            
            # UI 업데이트
            self.start_button.setEnabled(False)
            self.start_button.setText('캘리브레이션 진행 중...')
            
            # 첫 번째 단계 시작
            self.start_next_step()
            
        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}')
    
    def calibrate_offset(self):
        """오프셋 캘리브레이션"""
        frames = []
        for _ in range(10):
            raw = self.ser.read(1200)
            if len(raw) == 1200:
                data = np.frombuffer(raw, dtype=np.uint8)
                frame = data.reshape(40, 30)
                frames.append(frame)
            time.sleep(0.01)
        
        return np.mean(frames, axis=0).astype(np.float32)
    
    def start_next_step(self):
        """다음 캘리브레이션 단계 시작"""
        self.current_step += 1
        self.progress_bar.setValue(self.current_step)
        
        # 마지막 터치 좌표 초기화
        self.last_touch_coords = None
        
        # 모든 포인트 비활성화
        for point in [self.top_left, self.top_right, self.bottom_left, self.bottom_right]:
            point.is_active = False
            point.update()
        
        if self.current_step == 1:
            self.status_label.setText('좌측 상단을 터치해주세요')
            self.top_left.is_active = True
            self.top_left.update()
        elif self.current_step == 2:
            self.status_label.setText('우측 상단을 터치해주세요')
            self.top_right.is_active = True
            self.top_right.update()
        elif self.current_step == 3:
            self.status_label.setText('좌측 하단을 터치해주세요')
            self.bottom_left.is_active = True
            self.bottom_left.update()
        elif self.current_step == 4:
            self.status_label.setText('우측 하단을 터치해주세요')
            self.bottom_right.is_active = True
            self.bottom_right.update()
        else:
            self.finish_calibration()
            return
        
        # 타이머 시작
        self.countdown = 5
        self.timer.start(1000)  # 1초마다
        self.touch_timer.start(5000)  # 5초 후 타임아웃
    
    def on_touch_detected(self, coords):
        """터치 감지 시 호출"""
        r, c = coords
        
        # 캘리브레이션이 시작되지 않은 상태에서 터치가 감지되면 10초 대기 리셋
        if self.current_step == 0:
            self.reset_wait_period()
            return
        
        # 캘리브레이션 진행 중일 때는 5초 타이머 리셋
        if self.current_step >= 1 and self.current_step <= 4:
            self.last_touch_coords = (r, c)  # 마지막 터치 좌표 저장
            self.reset_calibration_timer()
            return
    
    def reset_calibration_timer(self):
        """캘리브레이션 중 5초 타이머 리셋"""
        self.touch_timer.stop()
        self.touch_timer.start(5000)  # 다시 5초 시작
        self.countdown = 5
        self.timer.start(1000)  # 1초마다 업데이트
        self.timer_label.setText('5초')
        self.status_label.setText(f'{self.get_step_message()} (터치 감지됨 - 5초 리셋)')
    
    def get_step_message(self):
        """현재 단계에 따른 메시지 반환"""
        if self.current_step == 1:
            return '좌측 상단을 터치해주세요'
        elif self.current_step == 2:
            return '우측 상단을 터치해주세요'
        elif self.current_step == 3:
            return '좌측 하단을 터치해주세요'
        elif self.current_step == 4:
            return '우측 하단을 터치해주세요'
        return ''
    
    def on_touch_timeout(self):
        """5초 타임아웃 시 호출"""
        self.timer.stop()
        self.timer_label.setText('시간 초과')
        self.status_label.setText('5초 동안 새로운 터치가 없어 다음 단계로 진행합니다.')
        
        # 마지막 터치 좌표가 있으면 데이터 저장
        if self.last_touch_coords:
            r, c = self.last_touch_coords
            
            # 현재 단계에 따른 화면 좌표
            if self.current_step == 1:
                screen_coords = (0, 0)
            elif self.current_step == 2:
                screen_coords = (self.SCREEN_W-1, 0)
            elif self.current_step == 3:
                screen_coords = (0, self.SCREEN_H-1)
            elif self.current_step == 4:
                screen_coords = (self.SCREEN_W-1, self.SCREEN_H-1)
            else:
                return
            
            # 데이터 저장
            self.calibration_data['touch_points'].append((r, c))
            self.calibration_data['screen_points'].append(screen_coords)
            
            # 현재 포인트 완료 표시
            if self.current_step == 1:
                self.top_left.is_completed = True
                self.top_left.is_active = False
                self.top_left.update()
            elif self.current_step == 2:
                self.top_right.is_completed = True
                self.top_right.is_active = False
                self.top_right.update()
            elif self.current_step == 3:
                self.bottom_left.is_completed = True
                self.bottom_left.is_active = False
                self.bottom_left.update()
            elif self.current_step == 4:
                self.bottom_right.is_completed = True
                self.bottom_right.is_active = False
                self.bottom_right.update()
            
            # 마지막 터치 좌표 초기화
            self.last_touch_coords = None
        
        # 다음 단계로
        QTimer.singleShot(2000, self.start_next_step)
    
    def update_timer(self):
        """타이머 업데이트"""
        self.countdown -= 1
        self.timer_label.setText(f'{self.countdown}초')
        
        if self.countdown <= 0:
            self.timer.stop()
    
    def finish_calibration(self):
        """캘리브레이션 완료"""
        if len(self.calibration_data['touch_points']) >= 4:
            # 변환 행렬 계산
            touch_points = np.array(self.calibration_data['touch_points'])
            screen_points = np.array(self.calibration_data['screen_points'])
            
            x_matrix = np.polyfit(touch_points[:, 0], screen_points[:, 0], 1)
            y_matrix = np.polyfit(touch_points[:, 1], screen_points[:, 1], 1)
            
            self.calibration_data['matrix'] = (x_matrix, y_matrix)
            
            self.status_label.setText('캘리브레이션이 완료되었습니다! 마우스 제어를 시작합니다.')
            self.timer_label.setText('완료')
            self.progress_bar.setValue(4)
            
            # 캘리브레이션 스레드 정지
            if self.calibration_thread:
                self.calibration_thread.stop()
                self.calibration_thread.wait()
            
            # 마우스 제어 스레드 시작
            self.mouse_control_thread = MouseControlThread(
                self.ser, self.offset, self.calibration_data['matrix']
            )
            self.mouse_control_thread.start()
            
            # GUI 최소화 (마우스 제어가 가능하도록)
            QTimer.singleShot(3000, self.minimize_window)
            
        else:
            self.status_label.setText('캘리브레이션에 실패했습니다. 다시 시도해주세요.')
        
        self.start_button.setEnabled(True)
        self.start_button.setText('다시 시작')
    
    def minimize_window(self):
        """창을 최소화하여 마우스 제어가 가능하도록 함"""
        self.showMinimized()
        self.status_label.setText('창이 최소화되었습니다. 터치패드로 마우스를 제어할 수 있습니다.')
    
    def closeEvent(self, event):
        """창 닫을 때 정리"""
        if self.calibration_thread:
            self.calibration_thread.stop()
            self.calibration_thread.wait()
        
        if self.mouse_control_thread:
            self.mouse_control_thread.stop()
            self.mouse_control_thread.wait()
        
        if self.ser:
            self.ser.close()
        
        event.accept()
    
    def reset_wait_period(self):
        """10초 대기 기간 리셋"""
        self.timer.stop()
        self.wait_timer.stop()
        self.countdown = 10
        self.timer.start(1000)
        self.wait_timer.start(10000)
        self.status_label.setText('터치가 감지되어 10초 대기가 리셋되었습니다...')
        self.timer_label.setText('10초')

def main():
    app = QApplication(sys.argv)
    
    # 라즈베리파이에서 전체화면으로 실행
    if 'linux' in sys.platform:
        app.setOverrideCursor(Qt.BlankCursor)  # 커서 숨기기
    
    window = CalibrationGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
