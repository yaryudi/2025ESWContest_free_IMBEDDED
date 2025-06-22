import sys
import time
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
import serial
import serial.tools.list_ports
import numpy as np
import pyautogui

# pyautogui 설정
pyautogui.FAILSAFE = False

def find_serial_port():
    """시리얼 포트 자동 감지"""
    ports = list(serial.tools.list_ports.comports())
    
    # 라즈베리파이에서 일반적인 포트들 우선 순위
    preferred_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']
    
    # 우선 순위 포트 확인
    for port_name in preferred_ports:
        for port in ports:
            if port.device == port_name:
                return port_name
    
    # 사용 가능한 첫 번째 포트 반환
    if ports:
        return ports[0].device
    
    return None

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
    error_occurred = pyqtSignal(str)  # 에러 메시지
    
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
        self.last_touch_coords = None  # 이전 터치 좌표 (row, col)
        self.cool_down_time = 3.0  # 3초 쿨다운
        self.double_click_threshold = 1.0  # 1초 내 같은 위치 터치 방지
        self.position_threshold = 3  # 3칸 이내를 같은 위치로 간주 (터치패드 좌표 기준)
        
        # 사분면 정의 (clikmap_raspi.py 참고)
        center_r, center_c = self.NUM_ROWS//2, self.NUM_COLS//2
        self.quadrants = {
            'top-left': (slice(0, center_r), slice(0, center_c)),
            'top-right': (slice(0, center_r), slice(center_c, self.NUM_COLS)),
            'bottom-left': (slice(center_r, self.NUM_ROWS), slice(0, center_c)),
            'bottom-right': (slice(center_r, self.NUM_ROWS), slice(center_c, self.NUM_COLS)),
        }
        
    def run(self):
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    self.error_occurred.emit("시리얼 포트가 닫혔습니다.")
                    break
                    
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)
                    
                    # 사분면별로 터치 감지 (clikmap_raspi.py 방식)
                    touch_found = False
                    for quadrant_name, (rs, cs) in self.quadrants.items():
                        if touch_found:
                            break
                            
                        peak = self.find_peak(filtered, rs, cs)
                        if peak:
                            r, c, v = peak
                            if v >= self.TOUCH_THRESHOLD:
                                current_time = time.time()
                                
                                # 더블클릭 방지 검사
                                if self.is_double_click_calibration(r, c, current_time):
                                    print(f"더블클릭 방지: ({r}, {c}) - 이전 터치와 너무 가까움")
                                    continue
                                
                                # 3초 쿨다운 체크
                                if current_time - self.last_touch_time >= self.cool_down_time:
                                    print(f"터치 감지: {quadrant_name} - ({r}, {c}) - 값: {v}")
                                    self.touch_detected.emit((r, c))
                                    self.last_touch_time = current_time
                                    self.last_touch_coords = (r, c)
                                    touch_found = True
                            
            except serial.SerialException as e:
                self.error_occurred.emit(f"시리얼 통신 오류: {e}")
                break
            except Exception as e:
                print(f"터치 감지 오류: {e}")
                
            time.sleep(0.01)  # 10ms 간격
    
    def read_frame(self):
        try:
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
        except Exception as e:
            print(f"프레임 읽기 오류: {e}")
            return None
    
    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask
    
    def find_peak(self, arr, rs, cs):
        """사분면별 피크 검출 (clikmap_raspi.py 방식)"""
        try:
            sub = arr[rs, cs]
            candidates = sorted(
                ((v, r, c) for (r, c), v in np.ndenumerate(sub)),
                key=lambda x: x[0], reverse=True
            )
            for value, r_sub, c_sub in candidates:
                if value < self.TOUCH_THRESHOLD:
                    continue
                r = rs.start + r_sub
                c = cs.start + c_sub
                if np.max(arr[r, :]) > value or np.max(arr[:, c]) > value:
                    continue
                return r, c, value
            return None
        except Exception as e:
            print(f"피크 검출 오류: {e}")
            return None
    
    def is_double_click_calibration(self, r, c, current_time):
        """캘리브레이션 중 더블클릭 여부를 판단하는 메서드"""
        # 이전 터치가 없으면 더블클릭이 아님
        if self.last_touch_coords is None:
            return False
        
        # 시간 차이 계산
        time_diff = current_time - self.last_touch_time
        
        # 시간 임계값을 초과하면 더블클릭이 아님
        if time_diff > self.double_click_threshold:
            return False
        
        # 위치 차이 계산 (터치패드 좌표 기준)
        last_r, last_c = self.last_touch_coords
        distance = ((r - last_r) ** 2 + (c - last_c) ** 2) ** 0.5
        
        # 위치가 너무 가까우면 더블클릭으로 판단
        if distance <= self.position_threshold:
            return True
        
        return False
    
    def stop(self):
        self.running = False

class MouseControlThread(QThread):
    """마우스 제어를 위한 별도 스레드"""
    error_occurred = pyqtSignal(str)  # 에러 메시지
    
    def __init__(self, ser, offset, calibration_matrix):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.calibration_matrix = calibration_matrix
        self.running = True
        self.NUM_ROWS = 40
        self.NUM_COLS = 30
        self.FRAME_SIZE = self.NUM_ROWS * self.NUM_COLS
        self.TOUCH_THRESHOLD = 20
        self.SCREEN_W = 1280
        self.SCREEN_H = 800
        
        # 더블클릭 방지를 위한 변수들
        self.last_touch_time = 0
        self.last_touch_coords = None
        self.double_click_threshold = 0.5  # 0.5초 내 같은 위치 터치 방지
        self.position_threshold = 5  # 5픽셀 이내를 같은 위치로 간주
        
        # 사분면 정의 (clikmap_raspi.py 참고)
        center_r, center_c = self.NUM_ROWS//2, self.NUM_COLS//2
        self.quadrants = {
            'top-left': (slice(0, center_r), slice(0, center_c)),
            'top-right': (slice(0, center_r), slice(center_c, self.NUM_COLS)),
            'bottom-left': (slice(center_r, self.NUM_ROWS), slice(0, center_c)),
            'bottom-right': (slice(center_r, self.NUM_ROWS), slice(center_c, self.NUM_COLS)),
        }
        
    def run(self):
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    self.error_occurred.emit("시리얼 포트가 닫혔습니다.")
                    break
                    
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)
                    
                    # 사분면별로 터치 감지 (clikmap_raspi.py 방식)
                    touch_found = False
                    for quadrant_name, (rs, cs) in self.quadrants.items():
                        if touch_found:
                            break
                            
                        peak = self.find_peak(filtered, rs, cs)
                        if peak:
                            r, c, v = peak
                            if v >= self.TOUCH_THRESHOLD:
                                # 캘리브레이션된 좌표로 변환
                                screen_coords = self.map_touch_to_screen(r, c)
                                if screen_coords:
                                    x_px, y_px = screen_coords
                                    
                                    # 더블클릭 방지 검사
                                    if self.is_double_click(x_px, y_px):
                                        print(f"더블클릭 방지: ({x_px}, {y_px}) - 이전 터치와 너무 가까움")
                                        continue
                                    
                                    try:
                                        pyautogui.moveTo(x_px, y_px)
                                        print(f"마우스 이동: {quadrant_name} - ({r}, {c}) -> ({x_px}, {y_px})")
                                        
                                        # 터치 정보 업데이트
                                        self.last_touch_time = time.time()
                                        self.last_touch_coords = (x_px, y_px)
                                        
                                    except Exception as e:
                                        print(f"마우스 이동 오류: {e}")
                                    touch_found = True
                            
            except serial.SerialException as e:
                self.error_occurred.emit(f"시리얼 통신 오류: {e}")
                break
            except Exception as e:
                print(f"마우스 제어 오류: {e}")
                
            time.sleep(0.01)  # 10ms 간격
    
    def read_frame(self):
        try:
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
        except Exception as e:
            print(f"프레임 읽기 오류: {e}")
            return None
    
    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask
    
    def find_peak(self, arr, rs, cs):
        """사분면별 피크 검출 (clikmap_raspi.py 방식)"""
        try:
            sub = arr[rs, cs]
            candidates = sorted(
                ((v, r, c) for (r, c), v in np.ndenumerate(sub)),
                key=lambda x: x[0], reverse=True
            )
            for value, r_sub, c_sub in candidates:
                if value < self.TOUCH_THRESHOLD:
                    continue
                r = rs.start + r_sub
                c = cs.start + c_sub
                if np.max(arr[r, :]) > value or np.max(arr[:, c]) > value:
                    continue
                return r, c, value
            return None
        except Exception as e:
            print(f"피크 검출 오류: {e}")
            return None
    
    def map_touch_to_screen(self, r, c):
        """터치 좌표를 화면 좌표로 변환 (clikmap_raspi.py 방식)"""
        try:
            if self.calibration_matrix is None:
                # 캘리브레이션 없을 때는 기본 변환 사용
                x_px = int(round((self.NUM_ROWS - 1 - r) / (self.NUM_ROWS - 1) * (self.SCREEN_W - 1)))
                y_px = int(round((self.NUM_COLS - 1 - c) / (self.NUM_COLS - 1) * (self.SCREEN_H - 1)))
                y_px = self.SCREEN_H - y_px
                return x_px, y_px
            else:
                # 캘리브레이션된 변환 사용
                x_matrix, y_matrix = self.calibration_matrix
                screen_x = int(np.polyval(x_matrix, r))
                screen_y = int(np.polyval(y_matrix, c))
                
                # 화면 범위 제한
                screen_x = np.clip(screen_x, 0, self.SCREEN_W-1)
                screen_y = np.clip(screen_y, 0, self.SCREEN_H-1)
                
                return screen_x, screen_y
        except Exception as e:
            print(f"좌표 변환 오류: {e}")
            return None
    
    def is_double_click(self, x, y):
        """더블클릭 여부를 판단하는 메서드"""
        current_time = time.time()
        
        # 이전 터치가 없으면 더블클릭이 아님
        if self.last_touch_coords is None:
            return False
        
        # 시간 차이 계산
        time_diff = current_time - self.last_touch_time
        
        # 시간 임계값을 초과하면 더블클릭이 아님
        if time_diff > self.double_click_threshold:
            return False
        
        # 위치 차이 계산
        last_x, last_y = self.last_touch_coords
        distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
        
        # 위치가 너무 가까우면 더블클릭으로 판단
        if distance <= self.position_threshold:
            return True
        
        return False
    
    def stop(self):
        self.running = False

class CalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ser = None
        self.calibration_thread = None
        self.mouse_control_thread = None
        self.current_step = 0
        self.calibration_data = {
            'touch_points': [],
            'screen_points': [],
            'matrix': None
        }
        self.SCREEN_W = 1280
        self.SCREEN_H = 800
        self.last_touch_time = time.time()
        self.last_touch_coords = None
        self.offset = None
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
        self.status_label = QLabel('시리얼 포트를 연결하는 중...')
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
        self.timer_label = QLabel('5초')
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
        self.countdown = 5
        
        # 터치 감지 타이머
        self.touch_timer = QTimer()
        self.touch_timer.timeout.connect(self.on_touch_timeout)
        self.touch_timer.setSingleShot(True)
        
        # 5초 대기 타이머
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(self.on_wait_timeout)
        self.wait_timer.setSingleShot(True)
        
        # 초기 연결 시도
        QTimer.singleShot(1000, self.try_connect_serial)
        
    def try_connect_serial(self):
        """시리얼 포트 연결 시도"""
        try:
            port = find_serial_port()
            if port is None:
                self.status_label.setText('시리얼 포트를 찾을 수 없습니다. 장치를 연결해주세요.')
                # 5초 후 다시 시도
                QTimer.singleShot(5000, self.try_connect_serial)
                return
            
            print(f"시리얼 포트 연결 시도: {port}")
            self.ser = serial.Serial(port, 115200, timeout=1)
            
            # 오프셋 캘리브레이션
            self.status_label.setText('오프셋 캘리브레이션 중... 터치패드를 건드리지 마세요.')
            QTimer.singleShot(1000, self.calibrate_offset)
            
        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}')
            # 5초 후 다시 시도
            QTimer.singleShot(5000, self.try_connect_serial)
    
    def calibrate_offset(self):
        """오프셋 캘리브레이션 (수정된 버전)"""
        try:
            frames = []
            for i in range(10):
                frame = self.read_frame_for_calibration()
                if frame is not None:
                    frames.append(frame)
                    self.status_label.setText(f'오프셋 캘리브레이션 중... ({i+1}/10)')
                    QApplication.processEvents()  # UI 업데이트
                time.sleep(0.1)
            
            if len(frames) >= 5:  # 최소 5개 프레임이 있으면 계속 진행
                self.offset = np.mean(frames, axis=0).astype(np.float32)
                print("오프셋 캘리브레이션 완료")
                self.start_wait_period()
            else:
                self.status_label.setText('오프셋 캘리브레이션 실패. 다시 시도합니다.')
                QTimer.singleShot(2000, self.calibrate_offset)
                
        except Exception as e:
            print(f"오프셋 캘리브레이션 오류: {e}")
            self.status_label.setText(f'오프셋 캘리브레이션 오류: {e}')
            QTimer.singleShot(5000, self.try_connect_serial)
    
    def read_frame_for_calibration(self):
        """캘리브레이션용 프레임 읽기 (read_frame과 동일한 방식)"""
        try:
            if not self.ser or not self.ser.is_open:
                return None
                
            raw = self.ser.read(1200)  # 40 * 30 = 1200
            if len(raw) != 1200:
                return None
            
            data = np.frombuffer(raw, dtype=np.uint8)
            frame = np.zeros((40, 30), dtype=np.uint8)
            
            for row in range(40):
                for mux_ch in range(8):
                    for dev in range(4):
                        col = dev * 8 + mux_ch
                        if col >= 30:
                            continue
                        idx = row * 30 + mux_ch * 4 + dev
                        if idx < len(data):
                            if col == 15:
                                frame[row, 23] = data[idx]
                            elif col == 7:
                                frame[row, 16] = data[idx]
                            else:
                                frame[row, col] = data[idx]
            
            return frame
        except Exception as e:
            print(f"캘리브레이션용 프레임 읽기 오류: {e}")
            return None
        
    def start_wait_period(self):
        """5초 대기 기간 시작"""
        self.countdown = 5
        self.timer.start(1000)  # 1초마다
        self.wait_timer.start(5000)  # 5초 후 타임아웃
        self.status_label.setText('5초 동안 터치 신호가 없으면 캘리브레이션이 시작됩니다...')
        
        # 터치 감지 스레드 시작
        try:
            if self.offset is not None:
                self.calibration_thread = CalibrationThread(self.ser, self.offset)
                self.calibration_thread.touch_detected.connect(self.on_touch_detected)
                self.calibration_thread.error_occurred.connect(self.on_thread_error)
                self.calibration_thread.start()
        except Exception as e:
            self.status_label.setText(f'터치 감지 스레드 시작 오류: {e}')
        
    def on_thread_error(self, error_msg):
        """스레드에서 발생한 오류 처리"""
        print(f"스레드 오류: {error_msg}")
        self.status_label.setText(f'오류: {error_msg}')
        
        # 스레드 정리 및 재연결 시도
        self.cleanup_threads()
        QTimer.singleShot(3000, self.try_connect_serial)
        
    def on_wait_timeout(self):
        """5초 대기 타임아웃 시 호출"""
        self.timer.stop()
        self.timer_label.setText('캘리브레이션 시작!')
        self.status_label.setText('캘리브레이션을 시작합니다...')
        self.start_button.show()
        self.start_button.click()
        
    def start_calibration(self):
        try:
            # 이미 스레드가 실행 중이면 재사용
            if self.calibration_thread is None or not self.calibration_thread.isRunning():
                if self.offset is not None:
                    self.calibration_thread = CalibrationThread(self.ser, self.offset)
                    self.calibration_thread.touch_detected.connect(self.on_touch_detected)
                    self.calibration_thread.error_occurred.connect(self.on_thread_error)
                    self.calibration_thread.start()
            
            # UI 업데이트
            self.start_button.setEnabled(False)
            self.start_button.setText('캘리브레이션 진행 중...')
            
            # 첫 번째 단계 시작
            self.start_next_step()
            
        except Exception as e:
            self.status_label.setText(f'캘리브레이션 시작 오류: {e}')
    
    def cleanup_threads(self):
        """스레드 정리"""
        if self.calibration_thread:
            self.calibration_thread.stop()
            self.calibration_thread.wait(3000)  # 3초 대기
            self.calibration_thread = None
        
        if self.mouse_control_thread:
            self.mouse_control_thread.stop()
            self.mouse_control_thread.wait(3000)  # 3초 대기
            self.mouse_control_thread = None
    
    def start_next_step(self):
        """다음 캘리브레이션 단계 시작"""
        try:
            self.current_step += 1
            self.progress_bar.setValue(self.current_step)
            
            # 마지막 터치 좌표 초기화
            self.last_touch_coords = None
            self.touch_detected_in_step = False  # 현재 단계에서 터치 감지 여부
            
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
        except Exception as e:
            print(f"다음 단계 시작 오류: {e}")
            self.status_label.setText(f'단계 시작 오류: {e}')
    
    def on_touch_detected(self, coords):
        """터치 감지 시 호출"""
        try:
            r, c = coords
            
            # 캘리브레이션이 시작되지 않은 상태에서 터치가 감지되면 5초 대기 리셋
            if self.current_step == 0:
                self.reset_wait_period()
                return
            
            # 캘리브레이션 진행 중일 때는 5초 타이머 리셋
            if self.current_step >= 1 and self.current_step <= 4:
                self.last_touch_coords = (r, c)  # 마지막 터치 좌표 저장
                self.touch_detected_in_step = True  # 터치 감지됨 표시
                self.reset_calibration_timer()
                return
        except Exception as e:
            print(f"터치 감지 처리 오류: {e}")
    
    def reset_calibration_timer(self):
        """캘리브레이션 중 5초 타이머 리셋"""
        try:
            self.touch_timer.stop()
            self.touch_timer.start(5000)  # 다시 5초 시작
            self.countdown = 5
            self.timer.start(1000)  # 1초마다 업데이트
            self.timer_label.setText('5초')
            self.status_label.setText(f'{self.get_step_message()} (터치 감지됨 - 5초 리셋)')
        except Exception as e:
            print(f"캘리브레이션 타이머 리셋 오류: {e}")
    
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
        try:
            self.timer.stop()
            
            if self.touch_detected_in_step:
                # 터치가 감지된 경우 - 성공 메시지
                self.timer_label.setText('터치 성공!')
                self.status_label.setText('터치가 성공적으로 감지되었습니다. 다음 단계로 진행합니다.')
                
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
                
            else:
                # 터치가 감지되지 않은 경우 - 실패 메시지
                self.timer_label.setText('시간 초과!')
                self.status_label.setText('터치가 감지되지 않았습니다. 다시 시도해주세요.')
                
                # 현재 단계를 다시 시작
                QTimer.singleShot(2000, self.retry_current_step)
        except Exception as e:
            print(f"터치 타임아웃 처리 오류: {e}")
            self.status_label.setText(f'터치 타임아웃 처리 오류: {e}')
    
    def retry_current_step(self):
        """현재 단계를 다시 시도"""
        try:
            self.current_step -= 1  # 단계를 되돌려서 다시 시작
            self.start_next_step()
        except Exception as e:
            print(f"단계 재시도 오류: {e}")
            self.status_label.setText(f'단계 재시도 오류: {e}')
    
    def update_timer(self):
        """타이머 업데이트"""
        try:
            self.countdown -= 1
            self.timer_label.setText(f'{self.countdown}초')
            
            if self.countdown <= 0:
                self.timer.stop()
        except Exception as e:
            print(f"타이머 업데이트 오류: {e}")
    
    def finish_calibration(self):
        """캘리브레이션 완료"""
        try:
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
                    self.calibration_thread.wait(3000)
                    self.calibration_thread = None
                
                # 마우스 제어 스레드 시작
                try:
                    self.mouse_control_thread = MouseControlThread(
                        self.ser, self.offset, self.calibration_data['matrix']
                    )
                    self.mouse_control_thread.error_occurred.connect(self.on_thread_error)
                    self.mouse_control_thread.start()
                    
                    # GUI 최소화 (마우스 제어가 가능하도록)
                    QTimer.singleShot(3000, self.minimize_window)
                    
                except Exception as e:
                    self.status_label.setText(f'마우스 제어 스레드 시작 오류: {e}')
                
            else:
                self.status_label.setText('캘리브레이션에 실패했습니다. 다시 시도해주세요.')
            
            self.start_button.setEnabled(True)
            self.start_button.setText('다시 시작')
            
        except Exception as e:
            print(f"캘리브레이션 완료 처리 오류: {e}")
            self.status_label.setText(f'캘리브레이션 완료 처리 오류: {e}')
    
    def minimize_window(self):
        """창을 최소화하여 마우스 제어가 가능하도록 함"""
        try:
            self.showMinimized()
            self.status_label.setText('창이 최소화되었습니다. 터치패드로 마우스를 제어할 수 있습니다.')
            
            # 게임 런처 실행
            try:
                launcher_process = subprocess.Popen([sys.executable, 'launcher.py'])
                print("게임 런처가 시작되었습니다.")
            except Exception as e:
                print(f"게임 런처 실행 중 오류 발생: {e}")
                
        except Exception as e:
            print(f"창 최소화 오류: {e}")
    
    def closeEvent(self, event):
        """창 닫을 때 정리"""
        try:
            print("애플리케이션 종료 중...")
            
            # 모든 타이머 정지
            if hasattr(self, 'timer'):
                self.timer.stop()
            if hasattr(self, 'touch_timer'):
                self.touch_timer.stop()
            if hasattr(self, 'wait_timer'):
                self.wait_timer.stop()
            
            # 스레드 정리
            self.cleanup_threads()
            
            # 시리얼 포트 닫기
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                    print("시리얼 포트가 닫혔습니다.")
                except Exception as e:
                    print(f"시리얼 포트 닫기 오류: {e}")
            
            event.accept()
            
        except Exception as e:
            print(f"종료 처리 오류: {e}")
            event.accept()  # 오류가 있어도 종료 진행
    
    def reset_wait_period(self):
        """5초 대기 기간 리셋"""
        try:
            self.timer.stop()
            self.wait_timer.stop()
            self.countdown = 5
            self.timer.start(1000)
            self.wait_timer.start(5000)
            self.status_label.setText('터치가 감지되어 5초 대기가 리셋되었습니다...')
            self.timer_label.setText('5초')
        except Exception as e:
            print(f"대기 기간 리셋 오류: {e}")

def main():
    app = QApplication(sys.argv)
    
    # 라즈베리파이에서 전체화면으로 실행
    if 'linux' in sys.platform:
        app.setOverrideCursor(Qt.BlankCursor)  # 커서 숨기기
    
    window = CalibrationGUI()
    window.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"애플리케이션 실행 오류: {e}")

if __name__ == '__main__':
    main()