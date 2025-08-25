import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
import serial
import numpy as np
import pyautogui

# ───────────────── 설정 ─────────────────
SER_PORT    = '/dev/ttyUSB0'
BAUDRATE    = 1000000
PANEL_ROWS    = 80
PANEL_COLS    = 63
FRAME_SIZE    = PANEL_ROWS * PANEL_COLS
MUX_CHANNELS  = 8
DEVICES       = 8
TOUCH_THRESHOLD = 20
SYNC_HDR = b'\xAA\x55'

pyautogui.FAILSAFE = False

# ───────────────── 유틸리티 함수들 ─────────────────
def read_n(ser, n):
    """시리얼에서 정확히 n 바이트 읽기."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            time.sleep(0.001)
            continue
        buf.extend(chunk)
    return bytes(buf)

def sync_to_header(ser):
    """스트림에서 0xAA 0x55 헤더를 찾고, 그 직후로 포인터를 맞춘다."""
    while True:
        b = ser.read(1)
        if not b:
            time.sleep(0.001)
            continue
        if b == b'\xAA':
            b2 = ser.read(1)
            if not b2:
                continue
            if b2 == b'\x55':
                return

def read_one_frame(ser):
    """헤더 동기화 → 프레임 데이터 읽기 → numpy 프레임 생성"""
    sync_to_header(ser)
    raw = read_n(ser, FRAME_SIZE)
    data = np.frombuffer(raw, dtype=np.uint8)

    frame = np.zeros((PANEL_ROWS, PANEL_COLS), dtype=np.uint8)
    ptr = 0
    for row in range(PANEL_ROWS):
        for mux_ch in range(MUX_CHANNELS):
            for dev in range(DEVICES):
                col = dev * 8 + mux_ch
                if col >= PANEL_COLS:
                    continue
                val = data[ptr]
                ptr += 1
                rev_col = PANEL_COLS - 1 - col
                frame[row, rev_col] = val
    return frame

# ───────────────── 위젯 ─────────────────
class CalibrationPoint(QFrame):
    """캘리브레이션 포인트 표시 위젯"""
    def __init__(self, position, parent=None):
        super().__init__(parent)
        self.position = position
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

        if self.position == 'top-left':
            x, y = 30, 30
        elif self.position == 'top-right':
            x, y = self.width() - 30, 30
        elif self.position == 'bottom-left':
            x, y = 30, self.height() - 30
        else:  # bottom-right
            x, y = self.width() - 30, self.height() - 30

        if self.is_completed:
            color = QColor(0, 255, 0)
        elif self.is_active:
            color = QColor(255, 165, 0)
        else:
            color = QColor(128, 128, 128)

        painter.setPen(QPen(color, 4))
        painter.setBrush(color)
        painter.drawEllipse(x - 10, y - 10, 20, 20)

        painter.setPen(QPen(Qt.black, 2))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        text = self.position.replace('-', ' ').title()
        painter.drawText(self.rect(), Qt.AlignCenter, text)

# ───────────────── 스레드 ─────────────────
class SerialReaderThread(QThread):
    """시리얼 포트에서 프레임 데이터를 지속적으로 읽는 스레드"""
    frame_received = pyqtSignal(object)

    def __init__(self, ser, parent=None):
        super().__init__(parent)
        self.ser = ser
        self.running = True

    def run(self):
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    frame = read_one_frame(self.ser)
                    if frame is not None:
                        self.frame_received.emit(frame)
                else:
                    time.sleep(0.5)
            except serial.SerialException as e:
                print(f"시리얼 오류 발생: {e}. 재연결을 시도합니다.")
                time.sleep(1)
            except Exception as e:
                print(f"데이터 읽기 스레드 오류: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        print("시리얼 리더 스레드 중지 요청됨.")

# ───────────────── 메인 GUI ─────────────────
class DragCalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.NUM_ROWS = PANEL_ROWS
        self.NUM_COLS = PANEL_COLS
        self.FRAME_SIZE = FRAME_SIZE

        # 애플리케이션 상태 관리
        self.app_state = 'initializing'  # 'initializing', 'waiting_start', 'calibrating', 'controlling'

        self.ser = None
        self.offset = None
        self.reader_thread = None
        
        # 드래그 감지를 위한 변수들
        self.last_touch_coords = None
        self.last_touch_time = 0
        self.touch_cool_down = 0.1  # 터치 간 최소 간격 (초) - 드래그를 위해 줄임
        self.drag_threshold = 15    # 드래그 시작 임계값 (센서 단위)
        self.is_dragging = False    # 현재 드래그 중인지 여부

        self.current_step = 0
        self.calibration_data = {
            'touch_points': [],
            'screen_points': [],
            'matrix': None
        }

        self.SCREEN_W = 1280
        self.SCREEN_H = 800

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('드래그 지원 터치패드 캘리브레이션')
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(20, 20, 20, 20)

        title = QLabel('드래그 지원 터치패드 캘리브레이션')
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        self.status_label = QLabel('시리얼 포트 연결 중...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { font-size:14px; color:#333; padding:10px; background:#e8f4fd; border-radius:5px; }")
        main.addWidget(self.status_label)

        # 드래그 상태 표시 라벨 추가
        self.drag_status_label = QLabel('드래그 상태: 대기 중')
        self.drag_status_label.setAlignment(Qt.AlignCenter)
        self.drag_status_label.setStyleSheet("QLabel { font-size:12px; color:#666; padding:5px; }")
        main.addWidget(self.drag_status_label)

        self.timer_label = QLabel('')
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("QLabel { font-size:24px; font-weight:bold; color:#f60; padding:10px; }")
        main.addWidget(self.timer_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border:2px solid #ccc; border-radius:5px; text-align:center; } QProgressBar::chunk { background-color:#4CAF50; border-radius:3px; }")
        main.addWidget(self.progress_bar)

        pts_layout = QHBoxLayout()
        self.top_left     = CalibrationPoint('top-left')
        self.top_right    = CalibrationPoint('top-right')
        self.bottom_left  = CalibrationPoint('bottom-left')
        self.bottom_right = CalibrationPoint('bottom-right')
        for w in (self.top_left, self.top_right, self.bottom_left, self.bottom_right):
            pts_layout.addWidget(w)
        main.addLayout(pts_layout)

        self.start_button = QPushButton('캘리브레이션 시작')
        self.start_button.setStyleSheet("QPushButton { background:#4CAF50; color:white; padding:15px; font-size:16px; border:none; border-radius:5px; } QPushButton:hover { background:#45a049; } QPushButton:pressed { background:#3d8b40; }")
        self.start_button.clicked.connect(self.start_calibration)
        self.start_button.hide()
        main.addWidget(self.start_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.touch_timer = QTimer(singleShot=True)
        self.touch_timer.timeout.connect(self.on_touch_timeout)
        self.wait_timer = QTimer(singleShot=True)
        self.wait_timer.timeout.connect(self.on_wait_timeout)

        QTimer.singleShot(100, self.setup_serial_and_reader)

    def setup_serial_and_reader(self):
        try:
            self.ser = serial.Serial(SER_PORT, BAUDRATE, timeout=1)
            self.status_label.setText('오프셋 측정 중... 패널을 만지지 마세요.')
            QApplication.processEvents()

            self.offset = self.calibrate_offset()
            if self.offset is None:
                raise ConnectionError("오프셋 측정 실패")

            self.reader_thread = SerialReaderThread(self.ser)
            self.reader_thread.frame_received.connect(self.handle_frame)
            self.reader_thread.start()

            self.start_wait_period()

        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}. 5초 후 재시도합니다.')
            QTimer.singleShot(5000, self.setup_serial_and_reader)

    def start_wait_period(self):
        self.app_state = 'waiting_start'
        self.countdown = 10
        self.timer_label.setText(f'{self.countdown}초')
        self.status_label.setText('10초 동안 터치 없으면 캘리브레이션이 시작됩니다...')
        self.timer.start(1000)
        self.wait_timer.start(10000)

    def on_wait_timeout(self):
        if self.app_state == 'waiting_start':
            self.timer.stop()
            self.timer_label.setText('시작!')
            self.status_label.setText('캘리브레이션을 시작합니다...')
            self.start_button.show()
            self.start_button.click()

    def start_calibration(self):
        self.wait_timer.stop()
        self.app_state = 'calibrating'
        self.start_button.setEnabled(False)
        self.start_button.setText('진행 중...')
        self.current_step = 1
        self.progress_bar.setValue(0)
        self.start_current_step()

    def calibrate_offset(self, num_frames=10):
        frames = []
        self.ser.reset_input_buffer()
        while len(frames) < num_frames:
            try:
                frame = read_one_frame(self.ser)
                if frame is not None:
                    frames.append(frame.astype(np.float32))
            except Exception as e:
                print(f"오프셋 보정 중 오류: {e}")
                return None
        return np.mean(frames, axis=0).astype(np.float32)

    def handle_frame(self, frame):
        """원본 방식의 프레임 처리 + 드래그 감지 추가"""
        if self.offset is None: return

        # 1. 프레임 전처리 (오프셋 보정 및 필터링)
        corr = frame.astype(np.float32) - self.offset
        corr = np.clip(corr, 0, 255).astype(np.uint8)
        filtered = self.keep_row_col_max_intersection(corr)
        peak = self.find_peak(filtered)

        if not peak: 
            # 터치가 없으면 드래그 상태 리셋
            if self.is_dragging:
                self.is_dragging = False
                self.drag_status_label.setText('드래그 상태: 종료')
            return

        r, c, v = peak
        if v < TOUCH_THRESHOLD: return

        # 2. 현재 상태에 따라 다른 작업 수행
        now = time.time()
        if (now - self.last_touch_time) < self.touch_cool_down:
            return # 쿨다운 시간 동안 터치 무시
        self.last_touch_time = now

        if self.app_state == 'waiting_start':
            self.on_touch_detected((r, c))
        elif self.app_state == 'calibrating':
            self.on_touch_detected((r, c))
        elif self.app_state == 'controlling':
            self.control_mouse_with_drag((r, c))

    def control_mouse_with_drag(self, coords):
        """드래그 지원 마우스 제어"""
        r, c = coords
        
        # 드래그 감지 로직
        if self.last_touch_coords:
            last_r, last_c = self.last_touch_coords
            distance = np.sqrt((r - last_r)**2 + (c - last_c)**2)
            
            if distance >= self.drag_threshold:
                if not self.is_dragging:
                    self.is_dragging = True
                    self.drag_status_label.setText('드래그 상태: 시작')
                    print(f"DEBUG: 드래그 시작 - 거리: {distance:.1f}")
            else:
                if self.is_dragging:
                    self.is_dragging = False
                    self.drag_status_label.setText('드래그 상태: 종료')
                    print(f"DEBUG: 드래그 종료 - 거리: {distance:.1f}")
        
        # 마우스 제어
        if self.calibration_data['matrix'] is None:
            return

        x_mat, y_mat = self.calibration_data['matrix']
        x = int(np.polyval(x_mat, r))
        y = int(np.polyval(y_mat, c))
        x = int(np.clip(x, 0, self.SCREEN_W - 1))
        y = int(np.clip(y, 0, self.SCREEN_H - 1))
        
        print(f"DEBUG: 마우스 제어 - 터치: ({r}, {c}) → 스크린: ({x}, {y}) {'[드래그]' if self.is_dragging else ''}")
        pyautogui.moveTo(x, y)
        
        # 현재 좌표 저장
        self.last_touch_coords = (r, c)

    def on_touch_detected(self, coords):
        if self.app_state == 'waiting_start':
            self.reset_wait_period()
            return

        if self.app_state == 'calibrating' and 1 <= self.current_step <= 4:
            self.last_touch_coords = coords
            self.touch_timer.stop()
            self.touch_timer.start(5000)
            self.countdown = 5
            self.timer.start(1000)
            msgs = {
                1: '좌측 상단', 2: '우측 상단',
                3: '좌측 하단', 4: '우측 하단'
            }
            self.status_label.setText(f"{msgs[self.current_step]} 터치 감지됨 - 5초 후 기록됩니다.")

    def on_touch_timeout(self):
        self.timer.stop()
        if self.last_touch_coords:
            self.timer_label.setText('기록 완료')
            self.status_label.setText('터치가 기록되었습니다. 다음 단계로 진행합니다.')

            r, c = self.last_touch_coords
            corners = {
                1: (0, 0),
                2: (self.SCREEN_W - 1, 0),
                3: (0, self.SCREEN_H - 1),
                4: (self.SCREEN_W - 1, self.SCREEN_H - 1)
            }
            self.calibration_data['touch_points'].append((r, c))
            self.calibration_data['screen_points'].append(corners[self.current_step])
            
            pts = [self.top_left, self.top_right, self.bottom_left, self.bottom_right]
            pts[self.current_step - 1].is_completed = True
            pts[self.current_step - 1].update()
            self.last_touch_coords = None

            QTimer.singleShot(500, self.start_next_step)
        else:
            self.timer_label.setText('시간 초과')
            self.status_label.setText('터치가 없어 같은 단계를 다시 시작합니다.')
            QTimer.singleShot(800, self.start_current_step)

    def start_current_step(self):
        for p in (self.top_left, self.top_right, self.bottom_left, self.bottom_right):
            p.is_active = False
            p.update()

        messages = {
            1: '좌측 상단을 5초간 터치해주세요', 2: '우측 상단을 5초간 터치해주세요',
            3: '좌측 하단을 5초간 터치해주세요', 4: '우측 하단을 5초간 터치해주세요'
        }
        widgets = {
            1: self.top_left, 2: self.top_right,
            3: self.bottom_left, 4: self.bottom_right
        }

        if self.current_step in messages:
            self.status_label.setText(messages[self.current_step])
            widgets[self.current_step].is_active = True
            widgets[self.current_step].update()

            self.last_touch_coords = None
            self.countdown = 5
            self.timer_label.setText(f'{self.countdown}초')
            self.timer.start(1000)
            self.touch_timer.start(5000)
        else:
            self.finish_calibration()

    def start_next_step(self):
        self.current_step += 1
        self.progress_bar.setValue(min(self.current_step-1, 4))
        self.start_current_step()

    def finish_calibration(self):
        if len(self.calibration_data['touch_points']) >= 4:
            tp = np.array(self.calibration_data['touch_points'])
            sp = np.array(self.calibration_data['screen_points'])
            x_mat = np.polyfit(tp[:, 0], sp[:, 0], 1)
            y_mat = np.polyfit(tp[:, 1], sp[:, 1], 1)
            self.calibration_data['matrix'] = (x_mat, y_mat)

            self.status_label.setText('캘리브레이션 완료! 드래그 지원 마우스 제어를 시작합니다.')
            self.timer_label.setText('완료')
            self.progress_bar.setValue(4)
            
            # 캘리브레이션 완료 신호 파일 생성
            try:
                with open('.calibration_complete', 'w') as f:
                    f.write('calibration_complete')
                print("DEBUG: 캘리브레이션 완료 신호 파일 생성됨")
            except Exception as e:
                print(f"DEBUG: 신호 파일 생성 실패: {e}")
            
            self.app_state = 'controlling'
            
            QTimer.singleShot(3000, self.minimize_window)
        else:
            self.status_label.setText('캘리브레이션 실패. 다시 시도해주세요.')

        self.start_button.setEnabled(True)
        self.start_button.setText('다시 시작')

    def update_timer(self):
        self.countdown -= 1
        if self.countdown >= 0:
            self.timer_label.setText(f'{self.countdown}초')
        if self.countdown < 0:
            self.timer.stop()

    def minimize_window(self):
        self.showMinimized()
        self.status_label.setText('창이 최소화되었습니다. 드래그 지원 터치패드 제어를 사용하세요.')

    def reset_wait_period(self):
        self.timer.stop()
        self.wait_timer.stop()
        self.start_wait_period()
        self.status_label.setText('터치 감지되어 10초 대기가 리셋되었습니다...')

    # ── 유틸리티 및 헬퍼 함수들 ──
    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask

    def find_peak(self, arr):
        max_val = arr.max()
        if max_val < TOUCH_THRESHOLD:
            return None
        
        loc = np.where(arr == max_val)
        r, c = loc[0][0], loc[1][0]
        return r, c, max_val

    def closeEvent(self, event):
        if self.reader_thread:
            self.reader_thread.stop()
            self.reader_thread.wait()
        if self.ser and self.ser.is_open:
            self.ser.close()
        event.accept()

# ───────────────── 엔트리 ─────────────────
def main():
    app = QApplication(sys.argv)
    if 'linux' in sys.platform:
        app.setOverrideCursor(Qt.BlankCursor)
    window = DragCalibrationGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
