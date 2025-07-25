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

# 패널 및 MUX 설정
PANEL_ROWS = 80
PANEL_COLS = 63
FRAME_SIZE = PANEL_ROWS * PANEL_COLS
MUX_CHANNELS = 8
DEVICES = 8

pyautogui.FAILSAFE = False

class CalibrationPoint(QFrame):
    """캘리브레이션 포인트 표시 위젯"""
    def __init__(self, position, parent=None):
        super().__init__(parent)
        self.position = position  # 'top-left', 'top-right', ...
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

        # 원의 좌표
        if self.position == 'top-left':
            x, y = 30, 30
        elif self.position == 'top-right':
            x, y = self.width() - 30, 30
        elif self.position == 'bottom-left':
            x, y = 30, self.height() - 30
        else:  # bottom-right
            x, y = self.width() - 30, self.height() - 30

        # 색상 결정
        if self.is_completed:
            color = QColor(0, 255, 0)
        elif self.is_active:
            color = QColor(255, 165, 0)
        else:
            color = QColor(128, 128, 128)

        painter.setPen(QPen(color, 4))
        painter.setBrush(color)
        painter.drawEllipse(x - 10, y - 10, 20, 20)

        # 텍스트
        painter.setPen(QPen(Qt.black, 2))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        text = self.position.replace('-', ' ').title()
        painter.drawText(self.rect(), Qt.AlignCenter, text)


class CalibrationThread(QThread):
    """터치 감지를 위한 스레드"""
    touch_detected = pyqtSignal(tuple)  # (row, col)

    def __init__(self, ser, offset):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.running = True
        self.TOUCH_THRESHOLD = 50
        self.last_touch_time = 0
        self.cool_down_time = 1.0  # 1초

    def run(self):
        while self.running:
            try:
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)

                    peak = self.find_peak(filtered)
                    if peak:
                        r, c, v = peak
                        if v >= self.TOUCH_THRESHOLD:
                            now = time.time()
                            if now - self.last_touch_time >= self.cool_down_time:
                                self.touch_detected.emit((r, c))
                                self.last_touch_time = now
            except Exception as e:
                print(f"터치 감지 오류: {e}")
            time.sleep(0.01)

    def read_frame(self):
        raw = self.ser.read(FRAME_SIZE)
        if len(raw) != FRAME_SIZE:
            return None
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
                    # 열 뒤집기
                    rev_col = PANEL_COLS - 1 - col
                    frame[row, rev_col] = val
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
        for v, r, c in candidates:
            if v < self.TOUCH_THRESHOLD:
                continue
            if np.max(arr[r, :]) > v or np.max(arr[:, c]) > v:
                continue
            return r, c, v
        return None

    def stop(self):
        self.running = False


class MouseControlThread(QThread):
    """마우스 제어 스레드"""
    def __init__(self, ser, offset, calibration_matrix):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.calibration_matrix = calibration_matrix
        self.running = True
        self.TOUCH_THRESHOLD = 100
        self.SCREEN_W = 1200
        self.SCREEN_H = 800

    def run(self):
        while self.running:
            try:
                frame = self.read_frame()
                if frame is not None:
                    corr = frame.astype(np.float32) - self.offset
                    corr = np.clip(corr, 0, 255).astype(np.uint8)
                    filtered = self.keep_row_col_max_intersection(corr)

                    peak = self.find_peak(filtered)
                    if peak:
                        r, c, v = peak
                        if v >= self.TOUCH_THRESHOLD:
                            pts = self.map_touch_to_screen(r, c)
                            if pts:
                                x_px, y_px = pts
                                pyautogui.moveTo(x_px, y_px)
            except Exception as e:
                print(f"마우스 제어 오류: {e}")
            time.sleep(0.01)

    def read_frame(self):
        raw = self.ser.read(FRAME_SIZE)
        if len(raw) != FRAME_SIZE:
            return None
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
        for v, r, c in candidates:
            if v < self.TOUCH_THRESHOLD:
                continue
            if np.max(arr[r, :]) > v or np.max(arr[:, c]) > v:
                continue
            return r, c, v
        return None

    def map_touch_to_screen(self, r, c):
        if self.calibration_matrix is None:
            return None
        x_mat, y_mat = self.calibration_matrix
        x = int(np.polyval(x_mat, r))
        y = int(np.polyval(y_mat, c))
        x = np.clip(x, 0, self.SCREEN_W - 1)
        y = np.clip(y, 0, self.SCREEN_H - 1)
        return x, y

    def stop(self):
        self.running = False


class CalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # 패널 크기
        self.NUM_ROWS = PANEL_ROWS
        self.NUM_COLS = PANEL_COLS
        self.FRAME_SIZE = FRAME_SIZE
        # 초기 대기 플래그
        self.waiting_for_start = True

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
        self.last_touch_coords = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('터치패드 캘리브레이션')
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(20, 20, 20, 20)

        title = QLabel('터치패드 캘리브레이션')
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        self.status_label = QLabel('10초 동안 터치 없으면 캘리브레이션이 시작됩니다...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel { font-size:14px; color:#333; padding:10px;
                     background:#e8f4fd; border-radius:5px; }
        """)
        main.addWidget(self.status_label)

        self.timer_label = QLabel('10초')
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("""
            QLabel { font-size:24px; font-weight:bold; color:#f60; padding:10px; }
        """)
        main.addWidget(self.timer_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border:2px solid #ccc; border-radius:5px; text-align:center; }
            QProgressBar::chunk { background-color:#4CAF50; border-radius:3px; }
        """)
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
        self.start_button.setStyleSheet("""
            QPushButton { background:#4CAF50; color:white; padding:15px; font-size:16px;
                          border:none; border-radius:5px; }
            QPushButton:hover { background:#45a049; }
            QPushButton:pressed { background:#3d8b40; }
        """)
        self.start_button.clicked.connect(self.start_calibration)
        self.start_button.hide()
        main.addWidget(self.start_button)

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.touch_timer = QTimer(singleShot=True)
        self.touch_timer.timeout.connect(self.on_touch_timeout)
        self.wait_timer = QTimer(singleShot=True)
        self.wait_timer.timeout.connect(self.on_wait_timeout)

        self.start_wait_period()

    def start_wait_period(self):
        self.countdown = 10
        self.timer.start(1000)
        self.wait_timer.start(10000)
        self.status_label.setText('10초 동안 터치 없으면 캘리브레이션이 시작됩니다...')
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            self.offset = self.calibrate_offset()
            self.calibration_thread = CalibrationThread(self.ser, self.offset)
            self.calibration_thread.touch_detected.connect(self.on_touch_detected)
            self.calibration_thread.start()
        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}')

    def on_wait_timeout(self):
        self.timer.stop()
        self.timer_label.setText('시작!')
        self.status_label.setText('캘리브레이션을 시작합니다...')
        self.start_button.show()
        self.start_button.click()

    def start_calibration(self):
        # 초기 대기 단계 해제
        self.waiting_for_start = False
        self.start_button.setEnabled(False)
        self.start_button.setText('진행 중...')
        self.start_next_step()

    def calibrate_offset(self):
        frames = []
        while len(frames) < 10:
            raw = self.ser.read(FRAME_SIZE)
            if len(raw) == FRAME_SIZE:
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
                frames.append(frame.astype(np.float32))
            time.sleep(0.01)
        return np.mean(frames, axis=0).astype(np.float32)

    def start_next_step(self):
        self.current_step += 1
        self.progress_bar.setValue(self.current_step)
        self.last_touch_coords = None
        for p in (self.top_left, self.top_right, self.bottom_left, self.bottom_right):
            p.is_active = False
            p.update()

        messages = {
            1: '좌측 상단을 터치해주세요',
            2: '우측 상단을 터치해주세요',
            3: '좌측 하단을 터치해주세요',
            4: '우측 하단을 터치해주세요'
        }
        widgets = {
            1: self.top_left,
            2: self.top_right,
            3: self.bottom_left,
            4: self.bottom_right
        }

        if self.current_step in messages:
            self.status_label.setText(messages[self.current_step])
            widgets[self.current_step].is_active = True
            widgets[self.current_step].update()
            self.countdown = 5
            self.timer.start(1000)
            self.touch_timer.start(5000)
        else:
            self.finish_calibration()

    def on_touch_detected(self, coords):
        # 초기 10초 대기 중에만 리셋
        if self.waiting_for_start and self.current_step == 0:
            self.reset_wait_period()
            return
        # 캘리브레이션 단계 중 터치 리셋
        if 1 <= self.current_step <= 4:
            self.last_touch_coords = coords
            self.touch_timer.stop()
            self.touch_timer.start(5000)
            self.countdown = 5
            self.timer.start(1000)
            msgs = {
                1: '좌측 상단을 터치해주세요',
                2: '우측 상단을 터치해주세요',
                3: '좌측 하단을 터치해주세요',
                4: '우측 하단을 터치해주세요'
            }
            self.status_label.setText(f"{msgs[self.current_step]} (터치 감지됨 - 5초 리셋)")

    def on_touch_timeout(self):
        self.timer.stop()
        self.timer_label.setText('시간 초과')
        self.status_label.setText('5초 동안 터치 없어 다음 단계로 진행합니다.')

        if self.last_touch_coords:
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

        QTimer.singleShot(2000, self.start_next_step)

    def update_timer(self):
        self.countdown -= 1
        self.timer_label.setText(f'{self.countdown}초')
        if self.countdown <= 0:
            self.timer.stop()

    def finish_calibration(self):
        if len(self.calibration_data['touch_points']) >= 4:
            tp = np.array(self.calibration_data['touch_points'])
            sp = np.array(self.calibration_data['screen_points'])
            x_mat = np.polyfit(tp[:, 0], sp[:, 0], 1)
            y_mat = np.polyfit(tp[:, 1], sp[:, 1], 1)
            self.calibration_data['matrix'] = (x_mat, y_mat)

            self.status_label.setText('캘리브레이션 완료! 마우스 제어 시작합니다.')
            self.timer_label.setText('완료')
            self.progress_bar.setValue(4)

            if self.calibration_thread:
                self.calibration_thread.stop()
                self.calibration_thread.wait()

            self.mouse_control_thread = MouseControlThread(
                self.ser, self.offset, self.calibration_data['matrix']
            )
            self.mouse_control_thread.start()
            QTimer.singleShot(3000, self.minimize_window)
        else:
            self.status_label.setText('캘리브레이션 실패. 다시 시도해주세요.')

        self.start_button.setEnabled(True)
        self.start_button.setText('다시 시작')

    def minimize_window(self):
        self.showMinimized()
        self.status_label.setText('창이 최소화되었습니다. 터치패드 제어를 사용하세요.')

    def reset_wait_period(self):
        self.timer.stop()
        self.wait_timer.stop()
        self.countdown = 10
        self.timer.start(1000)
        self.wait_timer.start(10000)
        self.status_label.setText('터치 감지되어 10초 대기가 리셋되었습니다...')
        self.timer_label.setText('10초')

    def closeEvent(self, event):
        if self.calibration_thread:
            self.calibration_thread.stop()
            self.calibration_thread.wait()
        if self.mouse_control_thread:
            self.mouse_control_thread.stop()
            self.mouse_control_thread.wait()
        if self.ser:
            self.ser.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    if 'linux' in sys.platform:
        app.setOverrideCursor(Qt.BlankCursor)
    window = CalibrationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
