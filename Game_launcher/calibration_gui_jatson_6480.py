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
import os

# ───────────────── 설정 ─────────────────
# 시리얼 포트/보레이트
SER_PORT   = '/dev/ttyACM0'
BAUDRATE   = 115200

# 패널 및 MUX 설정
PANEL_ROWS    = 80
PANEL_COLS    = 63
FRAME_SIZE    = PANEL_ROWS * PANEL_COLS  # 5040
MUX_CHANNELS  = 8
DEVICES       = 8

# 터치 임계값
TOUCH_THRESHOLD = 30

# 프레임 헤더
SYNC_HDR = b'\xAA\x55'

pyautogui.FAILSAFE = False


# ───────────────── 유틸(프레임 동기화/읽기) ─────────────────
def read_n(ser, n):
    """시리얼에서 정확히 n 바이트 읽기."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            # timeout 등으로 부족하면 잠깐 쉬고 재시도
            time.sleep(0.001)
            continue
        buf.extend(chunk)
    return bytes(buf)

def sync_to_header(ser):
    """
    스트림에서 0xAA 0x55 헤더를 찾고, 그 직후로 포인터를 맞춘다.
    버퍼를 리셋하지 않고, 소비하면서 맞춘다.
    """
    # 간단한 상태기계
    while True:
        b = ser.read(1)
        if not b:
            time.sleep(0.001)
            continue
        if b == b'\xAA':
            b2 = ser.read(1)
            if not b2:
                # 두 번째 바이트가 아직 안 온 경우, 다시 탐색
                continue
            if b2 == b'\x55':
                return  # 동기화 완료
            # 아니면, b2를 새로운 시작 후보로 쓸 수 있도록 루프 계속

def read_one_frame(ser):
    """
    헤더 동기화 → 프레임 데이터(FRAME_SIZE) 읽기 → numpy (rows, cols) 프레임 생성
    """
    sync_to_header(ser)
    raw = read_n(ser, FRAME_SIZE)
    data = np.frombuffer(raw, dtype=np.uint8)

    frame = np.zeros((PANEL_ROWS, PANEL_COLS), dtype=np.uint8)
    ptr = 0
    for row in range(PANEL_ROWS):
        for mux_ch in range(MUX_CHANNELS):   # 0~7
            for dev in range(DEVICES):       # A0~A7
                col = dev * 8 + mux_ch
                if col >= PANEL_COLS:
                    continue
                val = data[ptr]
                ptr += 1
                # 열 뒤집기
                rev_col = PANEL_COLS - 1 - col
                frame[row, rev_col] = val
    return frame


# ───────────────── 위젯 ─────────────────
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


# ───────────────── 스레드(터치 감지) ─────────────────
class CalibrationThread(QThread):
    """터치 감지를 위한 스레드"""
    touch_detected = pyqtSignal(tuple)  # (row, col)

    def __init__(self, ser, offset):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.running = True
        self.TOUCH_THRESHOLD = TOUCH_THRESHOLD
        self.last_touch_time = 0
        self.cool_down_time = 1.0  # 1초

    def run(self):
        while self.running:
            try:
                frame = read_one_frame(self.ser)
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

    def keep_row_col_max_intersection(self, arr):
        row_max = arr.max(axis=1, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        mask = (arr == row_max) & (arr == col_max)
        return arr * mask

    def find_peak(self, arr):
        # 가장 큰 값 우선 탐색
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


# ───────────────── 스레드(마우스 제어) ─────────────────
class MouseControlThread(QThread):
    """마우스 제어 스레드"""
    def __init__(self, ser, offset, calibration_matrix, screen_w, screen_h):
        super().__init__()
        self.ser = ser
        self.offset = offset
        self.calibration_matrix = calibration_matrix
        self.running = True
        self.TOUCH_THRESHOLD = TOUCH_THRESHOLD
        self.SCREEN_W = int(screen_w)
        self.SCREEN_H = int(screen_h)

    def run(self):
        while self.running:
            try:
                frame = read_one_frame(self.ser)
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
        # 행(r) → X(px), 열(c) → Y(px) 매핑 (선형)
        x = int(np.polyval(x_mat, r))
        y = int(np.polyval(y_mat, c))
        x = int(np.clip(x, 0, self.SCREEN_W - 1))
        y = int(np.clip(y, 0, self.SCREEN_H - 1))
        return x, y

    def stop(self):
        self.running = False


# ───────────────── 메인 GUI ─────────────────
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
        self.offset = None
        self.calibration_thread = None
        self.mouse_control_thread = None
        self.current_step = 0
        self.calibration_data = {
            'touch_points': [],
            'screen_points': [],
            'matrix': None
        }

        # 화면 해상도(원하는 해상도로 설정)
        self.SCREEN_W = 1280
        self.SCREEN_H = 800
        self.last_touch_coords = None

        self.init_ui()

    # ── UI 초기화 ──
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

        # 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.touch_timer = QTimer(singleShot=True)
        self.touch_timer.timeout.connect(self.on_touch_timeout)
        self.wait_timer = QTimer(singleShot=True)
        self.wait_timer.timeout.connect(self.on_wait_timeout)

        self.start_wait_period()

    # ── 초기 대기 (UI 먼저, 헤비 작업 나중에) ──
    def start_wait_period(self):
        self.countdown = 10
        self.timer.start(1000)
        self.wait_timer.start(10000)
        self.status_label.setText('10초 동안 터치 없으면 캘리브레이션이 시작됩니다...')
        self.timer_label.setText('10초')

        # 이벤트 루프가 돌기 시작한 다음 시리얼/오프셋 준비(블로킹 방지)
        QTimer.singleShot(0, self._setup_serial_and_threads_safe)

    def _setup_serial_and_threads_safe(self):
        try:
            self.ser = serial.Serial(SER_PORT, BAUDRATE, timeout=1)
            self.offset = self.calibrate_offset()
            self.calibration_thread = CalibrationThread(self.ser, self.offset)
            self.calibration_thread.touch_detected.connect(self.on_touch_detected)
            self.calibration_thread.start()
        except Exception as e:
            self.status_label.setText(f'연결 오류: {e}')

    # ── 카운트다운 완료 → 자동 시작 ──
    def on_wait_timeout(self):
        self.timer.stop()
        self.timer_label.setText('시작!')
        self.status_label.setText('캘리브레이션을 시작합니다...')
        self.start_button.show()
        self.start_button.click()

    # ── 캘리브레이션 시작 ──
    def start_calibration(self):
        self.waiting_for_start = False
        self.start_button.setEnabled(False)
        self.start_button.setText('진행 중...')
        self.current_step = 1
        self.progress_bar.setValue(0)
        self.start_current_step()

    # ── 오프셋 보정 ──
    def calibrate_offset(self, num_frames=10):
        frames = []
        # 첫 프레임 동기화
        # 이후 각 프레임마다 헤더 동기화 + 데이터 읽기
        while len(frames) < num_frames:
            try:
                frame = read_one_frame(self.ser)
                if frame is not None:
                    frames.append(frame.astype(np.float32))
            except Exception as e:
                print(f"오프셋 보정 중 오류: {e}")
            time.sleep(0.005)
        return np.mean(frames, axis=0).astype(np.float32)

    # ── 현재 단계 시작(증가 없이) ──
    def start_current_step(self):
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

            self.last_touch_coords = None
            self.countdown = 5
            self.timer.start(1000)
            self.touch_timer.start(5000)
        else:
            self.finish_calibration()

    # ── 다음 단계로 ──
    def start_next_step(self):
        self.current_step += 1
        self.progress_bar.setValue(min(self.current_step, 4))
        self.start_current_step()

    # ── 터치 감지 콜백 ──
    def on_touch_detected(self, coords):
        # 초기 10초 대기 중이면 대기 리셋
        if self.waiting_for_start and self.current_step == 0:
            self.reset_wait_period()
            return

        # 단계 중 터치 감지 → 5초 타이머 리셋
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

    # ── 터치 타임아웃(5초) ──
    def on_touch_timeout(self):
        self.timer.stop()

        if self.last_touch_coords:
            # 터치가 있었음 → 좌표 기록하고 다음 단계
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
            # 터치가 없었음 → 같은 단계 재시작
            self.timer_label.setText('시간 초과')
            self.status_label.setText('터치가 없어 같은 단계를 다시 시작합니다.')
            QTimer.singleShot(800, self.start_current_step)

    # ── 카운트다운 라벨 업데이트(초/1초 간격) ──
    def update_timer(self):
        self.countdown -= 1
        self.timer_label.setText(f'{self.countdown}초')
        if self.countdown <= 0:
            self.timer.stop()

    # ── 캘리브레이션 완료 ──
    def finish_calibration(self):
        if len(self.calibration_data['touch_points']) >= 4:
            tp = np.array(self.calibration_data['touch_points'])
            sp = np.array(self.calibration_data['screen_points'])
            # r→x, c→y 선형 근사
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
                self.ser, self.offset, self.calibration_data['matrix'],
                self.SCREEN_W, self.SCREEN_H
            )
            self.mouse_control_thread.start()

            # 캘리브레이션 완료 신호 파일 생성
            try:
                with open('.calibration_complete', 'w') as f:
                    pass
            except Exception as e:
                print(f"신호 파일 생성 실패: {e}")

            QTimer.singleShot(3000, self.minimize_window)
        else:
            self.status_label.setText('캘리브레이션 실패. 다시 시도해주세요.')

        self.start_button.setEnabled(True)
        self.start_button.setText('다시 시작')

    def minimize_window(self):
        self.showMinimized()
        self.status_label.setText('창이 최소화되었습니다. 터치패드 제어를 사용하세요.')

    # ── 초기 대기 리셋 ──
    def reset_wait_period(self):
        self.timer.stop()
        self.wait_timer.stop()
        self.countdown = 10
        self.timer.start(1000)
        self.wait_timer.start(10000)
        self.status_label.setText('터치 감지되어 10초 대기가 리셋되었습니다...')
        self.timer_label.setText('10초')

    # ── 종료 처리 ──
    def closeEvent(self, event):
        if self.calibration_thread:
            self.calibration_thread.stop()
            self.calibration_thread.wait()
        if self.mouse_control_thread:
            self.mouse_control_thread.stop()
            self.mouse_control_thread.wait()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        event.accept()


# ───────────────── 엔트리 ─────────────────
def main():
    app = QApplication(sys.argv)
    if 'linux' in sys.platform:
        app.setOverrideCursor(Qt.BlankCursor)
    window = CalibrationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
