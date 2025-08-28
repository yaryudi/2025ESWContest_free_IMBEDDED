import sys
import time
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
import serial
import numpy as np
import pyautogui

# 새로운 터치 매니저 import
from touch_manager import TouchManager, TouchState, TouchEvent

# ───────────────── 설정 ─────────────────
SER_PORT    = 'COM15'#'/dev/ttyUSB0'
BAUDRATE    = 2000000
PANEL_ROWS  = 80
PANEL_COLS  = 64
FRAME_SIZE  = PANEL_ROWS * PANEL_COLS
MUX_CHANNELS  = 8
DEVICES       = 8

# 기본 임계값(최저 보정치). 프레임별 동적 임계값이 이 값보다 낮아지지 않게 하는 하한선.
TOUCH_THRESHOLD_MIN = 8

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

# ───────────────── 평균값 보정(러닝 베이스라인) 유틸 ─────────────────
class RunningBaseline:
    """
    - 초기: N프레임 평균으로 베이스라인 시작
    - 이후: 터치가 아닌 픽셀만 선택(마스크)하여 EMA(알파)로 업데이트
    - subtract_and_update(frame) -> (corrected, used_mask, dyn_threshold)
    """
    def __init__(self, shape, init_mean=None, alpha=0.02, clip_min=0.0, clip_max=255.0):
        self.shape = shape
        self.alpha = float(alpha)
        self.clip_min = clip_min
        self.clip_max = clip_max

        if init_mean is None:
            self.baseline = np.zeros(shape, dtype=np.float32)
        else:
            self.baseline = init_mean.astype(np.float32)

        # 시간적 스무딩용 버퍼
        self.prev_corrected = np.zeros(shape, dtype=np.float32)

        # 노이즈 추정을 위한 단순 EMA(절대값)
        self.noise_ema = 1.0  # 전역적(스칼라) 노이즈 지표

    @staticmethod
    def robust_stats(arr):
        """
        중앙값과 MAD(중앙절대편차)의 간단한 근사.
        MAD를 ~1.4826 배하면 정규분포 std 근사.
        """
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        robust_std = 1.4826 * mad
        return med, robust_std

    def _provisional_touch_mask(self, corrected, base_thresh):
        """
        베이스라인 업데이트에서 제외할 터치 후보 마스크.
        corrected가 base_thresh보다 큰 곳을 터치 후보로 간주.
        """
        return corrected > base_thresh

    def subtract_and_update(self, frame_u8, app_state, last_event_state):
        """
        1) baseline을 뺀 corrected 계산
        2) 터치 후보 마스크 산출(보수적 임계)
        3) 터치가 없거나 약할 때만 baseline EMA 업데이트
        4) corrected에 시간적 스무딩 적용
        5) 프레임별 동적 임계값(dyn_thr) 산출
        """
        f = frame_u8.astype(np.float32)
        corrected = f - self.baseline
        corrected = np.clip(corrected, self.clip_min, self.clip_max)

        # 터치 보호용 임시 임계값(보수적으로 높게): 중앙값+3*MAD
        med, robust_std = self.robust_stats(corrected)
        provisional_thr = med + 3.0 * max(robust_std, 1.0)

        touch_mask = self._provisional_touch_mask(corrected, provisional_thr)

        # 앱이 제어모드이고 실제 드래그/터치가 진행 중일 때는 업데이트 억제 강하게
        is_touching_now = False
        if last_event_state in (TouchState.TOUCHING, TouchState.DRAGGING):
            is_touching_now = True

        # 베이스라인 업데이트: 터치 후보가 아닌 픽셀만 EMA
        # 캘리브레이션/대기 상태에서도 터치 없으면 업데이트
        if not is_touching_now:
            not_touch_mask = ~touch_mask
            # EMA: baseline = (1-alpha)*baseline + alpha*frame
            self.baseline[not_touch_mask] = (
                (1.0 - self.alpha) * self.baseline[not_touch_mask] +
                self.alpha * f[not_touch_mask]
            )

        # 시간적 스무딩(간단한 EMA)
        beta = 0.5  # 0~1 (높을수록 현재 프레임 반영 큼)
        smoothed = beta * corrected + (1.0 - beta) * self.prev_corrected
        self.prev_corrected = smoothed

        # 전역 노이즈 지표 업데이트(EMA of abs)
        frame_noise = float(np.median(np.abs(smoothed - np.median(smoothed))))
        self.noise_ema = 0.9 * self.noise_ema + 0.1 * max(frame_noise, 1.0)

        # 프레임별 동적 임계값: 중앙값 + k * MAD  (최소 하한 유지)
        med2, robust_std2 = self.robust_stats(smoothed)
        dyn_thr = int(max(TOUCH_THRESHOLD_MIN, med2 + 3.0 * max(robust_std2, 1.0)))

        # 최종 출력은 uint8 범위로 클립
        smoothed_u8 = np.clip(smoothed, 0, 255).astype(np.uint8)
        return smoothed_u8, touch_mask, dyn_thr

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
class ImprovedCalibrationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.NUM_ROWS = PANEL_ROWS
        self.NUM_COLS = PANEL_COLS
        self.FRAME_SIZE = FRAME_SIZE

        # 애플리케이션 상태 관리
        self.app_state = 'initializing'  # 'initializing', 'waiting_start', 'calibrating', 'controlling'

        self.ser = None
        self.reader_thread = None

        # 러닝 베이스라인(평균값 보정) 모듈
        self.baseline = None

        # 최근 이벤트 상태(터치매니저가 가진 enum 활용)
        self.last_event_state = None

        # 새로운 터치 매니저 초기화 (캘리브레이션용 설정)
        self.touch_manager = TouchManager(
            drag_threshold=1,       # 매우 민감
            touch_timeout=0.02,     # 빠른 반응
            min_drag_distance=2,    # 민감
            max_touch_points=1
        )
        # 히스토리도 작게(빠른 반응)
        self.touch_manager.max_history_size = 3

        # 콜백 등록
        self.touch_manager.on_touch_start = self.on_touch_start
        self.touch_manager.on_touch_move  = self.on_touch_move
        self.touch_manager.on_touch_end   = self.on_touch_end
        self.touch_manager.on_drag_start  = self.on_drag_start
        self.touch_manager.on_drag_move   = self.on_drag_move
        self.touch_manager.on_drag_end    = self.on_drag_end
        self.touch_manager.on_click       = self.on_click

        self.current_step = 0
        self.calibration_data = {
            'touch_points': [],
            'screen_points': [],
            'matrix': None
        }

        self.SCREEN_W = 1920
        self.SCREEN_H = 1080

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('개선된 터치패드 캘리브레이션')
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.BlankCursor)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(20, 20, 20, 20)

        title = QLabel('개선된 터치패드 캘리브레이션 (드래그 지원 + 평균값 보정)')
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        self.status_label = QLabel('시리얼 포트 연결 중...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { font-size:14px; color:#333; padding:10px; background:#e8f4fd; border-radius:5px; }")
        main.addWidget(self.status_label)

        # 터치 상태 표시 라벨
        self.touch_status_label = QLabel('터치 상태: 대기 중')
        self.touch_status_label.setAlignment(Qt.AlignCenter)
        self.touch_status_label.setStyleSheet("QLabel { font-size:12px; color:#666; padding:5px; }")
        main.addWidget(self.touch_status_label)

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

    # ─────────── 시리얼/오프셋 초기화 ───────────
    def setup_serial_and_reader(self):
        try:
            self.ser = serial.Serial(SER_PORT, BAUDRATE, timeout=1)
            self.status_label.setText('오프셋(초기 평균) 측정 중... 패널을 만지지 마세요.')
            QApplication.processEvents()

            init_mean = self.calibrate_offset(num_frames=12)
            if init_mean is None:
                raise ConnectionError("오프셋 측정 실패")

            # 러닝 베이스라인 초기화 (초기 평균으로 시작)
            self.baseline = RunningBaseline(
                shape=(PANEL_ROWS, PANEL_COLS),
                init_mean=init_mean,
                alpha=0.02  # 업데이트 속도 (환경에 맞춰 0.01~0.05 조절)
            )

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

    def calibrate_offset(self, num_frames=12):
        """초기 평균(오프셋) 산출"""
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

    # ─────────── 프레임 처리(평균값 보정 + 적응 임계) ───────────
    def handle_frame(self, frame):
        if self.baseline is None:
            return

        # 평균값 보정 + 시간 스무딩 + 프레임별 동적 임계값
        # last_event_state는 직전 이벤트 상태(터치/드래그 중이면 baseline 업데이트 보수적으로)
        corrected, touch_mask, dyn_thr = self.baseline.subtract_and_update(
            frame_u8=frame,
            app_state=self.app_state,
            last_event_state=self.last_event_state
        )

        # 디버깅 출력(필요시 주석)
        # if corrected.max() > 20:
        #     print(f"DEBUG: 보정 후 최대값: {corrected.max()}, 동적임계: {dyn_thr}")

        # 터치 매니저로 보정 프레임 전달(동적 임계값 사용)
        touch_event = self.touch_manager.process_frame(corrected, dyn_thr)
        if touch_event:
            self.last_event_state = touch_event.state
            self.update_touch_status(touch_event)
        else:
            self.last_event_state = None

    def update_touch_status(self, event: TouchEvent):
        """터치 상태 표시 업데이트"""
        status_text = f"터치 상태: {event.state.value}"
        if event.drag_distance > 0:
            status_text += f" (거리: {event.drag_distance:.1f})"
        if event.duration > 0:
            status_text += f" (시간: {event.duration:.2f}s)"
        self.touch_status_label.setText(status_text)

    def control_mouse(self, touch_point, click_action=None):
        """터치 좌표를 스크린 좌표로 변환하여 마우스 제어"""
        if self.calibration_data['matrix'] is None:
            return

        r, c = touch_point.y, touch_point.x
        x_mat, y_mat = self.calibration_data['matrix']
        x = int(np.polyval(x_mat, r))
        y = int(np.polyval(y_mat, c))
        x = int(np.clip(x, 0, self.SCREEN_W - 1))
        y = int(np.clip(y, 0, self.SCREEN_H - 1))

        try:
            import pyautogui
            pyautogui.moveTo(x, y)
            if click_action == 'click':
                pyautogui.click(x, y)
            elif click_action == 'mousedown':
                pyautogui.mouseDown(x, y)
            elif click_action == 'mouseup':
                pyautogui.mouseUp(x, y)
        except Exception as e:
            print(f"DEBUG: 마우스 제어 실패: {e}")

    # ── 터치 매니저 콜백 함수들 ──
    def on_touch_start(self, event: TouchEvent):
        if self.app_state == 'waiting_start':
            self.reset_wait_period()
            return

        if self.app_state == 'calibrating' and 1 <= self.current_step <= 4:
            self.touch_timer.stop()
            self.touch_timer.start(5000)
            self.countdown = 5
            self.timer.start(1000)
            msgs = {1:'좌측 상단', 2:'우측 상단', 3:'좌측 하단', 4:'우측 하단'}
            self.status_label.setText(f"{msgs[self.current_step]} 터치 감지됨 - 5초 후 기록됩니다.")
        elif self.app_state == 'controlling':
            self.control_mouse(event.current_point, 'click')

    def on_touch_move(self, event: TouchEvent):
        if self.app_state == 'controlling':
            self.control_mouse(event.current_point)

    def on_touch_end(self, event: TouchEvent):
        pass

    def on_drag_start(self, event: TouchEvent):
        self.status_label.setText("드래그 감지됨! 터치를 유지해주세요.")
        if self.app_state == 'controlling':
            self.control_mouse(event.current_point, 'mousedown')

    def on_drag_move(self, event: TouchEvent):
        if self.app_state == 'controlling':
            self.control_mouse(event.current_point)

    def on_drag_end(self, event: TouchEvent):
        self.status_label.setText("드래그 종료. 다시 터치해주세요.")
        if self.app_state == 'controlling':
            self.control_mouse(event.current_point, 'mouseup')

    def on_click(self, event: TouchEvent):
        if self.app_state == 'calibrating' and 1 <= self.current_step <= 4:
            self.timer.stop()
            self.touch_timer.stop()
            self.timer_label.setText('기록 완료')
            self.status_label.setText('터치가 기록되었습니다. 다음 단계로 진행합니다.')

            r, c = event.current_point.y, event.current_point.x
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

            QTimer.singleShot(500, self.start_next_step)

    def on_touch_timeout(self):
        self.timer.stop()
        self.timer_label.setText('시간 초과')
        self.status_label.setText('터치가 없어 같은 단계를 다시 시작합니다.')
        QTimer.singleShot(800, self.start_current_step)

    def start_current_step(self):
        for p in (self.top_left, self.top_right, self.bottom_left, self.bottom_right):
            p.is_active = False
            p.update()

        messages = {
            1: '좌측 상단을 클릭해주세요', 2: '우측 상단을 클릭해주세요',
            3: '좌측 하단을 클릭해주세요', 4: '우측 하단을 클릭해주세요'
        }
        widgets = {1:self.top_left, 2:self.top_right, 3:self.bottom_left, 4:self.bottom_right}

        if self.current_step in messages:
            self.status_label.setText(messages[self.current_step])
            widgets[self.current_step].is_active = True
            widgets[self.current_step].update()
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

            self.status_label.setText('캘리브레이션 완료! 마우스 제어를 시작합니다.')
            self.timer_label.setText('완료')
            self.progress_bar.setValue(4)

            try:
                with open('.calibration_complete', 'w') as f:
                    f.write('calibration_complete')
            except Exception as e:
                print(f"DEBUG: 신호 파일 생성 실패: {e}")

            self.app_state = 'controlling'
            QTimer.singleShot(3000, self.minimize_window)
        else:
            self.status_label.setText('캘리브レー션 실패. 다시 시도해주세요.')

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
        self.status_label.setText('창이 최소화되었습니다. 터치패드 제어를 사용하세요.')

    def reset_wait_period(self):
        self.timer.stop()
        self.wait_timer.stop()
        self.start_wait_period()
        self.status_label.setText('터치 감지되어 10초 대기가 리셋되었습니다...')

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
    window = ImprovedCalibrationGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
