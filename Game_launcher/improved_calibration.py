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
SER_PORT    = '/dev/ttyUSB0'
BAUDRATE    = 2000000
PANEL_ROWS    = 80
PANEL_COLS    = 63
FRAME_SIZE    = PANEL_ROWS * PANEL_COLS
MUX_CHANNELS  = 8
DEVICES       = 8
TOUCH_THRESHOLD = 8   # 극한 민감한 터치 감지를 위해 임계값 더 낮춤
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
class ImprovedCalibrationGUI(QMainWindow):
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
        
        # 새로운 터치 매니저 초기화 (캘리브레이션용 설정)
        self.touch_manager = TouchManager(
            drag_threshold=1,       # 드래그 시작 임계값 (센서 단위) - 매우 민감하게 설정
            touch_timeout=0.02,     # 터치 타임아웃 (초) - 극한 빠른 반응
            min_drag_distance=2,    # 최소 드래그 거리 - 극한 민감하게 설정
            max_touch_points=1      # 최대 터치 포인트 수
        )
        
        # 터치 히스토리 크기 극한 감소 (극한 반응 속도를 위해)
        self.touch_manager.max_history_size = 3
        
        # 터치 매니저 콜백 설정
        self.touch_manager.on_touch_start = self.on_touch_start
        self.touch_manager.on_touch_move = self.on_touch_move
        self.touch_manager.on_touch_end = self.on_touch_end
        self.touch_manager.on_drag_start = self.on_drag_start
        self.touch_manager.on_drag_move = self.on_drag_move
        self.touch_manager.on_drag_end = self.on_drag_end
        self.touch_manager.on_click = self.on_click

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

        title = QLabel('개선된 터치패드 캘리브레이션 (드래그 지원)')
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        self.status_label = QLabel('시리얼 포트 연결 중...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { font-size:14px; color:#333; padding:10px; background:#e8f4fd; border-radius:5px; }")
        main.addWidget(self.status_label)

        # 터치 상태 표시 라벨 추가
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
        """새로운 터치 매니저를 사용한 프레임 처리"""
        if self.offset is None:
            return

        # 프레임 전처리
        corr = frame.astype(np.float32) - self.offset
        corr = np.clip(corr, 0, 255).astype(np.uint8)
        
        # 최대 압력값 확인 (노이즈 디버깅)
        max_pressure = corr.max()
        if max_pressure > 20:  # 임계값 이상일 때만 출력
            print(f"DEBUG: 프레임 최대 압력: {max_pressure}")
        
        # 터치 매니저로 프레임 처리
        touch_event = self.touch_manager.process_frame(corr, TOUCH_THRESHOLD)
        
        if touch_event:
            self.update_touch_status(touch_event)

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
        
        # 캘리브레이션 매트릭스를 사용하여 좌표 변환
        x = int(np.polyval(x_mat, r))
        y = int(np.polyval(y_mat, c))
        
        # 화면 범위 내로 제한
        x = int(np.clip(x, 0, self.SCREEN_W - 1))
        y = int(np.clip(y, 0, self.SCREEN_H - 1))
        
        print(f"DEBUG: 마우스 제어 - 터치: ({r}, {c}) → 스크린: ({x}, {y}) - 액션: {click_action}")
        
        try:
            import pyautogui
            pyautogui.moveTo(x, y)
            
            # 클릭 액션 처리
            if click_action == 'click':
                pyautogui.click(x, y)
                print(f"DEBUG: 클릭 실행 - ({x}, {y})")
            elif click_action == 'mousedown':
                pyautogui.mouseDown(x, y)
                print(f"DEBUG: 마우스 다운 - ({x}, {y})")
            elif click_action == 'mouseup':
                pyautogui.mouseUp(x, y)
                print(f"DEBUG: 마우스 업 - ({x}, {y})")
                
        except Exception as e:
            print(f"DEBUG: 마우스 제어 실패: {e}")

    # ── 터치 매니저 콜백 함수들 ──
    def on_touch_start(self, event: TouchEvent):
        """터치 시작 콜백"""
        print(f"DEBUG: 터치 시작 - 상태: {self.app_state}, 단계: {self.current_step}")
        if self.app_state == 'waiting_start':
            self.reset_wait_period()
            return

        if self.app_state == 'calibrating' and 1 <= self.current_step <= 4:
            print(f"DEBUG: 캘리브레이션 터치 감지 - 단계 {self.current_step}")
            self.touch_timer.stop()
            self.touch_timer.start(5000)
            self.countdown = 5
            self.timer.start(1000)
            msgs = {
                1: '좌측 상단', 2: '우측 상단',
                3: '좌측 하단', 4: '우측 하단'
            }
            self.status_label.setText(f"{msgs[self.current_step]} 터치 감지됨 - 5초 후 기록됩니다.")
        elif self.app_state == 'controlling':
            # 마우스 제어 모드에서는 터치 시작 시 마우스 이동 + 클릭
            print(f"DEBUG: 마우스 제어 모드 - 터치 시작")
            self.control_mouse(event.current_point, 'click')

    def on_touch_move(self, event: TouchEvent):
        """터치 이동 콜백"""
        print(f"DEBUG: 터치 이동 - 거리: {event.drag_distance:.1f}")
        if self.app_state == 'controlling':
            print(f"DEBUG: 마우스 제어 모드 - 터치 이동")
            self.control_mouse(event.current_point)
        pass  # 캘리브레이션에서는 이동 무시

    def on_touch_end(self, event: TouchEvent):
        """터치 종료 콜백"""
        print(f"DEBUG: 터치 종료 - 지속시간: {event.duration:.2f}s, 거리: {event.drag_distance:.1f}")
        pass  # 클릭 이벤트에서 처리

    def on_drag_start(self, event: TouchEvent):
        """드래그 시작 콜백"""
        print(f"DEBUG: 드래그 시작 - 거리: {event.drag_distance:.1f}, 임계값: {self.touch_manager.drag_threshold}")
        self.status_label.setText("드래그 감지됨! 터치를 유지해주세요.")
        if self.app_state == 'controlling':
            print(f"DEBUG: 마우스 제어 모드 - 드래그 시작 (마우스 다운)")
            self.control_mouse(event.current_point, 'mousedown')

    def on_drag_move(self, event: TouchEvent):
        """드래그 이동 콜백"""
        print(f"DEBUG: 드래그 이동 - 거리: {event.drag_distance:.1f}")
        if self.app_state == 'controlling':
            print(f"DEBUG: 마우스 제어 모드 - 드래그 이동")
            self.control_mouse(event.current_point)
        pass

    def on_drag_end(self, event: TouchEvent):
        """드래그 종료 콜백"""
        print(f"DEBUG: 드래그 종료 - 총 거리: {event.drag_distance:.1f}")
        self.status_label.setText("드래그 종료. 다시 터치해주세요.")
        if self.app_state == 'controlling':
            print(f"DEBUG: 마우스 제어 모드 - 드래그 종료 (마우스 업)")
            self.control_mouse(event.current_point, 'mouseup')

    def on_click(self, event: TouchEvent):
        """클릭 콜백 - 캘리브레이션에서 사용"""
        print(f"DEBUG: 클릭 감지! - 상태: {self.app_state}, 단계: {self.current_step}")
        if self.app_state == 'calibrating' and 1 <= self.current_step <= 4:
            print(f"DEBUG: 캘리브레이션 클릭 처리 - 단계 {self.current_step}")
            
            # 모든 타이머 중지
            self.timer.stop()
            self.touch_timer.stop()
            
            self.timer_label.setText('기록 완료')
            self.status_label.setText('터치가 기록되었습니다. 다음 단계로 진행합니다.')

            # 터치 좌표를 센서 좌표로 변환
            r, c = event.current_point.y, event.current_point.x
            print(f"DEBUG: 터치 좌표 - 센서: ({r}, {c})")
            
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

            print(f"DEBUG: 다음 단계로 진행 예정 - 현재 단계: {self.current_step}")
            QTimer.singleShot(500, self.start_next_step)
        else:
            print(f"DEBUG: 클릭 무시 - 상태: {self.app_state}, 단계: {self.current_step}")

    def on_touch_timeout(self):
        """터치 타임아웃 처리"""
        self.timer.stop()
        self.timer_label.setText('시간 초과')
        self.status_label.setText('터치가 없어 같은 단계를 다시 시작합니다.')
        QTimer.singleShot(800, self.start_current_step)

    def start_current_step(self):
        print(f"DEBUG: start_current_step 호출됨 - 단계: {self.current_step}")
        
        for p in (self.top_left, self.top_right, self.bottom_left, self.bottom_right):
            p.is_active = False
            p.update()

        messages = {
            1: '좌측 상단을 클릭해주세요', 2: '우측 상단을 클릭해주세요',
            3: '좌측 하단을 클릭해주세요', 4: '우측 하단을 클릭해주세요'
        }
        widgets = {
            1: self.top_left, 2: self.top_right,
            3: self.bottom_left, 4: self.bottom_right
        }

        if self.current_step in messages:
            print(f"DEBUG: 단계 {self.current_step} 시작 - {messages[self.current_step]}")
            self.status_label.setText(messages[self.current_step])
            widgets[self.current_step].is_active = True
            widgets[self.current_step].update()

            self.countdown = 5
            self.timer_label.setText(f'{self.countdown}초')
            self.timer.start(1000)
            self.touch_timer.start(5000)
        else:
            print(f"DEBUG: 모든 단계 완료 - 캘리브레이션 종료")
            self.finish_calibration()

    def start_next_step(self):
        print(f"DEBUG: start_next_step 호출됨 - 현재 단계: {self.current_step}")
        self.current_step += 1
        print(f"DEBUG: 단계 증가됨 - 새로운 단계: {self.current_step}")
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
