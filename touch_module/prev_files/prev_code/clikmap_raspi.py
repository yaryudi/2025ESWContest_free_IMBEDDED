import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import pyautogui   # pip install pyautogui

#최적화용 python 코드

# 한글 폰트 설정 (Windows Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─── 설정 ───────────────────────────────────────────────────────────
PORT            = '/dev/ttyACM0'       # 실제 연결된 포트로 수정 #라즈베리에 연결할 경우, /dev/ttyACM0 으로 수정! - 다를 sudo 있음
BAUDRATE        = 115200
NUM_ROWS        = 40
NUM_COLS        = 30
FRAME_SIZE      = NUM_ROWS * NUM_COLS
CAL_FRAMES      = 10           # 캘리브레이션용 프레임 수
TOUCH_THRESHOLD = 30           # 터치로 판단할 최소값

# ─── 화면 해상도 ─────────────────────────────────────────────────────
SCREEN_W        = 1280       # 화면 해상도 가로(px)
SCREEN_H        = 800        # 화면 해상도 세로(px)
ADJUST_X        = 0#2560     # 마우스 X축 보정값(px), 오른쪽으로 치우칠 때 음수로 조정
ADJUST_Y        = 0#150    # 마우스 Y축 보정값(px)         # 화면 해상도 세로(px)

# ─── 사분면 정의 ────────────────────────────────────────────────────
center_r, center_c = NUM_ROWS//2, NUM_COLS//2
quadrants = {
    '제1사분면(상-좌)': (slice(0, center_r),    slice(0, center_c)),
    '제2사분면(상-우)': (slice(0, center_r),    slice(center_c, NUM_COLS)),
    '제3사분면(하-좌)': (slice(center_r, NUM_ROWS), slice(0, center_c)),
    '제4사분면(하-우)': (slice(center_r, NUM_ROWS), slice(center_c, NUM_COLS)),
}

# ─── 유틸 함수들 ─────────────────────────────────────────────────────
def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(" •", p.device)


def read_frame():
    raw = ser.read(FRAME_SIZE)
    if len(raw) != FRAME_SIZE:
        return None
    
    # 바이트 데이터를 1D 배열로 변환
    data = np.frombuffer(raw, dtype=np.uint8)
    
    # MUX 패턴에 따라 데이터 재구성
    frame = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8)
    for row in range(NUM_ROWS):
        for mux_ch in range(8):  # MUX 채널 (0-7)
            for dev in range(4):  # MUX 디바이스 (0-3)
                col = dev * 8 + mux_ch
                if col >= NUM_COLS:
                    continue
                # MUX 패턴에 맞는 인덱스 계산
                idx = row * NUM_COLS + mux_ch * 4 + dev
                if idx < len(data):
                    
                    #if(col == 6):
                    #    frame[row, 25] = data[idx]
                    if(col == 15):
                        frame[row, 23] = data[idx] 
                    elif(col == 7):
                        frame[row, 16] = data[idx] 
                    else:
                        frame[row, col] = data[idx]
    
    return frame


def calibrate():
    print(f"오프셋 보정: 첫 {CAL_FRAMES} 프레임 수집 중… 센서를 건드리지 마세요.")
    frames = []
    while len(frames) < CAL_FRAMES:
        m = read_frame()
        if m is not None:
            frames.append(m)
        else:
            time.sleep(0.01)
    offset = np.mean(frames, axis=0).astype(np.float32)
    print("오프셋 보정 완료.")
    return offset



def keep_row_col_max_intersection(arr):
    row_max = arr.max(axis=1, keepdims=True)
    col_max = arr.max(axis=0, keepdims=True)
    mask = (arr == row_max) & (arr == col_max)
    return arr * mask


def find_peak(arr, rs, cs):
    sub = arr[rs, cs]
    candidates = sorted(
        ((v, r, c) for (r, c), v in np.ndenumerate(sub)),
        key=lambda x: x[0], reverse=True
    )
    for value, r_sub, c_sub in candidates:
        if value < TOUCH_THRESHOLD:
            continue
        r = rs.start + r_sub
        c = cs.start + c_sub
        if np.max(arr[r, :]) > value or np.max(arr[:, c]) > value:
            continue
        return r, c, value
    return None

# ─── 메인 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    list_ports()
    mode = input("모드 선택 — 히트맵(h) / 위치이동(m): ").strip().lower()
    use_heatmap = (mode == 'h')

    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    offset = calibrate()
    last_print = time.time()

    # 히트맵용 Matplotlib 세팅
    if use_heatmap:
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8),
                       cmap='hot', vmin=0, vmax=255)
        ax.set_title("40x30 센서 배열 (필터링 후)")
        circles = []

    try:
        while True:
            frame = read_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            corr = frame.astype(np.float32) - offset
            corr = np.clip(corr, 0, 255).astype(np.uint8)
            filtered = keep_row_col_max_intersection(corr)

            # 히트맵 업데이트
            if use_heatmap:
                for circ in circles:
                    circ.remove()
                circles.clear()
                im.set_data(filtered)

            # 터치 피크 검출 및 처리
            for name, (rs, cs) in quadrants.items():
                peak = find_peak(filtered, rs, cs)
                if not peak:
                    continue
                r, c, v = peak

                if use_heatmap:
                    circ = Circle((c, r), radius=1.5, fill=False,
                                  edgecolor='cyan', linewidth=2)
                    ax.add_patch(circ)
                    circles.append(circ)
                else:
                    # x, y 좌표를 서로 바꿈: 행->x, 열->y 매핑
                    x_px = int(round((NUM_ROWS - 1 - r) / (NUM_ROWS - 1) * (SCREEN_W - 1)))
                    y_px = int(round((NUM_COLS - 1 - c) / (NUM_COLS - 1) * (SCREEN_H - 1)))
                    # 보정값 적용
                    x_px = x_px + ADJUST_X
                    y_px = y_px + ADJUST_Y
                    y_px = SCREEN_H - y_px
                    pyautogui.moveTo(x_px, y_px)
                    #pyautogui.click()

            # 히트맵 화면 갱신
            if use_heatmap:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # 주기적 콘솔 출력
            now = time.time()
            if now - last_print >= 0.5:
                for name, (rs, cs) in quadrants.items():
                    peak = find_peak(filtered, rs, cs)
                    print(f"{name}: 값={peak[2] if peak else '없음'}")
                last_print = now

    except KeyboardInterrupt:
        print("사용자 중단")
    except serial.SerialException as e:
        print(f"시리얼 오류: {e}")
    finally:
        ser.close()
        print("프로그램 종료 및 포트 닫힘")