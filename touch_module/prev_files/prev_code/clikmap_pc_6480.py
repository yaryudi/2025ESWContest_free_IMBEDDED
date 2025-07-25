import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# 한글 폰트 설정 (Windows Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─── 설정 ───────────────────────────────────────────────────────────
PORT            = 'COM12'
BAUDRATE        = 115200
NUM_ROWS        = 80            # 80행
NUM_COLS        = 63            # 63열
FRAME_SIZE      = NUM_ROWS * NUM_COLS  # 5040 바이트
CAL_FRAMES      = 10
TOUCH_THRESHOLD = 100

# ─── 사분면 정의 ────────────────────────────────────────────────────
center_r, center_c = NUM_ROWS // 2, NUM_COLS // 2
quadrants = {
    '제1사분면(상-좌)': (slice(0, center_r),    slice(0, center_c)),
    '제2사분면(상-우)': (slice(0, center_r),    slice(center_c, NUM_COLS)),
    '제3사분면(하-좌)': (slice(center_r, NUM_ROWS), slice(0, center_c)),
    '제4사분면(하-우)': (slice(center_r, NUM_ROWS), slice(center_c, NUM_COLS)),
}

def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(" •", p.device)

def read_frame():
    raw = ser.read(FRAME_SIZE)
    if len(raw) != FRAME_SIZE:
        return None

    data = np.frombuffer(raw, dtype=np.uint8)
    frame = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8)

    ptr = 0
    for row in range(NUM_ROWS):
        for mux_ch in range(8):        # MUX 채널 0~7
            for dev in range(8):       # MUX 디바이스 A0~A7
                col = dev * 8 + mux_ch
                if col >= NUM_COLS:
                    continue
                val = data[ptr]
                ptr += 1

                # **열을 뒤집어서** (0→NUM_COLS-1, …, NUM_COLS-1→0)
                rev_col = NUM_COLS - 1 - col
                frame[row, rev_col] = val

    return frame

def calibrate():
    print(f"오프셋 보정: 첫 {CAL_FRAMES} 프레임 수집 중… 센서를 건드리지 마세요.")
    buf = []
    while len(buf) < CAL_FRAMES:
        m = read_frame()
        if m is not None:
            buf.append(m.astype(np.float32))
        else:
            time.sleep(0.01)
    offset = np.mean(buf, axis=0).astype(np.float32)
    print("오프셋 보정 완료.")
    return offset

def find_peak(arr, rs, cs):
    sub = arr[rs, cs]
    for (r_off, c_off), v in sorted(
        ((idx, val) for idx, val in np.ndenumerate(sub)),
        key=lambda x: x[1], reverse=True
    ):
        if v < TOUCH_THRESHOLD:
            return None
        r = rs.start + r_off
        c = cs.start + c_off
        if v == arr[r, :].max() and v == arr[:, c].max():
            return r, c, int(v)
    return None

# ─── 메인 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    list_ports()
    mode = input("모드 선택 — 히트맵(h) / 콘솔(c): ").strip().lower()
    use_heatmap = (mode == 'h')

    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    offset = calibrate()
    last_print = time.time()

    if use_heatmap:
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8),
                       cmap='hot', vmin=0, vmax=255)
        ax.set_title(f"{NUM_ROWS}×{NUM_COLS} 센서 배열 (필터링 후)")
        circles = []

    try:
        while True:
            frame = read_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            corr = frame.astype(np.float32) - offset
            corr = np.clip(corr, 0, 255).astype(np.uint8)

            if use_heatmap:
                im.set_data(corr)
                for c in circles: c.remove()
                circles.clear()

            for name, (rs, cs) in quadrants.items():
                pk = find_peak(corr, rs, cs)
                if not pk:
                    continue
                r, c, v = pk

                if use_heatmap:
                    circ = Circle((c, r), radius=1.5, fill=False,
                                  edgecolor='cyan', linewidth=2)
                    ax.add_patch(circ)
                    circles.append(circ)
                else:
                    print(f"{name}: r={r}, c={c}, 값={v}")

            if use_heatmap:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # 콘솔 출력 0.5초마다
            now = time.time()
            if not use_heatmap and now - last_print >= 0.5:
                last_print = now

    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        ser.close()
