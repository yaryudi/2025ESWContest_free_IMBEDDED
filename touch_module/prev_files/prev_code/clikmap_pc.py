import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# 한글 폰트 설정 (NanumGothic)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ─── 설정 ───────────────────────────────────────────────────────────
PORT              = 'COM15'
BAUDRATE          = 2000000
NUM_ROWS          = 80            # 80행
NUM_COLS          = 64            # 64열
FRAME_SIZE        = NUM_ROWS * NUM_COLS
CAL_FRAMES        = 10
TOUCH_THRESHOLD_MIN = 8           # 프레임별 동적 임계값의 하한
SYNC_BYTES        = b'\xAA\x55'   # Arduino에서 보내는 프레임 헤더

# ─── 사분면 정의 ────────────────────────────────────────────────────
center_r, center_c = NUM_ROWS // 2, NUM_COLS // 2
quadrants = {
    '제1사분면(상-좌)': (slice(0, center_r),            slice(0, center_c)),
    '제2사분면(상-우)': (slice(0, center_r),            slice(center_c, NUM_COLS)),
    '제3사분면(하-좌)': (slice(center_r, NUM_ROWS),     slice(0, center_c)),
    '제4사분면(하-우)': (slice(center_r, NUM_ROWS),     slice(center_c, NUM_COLS)),
}

# ─── 인덱스 매핑 테이블 (버퍼 인덱스 → (row, col)) ─────────────────
index_map = []
for row in range(NUM_ROWS):
    for mux in range(8):
        for dev in range(8):
            col = dev * 8 + mux
            if col < NUM_COLS:
                rev_col = NUM_COLS - 1 - col
                index_map.append((row, rev_col))


def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(" •", p.device)


def sync_frame(ser):
    """0xAA 0x55 헤더를 찾아서 동기화"""
    while True:
        b = ser.read(1)
        if b == SYNC_BYTES[:1]:
            b2 = ser.read(1)
            if b2 == SYNC_BYTES[1:]:
                return


def read_frame(ser):
    # 1) 헤더 동기화
    sync_frame(ser)
    # 2) 본문 읽기
    raw = ser.read(FRAME_SIZE)
    if len(raw) != FRAME_SIZE:
        return None

    data = np.frombuffer(raw, dtype=np.uint8)
    frame = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8)
    for i, v in enumerate(data):
        r, c = index_map[i]
        frame[r, c] = v
    return frame


def calibrate(ser):
    print(f"오프셋 보정: 첫 {CAL_FRAMES} 프레임 수집 중… 센서를 건드리지 마세요.")
    buf = []
    ser.reset_input_buffer()
    while len(buf) < CAL_FRAMES:
        m = read_frame(ser)
        if m is not None:
            buf.append(m.astype(np.float32))
        else:
            time.sleep(0.01)
    offset = np.mean(buf, axis=0).astype(np.float32)
    print("오프셋 보정 완료.")
    return offset


# ─── 러닝 베이스라인(평균값 보정 + 동적 임계 + 시간 스무딩) ─────────
class RunningBaseline:
    """
    - 초기: N프레임 평균으로 baseline 시작
    - 이후: 터치 후보(고압) 제외한 픽셀만 EMA(alpha)로 baseline 업데이트
    - subtract_and_update(frame_u8) -> (corrected_u8, touch_mask, dyn_thr)
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

        # 시간적 스무딩 버퍼
        self.prev_corrected = np.zeros(shape, dtype=np.float32)

    @staticmethod
    def robust_stats(arr: np.ndarray):
        """중앙값과 MAD 기반의 강건한 분산 추정 (MAD*1.4826 ≈ std)."""
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        robust_std = 1.4826 * mad
        return med, robust_std

    def _provisional_touch_mask(self, corrected: np.ndarray, base_thr: float):
        """베이스라인 업데이트에서 제외할 터치 후보 마스크(보수적)."""
        return corrected > base_thr

    def subtract_and_update(self, frame_u8: np.ndarray):
        """
        1) baseline subtraction
        2) 중앙값+3*MAD로 provisional threshold → 터치 마스크 산출
        3) 터치 아닌 픽셀만 baseline EMA 업데이트
        4) 시간적 EMA 스무딩
        5) 프레임 동적 임계값 계산(하한 적용)
        """
        f = frame_u8.astype(np.float32)
        corrected = f - self.baseline
        corrected = np.clip(corrected, self.clip_min, self.clip_max)

        # 보수적 터치 마스크 산출용 provisional threshold
        med, rstd = self.robust_stats(corrected)
        provisional_thr = med + 3.0 * max(rstd, 1.0)
        touch_mask = self._provisional_touch_mask(corrected, provisional_thr)

        # 베이스라인 업데이트(터치 제외)
        not_touch = ~touch_mask
        self.baseline[not_touch] = (
            (1.0 - self.alpha) * self.baseline[not_touch] +
            self.alpha * f[not_touch]
        )

        # 시간적 스무딩(EMA)
        beta = 0.5  # 현재 프레임 가중치
        smoothed = beta * corrected + (1.0 - beta) * self.prev_corrected
        self.prev_corrected = smoothed

        # 프레임별 동적 임계값
        med2, rstd2 = self.robust_stats(smoothed)
        dyn_thr = int(max(TOUCH_THRESHOLD_MIN, med2 + 3.0 * max(rstd2, 1.0)))

        smoothed_u8 = np.clip(smoothed, 0, 255).astype(np.uint8)
        return smoothed_u8, touch_mask, dyn_thr


def find_peak(arr: np.ndarray, rs: slice, cs: slice, thr: int):
    """
    사분면(sub-array)에서 최대치 후보를 내려가며 찾되,
    - 값이 thr 미만이면 중단
    - 행/열 최댓값 조건을 만족하는 지점 반환
    """
    sub = arr[rs, cs]
    # 값 내림차순 순회
    for (r_off, c_off), v in sorted(
        ((idx, val) for idx, val in np.ndenumerate(sub)),
        key=lambda x: x[1], reverse=True
    ):
        if v < thr:
            return None
        r = rs.start + r_off
        c = cs.start + c_off
        if v == arr[r, :].max() and v == arr[:, c].max():
            return r, c, int(v)
    return None


if __name__ == "__main__":
    list_ports()
    mode = input("모드 선택 — 히트맵(h) / 콘솔(c): ").strip().lower()
    use_heatmap = (mode == 'h')

    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    try:
        # 초기 평균(오프셋)으로 러닝 베이스라인 시작
        init_mean = calibrate(ser)
        rb = RunningBaseline(shape=(NUM_ROWS, NUM_COLS), init_mean=init_mean, alpha=0.02)

        last_print = time.time()

        if use_heatmap:
            plt.ion()
            fig, ax = plt.subplots()
            im = ax.imshow(np.zeros((NUM_ROWS, NUM_COLS)), cmap='hot', vmin=0, vmax=255)
            title = ax.set_title(f"{NUM_ROWS}×{NUM_COLS} 센서 배열 (보정/스무딩 후)")
            circles = []

        while True:
            frame = read_frame(ser)
            if frame is None:
                time.sleep(0.001)
                continue

            # 평균값 보정 + 시간 스무딩 + 동적 임계값
            corr, touch_mask, dyn_thr = rb.subtract_and_update(frame)

            if use_heatmap:
                im.set_data(corr)
                # 기존 마커 제거
                for c in circles:
                    c.remove()
                circles.clear()

            # 사분면별 피크 탐지(프레임별 임계값 사용)
            for name, (rs, cs) in quadrants.items():
                pk = find_peak(corr, rs, cs, dyn_thr)
                if not pk:
                    continue
                r, c, v = pk
                if use_heatmap:
                    circ = Circle((c, r), radius=1.5, fill=False, edgecolor='cyan', linewidth=2)
                    ax.add_patch(circ)
                    circles.append(circ)
                else:
                    now = time.time()
                    # 콘솔 모드는 너무 자주 찍히면 지저분하니 약간 텀 두기(옵션)
                    if now - last_print >= 0.05:
                        print(f"{name}: r={r}, c={c}, 값={v}, 임계={dyn_thr}")
                        last_print = now

            if use_heatmap:
                # 제목에 현재 동적 임계값 표기
                title.set_text(f"{NUM_ROWS}×{NUM_COLS} 센서 배열 (보정/스무딩 후)  |  동적 임계값: {dyn_thr}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("프로그램 종료")
    finally:
        ser.close()
