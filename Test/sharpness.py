# sharpness_evaluation.py
# ----------------------------------
# Pi 카메라로 투사된 화면의 선명도(Sharpness)를 자동 평가하는 스크립트

import cv2
import numpy as np
import time
from picamera2 import Picamera2

# --- 캘리브레이션 상수 (Pi 카메라 기준) ---
S_MIN = 300.0      # 흐릿 샘플 평균 Laplacian 분산
S_MAX = 1200.0     # 선명 샘플 평균 Laplacian 분산
THRESHOLD = 70.0   # 0~100 점수 환산 후 합격 기준

# --- ROI 크기 (투사 영역 리사이즈) ---
ROI_WIDTH = 640
ROI_HEIGHT = 480

# ----------------------------------
# Pi 카메라로부터 BGR 이미지 캡처
# ----------------------------------
def capture_image() -> np.ndarray:
    picam2 = Picamera2()
    config = picam2.create_still_configuration({
        "format": "XRGB8888",
        "size": (1280, 720)
    })
    picam2.configure(config)
    picam2.start()
    frame = picam2.capture_array()
    picam2.stop()
    # RGB -> BGR 변환
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ----------------------------------
# 이미지에서 스크린(투사 영역) ROI 추출 및 투시 보정
# ----------------------------------
def extract_screen_roi(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))
    screen = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(screen, True)
    approx = cv2.approxPolyDP(screen, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        rect = np.array([tl, tr, br, bl], dtype="float32")
        dst = np.array([
            [0, 0],
            [ROI_WIDTH - 1, 0],
            [ROI_WIDTH - 1, ROI_HEIGHT - 1],
            [0, ROI_HEIGHT - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(gray, M, (ROI_WIDTH, ROI_HEIGHT))
    return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))

# ----------------------------------
# Laplacian 분산 기반 샤프니스 계산
# ----------------------------------
def compute_laplacian_sharpness(gray_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    return float(lap.var())

# ----------------------------------
# 0~100 점수화
# ----------------------------------
def normalize_score(S: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    N = (S - s_min) / (s_max - s_min)
    N = max(0.0, min(1.0, N))
    return N * 100.0

# ----------------------------------
# 메인 실행부
# ----------------------------------
def main():
    print("Pi 카메라로 이미지 캡처 전대기 중… 3초 대기")
    time.sleep(1)
    print("Pi 카메라로 이미지 캡처 전대기 중… 2초 대기")
    time.sleep(1)
    print("Pi 카메라로 이미지 캡처 전대기 중… 1초 대기")
    time.sleep(1)
    
    img = capture_image()
    print("Pi 카메라로 이미지 캡처 중...")
    img = capture_image()
    print("ROI 추출 및 투시 보정 중...")
    roi = extract_screen_roi(img)
    print("샤프니스 계산 중...")
    S = compute_laplacian_sharpness(roi)
    score = normalize_score(S)
    print(f"  Laplacian 분산: {S:.1f}")
    print(f"  Sharpness 점수: {score:.1f}/100 (기준 {THRESHOLD}점)")
    print("✅ 합격" if score >= THRESHOLD else "❌ 기준 미만")

if __name__ == '__main__':
    main()
