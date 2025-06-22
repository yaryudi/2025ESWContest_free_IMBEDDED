# sharpness_evaluation.py
# ----------------------------------
# Pi 카메라 또는 파일 이미지를 읽어 투사된 화면(테스트 차트) 선명도를 계산하는 스크립트

import cv2
import numpy as np
import argparse

# Picamera2 모듈 시도 임포트
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# 캘리브레이션: Pi 카메라 흐림/선명 샘플 평균 분산
S_MIN = 300.0
S_MAX = 1200.0

# ROI 크기 (투사 영역 리사이즈)
ROI_WIDTH = 640
ROI_HEIGHT = 480

# ----------------------------------
# Pi 카메라 캡처 함수
# ----------------------------------
def capture_image_from_pi() -> np.ndarray:
    if not CAMERA_AVAILABLE:
        raise RuntimeError("Picamera2 모듈을 찾을 수 없습니다. 설치 후 다시 시도하세요.")
    picam2 = Picamera2()
    config = picam2.create_still_configuration({"format": "XRGB8888", "size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    frame = picam2.capture_array()
    picam2.stop()
    # RGB -> BGR 변환
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ----------------------------------
# ROI 추출 및 투시 보정
# ----------------------------------
def extract_screen_roi(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))
    screen_cnt = max(contours, key=lambda c: cv2.contourArea(c))
    peri = cv2.arcLength(screen_cnt, True)
    approx = cv2.approxPolyDP(screen_cnt, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        rect = np.array([tl, tr, br, bl], dtype="float32")
        dst = np.array([[0,0],[ROI_WIDTH-1,0],[ROI_WIDTH-1,ROI_HEIGHT-1],[0,ROI_HEIGHT-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(gray, M, (ROI_WIDTH, ROI_HEIGHT))
    return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))

# ----------------------------------
# Sharpness 계산 & 점수화
# ----------------------------------
def compute_laplacian_sharpness(gray_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    return float(lap.var())

def normalize_score(S: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    N = (S - s_min) / (s_max - s_min)
    N = max(0.0, min(1.0, N))
    return N * 100.0

# ----------------------------------
# 평가 처리 함수
# ----------------------------------
def evaluate_image(img: np.ndarray, threshold: float):
    roi = extract_screen_roi(img)
    S = compute_laplacian_sharpness(roi)
    score = normalize_score(S)
    print(f"Laplacian variance: {S:.1f}")
    print(f"Sharpness score: {score:.1f}/100 (threshold={threshold})")
    print("✅ 만족" if score >= threshold else "❌ 미만")

# ----------------------------------
# 커맨드라인 인터페이스
# ----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate sharpness of a projected screen via file or Pi camera."
    )
    parser.add_argument('--capture', action='store_true', help='Pi 카메라로부터 캡처하여 평가')
    parser.add_argument('--image', type=str, help='평가할 이미지 파일 경로')
    parser.add_argument('--threshold', type=float, default=70.0, help='점수 임계값 (0-100)')
    args = parser.parse_args()

    if args.capture:
        print("카메라로부터 이미지 캡처 중...")
        img = capture_image_from_pi()
        evaluate_image(img, args.threshold)
    elif args.image:
        print(f"파일 읽기: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print(f"[Error] 이미지 로드 실패: {args.image}")
        else:
            evaluate_image(img, args.threshold)
    else:
        parser.print_help()

