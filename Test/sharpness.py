# sharpness_evaluation.py
# ----------------------------------
# 투사된 ISO e-SFR 차트나 기타 이미지를 파일로 읽어
# ROI(스크린 영역)를 추출한 뒤 Laplacian 분산 기반 선명도 점수(0~100)를 계산하는 스크립트

import cv2
import numpy as np
import argparse

# 1) 캘리브레이션 결과값: 흐림 샘플(S_MIN), 선명 샘플(S_MAX)
S_MIN = 300.0      # Pi 카메라 흐림 상태 평균 분산
S_MAX = 1200.0     # Pi 카메라 최적 선명 상태 평균 분산

# 2) ROI 크기 (투사된 화면 영역 리사이즈)
ROI_WIDTH = 640
ROI_HEIGHT = 480

# ----------------------------------
# ROI 추출: 스크린/차트 영역 검출 + 투시 보정
# ----------------------------------
def extract_screen_roi(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: 전체 이미지를 리사이즈
        return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))

    # 가장 큰 컨투어 선택
    screen_cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(screen_cnt, True)
    approx = cv2.approxPolyDP(screen_cnt, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        # 4점 정렬: tl, tr, br, bl
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
        warped = cv2.warpPerspective(gray, M, (ROI_WIDTH, ROI_HEIGHT))
        return warped
    else:
        # 컨투어이지만 사각형이 아니면 전체 리사이즈
        return cv2.resize(gray, (ROI_WIDTH, ROI_HEIGHT))

# ----------------------------------
# Sharpness 계산: Laplacian 분산
# ----------------------------------
def compute_laplacian_sharpness(gray_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    return float(lap.var())

# ----------------------------------
# 정규화 및 점수화
# ----------------------------------
def normalize_score(S: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    # 0~1 정규화 후 0~100 점수
    N = (S - s_min) / (s_max - s_min)
    N = max(0.0, min(1.0, N))
    return N * 100.0

# ----------------------------------
# 파일 평가 함수
# ----------------------------------
def evaluate_file(image_path: str, threshold: float):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Cannot load image: {image_path}")
        return
    roi = extract_screen_roi(img)
    S = compute_laplacian_sharpness(roi)
    score = normalize_score(S)

    print(f"Image: {image_path}")
    print(f"Laplacian variance: {S:.1f}")
    print(f"Sharpness score: {score:.1f}/100 (threshold={threshold})")
    print("✅ 만족" if score >= threshold else "❌ 미만")

# ----------------------------------
# 커맨드라인 인터페이스
# ----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate sharpness of a projected screen or test chart image."
    )
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--threshold', type=float, default=70.0,
                        help='Score threshold for pass/fail')
    args = parser.parse_args()
    evaluate_file(args.image, args.threshold)
