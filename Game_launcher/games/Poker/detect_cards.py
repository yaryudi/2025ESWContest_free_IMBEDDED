import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os
from pathlib import Path

def detect_card_edges(image, debug_path=None):
    MIN_CARD_AREA = 10000  # 카드 최소 면적 (완화)
    MIN_RATIO = 0.5       # 카드 가로/세로 최소 비율 (완화)
    MAX_RATIO = 1.0       # 카드 가로/세로 최대 비율 (완화)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    if debug_path is not None:
        cv2.imwrite(debug_path, thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CARD_AREA:
            continue
        epsilon = 0.04 * cv2.arcLength(contour, True)  # 더 완화
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # 비율 필터
            rect = order_points(approx.reshape(4, 2))
            w = np.linalg.norm(rect[0] - rect[1])
            h = np.linalg.norm(rect[0] - rect[3])
            ratio = w / h if h > 0 else 0
            if MIN_RATIO < ratio < MAX_RATIO:
                card_contours.append(approx)
                if len(card_contours) >= 5:
                    break
    return card_contours

def order_points(pts):
    # 좌표를 정렬: [상단 왼쪽, 상단 오른쪽, 하단 오른쪽, 하단 왼쪽]
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 좌표의 합이 가장 작은 것이 상단 왼쪽, 가장 큰 것이 하단 오른쪽
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 좌표의 차이가 가장 작은 것이 상단 오른쪽, 가장 큰 것이 하단 왼쪽
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts, margin=20):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # margin을 더한 크기
    newWidth = maxWidth + 2 * margin
    newHeight = maxHeight + 2 * margin

    dst = np.array([
        [margin, margin],
        [newWidth - margin - 1, margin],
        [newWidth - margin - 1, newHeight - margin - 1],
        [margin, newHeight - margin - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (newWidth, newHeight))

    return warped

def main():
    # YOLOv8 모델 로드
    try:
        model = YOLO('./playingCards.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # test_image 폴더의 모든 이미지 처리
    image_files = glob.glob('test_image/*.jpg')
    
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        orig_image = image.copy()  # 원본 보존
        debug_thresh_path = f"./debug_image/debug_thresh_{Path(image_path).stem}.jpg"
        card_contours = detect_card_edges(image, debug_path=debug_thresh_path)
        if card_contours:
            # 원본 이미지에만 윤곽선 시각화 (시각화 용도)
            cv2.drawContours(image, card_contours, -1, (0, 255, 0), 2)
            # 각 카드에 대해 처리
            for i, contour in enumerate(card_contours):
                # drawContours가 적용되지 않은 원본에서 crop
                warped = four_point_transform(orig_image, contour.reshape(4, 2), margin=40)
                # 샤프닝 필터 적용
                kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
                sharpened = cv2.filter2D(warped, -1, kernel)
                results = model(sharpened, conf=0.3)
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # 인식률이 가장 높은 박스 선택
                        best_box = max(boxes, key=lambda box: box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                        conf = best_box.conf[0].cpu().numpy()
                        cls = int(best_box.cls[0].cpu().numpy())
                        cv2.rectangle(sharpened, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{result.names[cls]} {conf:.2f}"
                        cv2.putText(sharpened, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                output_path = f"./result_image/result_{Path(image_path).stem}_card{i+1}.jpg"
                cv2.imwrite(output_path, sharpened)
                print(f"Processed card {i+1} from {image_path} -> {output_path}")
        else:
            print(f"No cards detected in {image_path}")

if __name__ == "__main__":
    main() 