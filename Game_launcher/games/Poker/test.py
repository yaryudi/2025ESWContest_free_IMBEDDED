"""
카드 인식 테스트 모듈
라즈베리파이 IMX708 광각 카메라로 이미지를 캡처하고 YOLO 모델을 사용하여 포커 카드를 인식합니다.
최적화된 버전: 좌표만 미리 추출하고 필요할 때만 카드 인식
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from picamera2 import Picamera2
import multiprocessing as mp
from functools import partial

def process_card_worker(model_path, warped_image):
    """별도의 프로세스에서 실행될 카드 처리 함수"""
    try:
        model = YOLO(model_path)
        results = model(warped_image, conf=0.3)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0 and len(results[0].boxes) > 0:
                first_box = boxes[0]
                cls = int(first_box.cls[0].cpu().numpy())
                return result.names[cls]
        return "Unknown"
    except Exception as e:
        print(f"Error in worker process: {e}")
        return "Unknown"

class CardDetector:
    def __init__(self, num_players=5):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (3840, 2160), "format": "RGB888"},
            controls={
                
                "ScalerCrop": (0, 0, 3840, 2160)
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.model_path = "playingCards.pt"
        self.num_players = num_players
        self.update_card_positions()
        # 멀티프로세싱을 위한 프로세스 풀 생성
        self.pool = mp.Pool(processes=mp.cpu_count())
        # 카드 좌표 저장 변수
        self.card_coordinates = None
        self.last_coordinate_extraction = 0

    def update_card_positions(self):
        """플레이어 수에 따라 카드 위치 정보를 업데이트"""

        positions = {}
        current_index = 0

        if self.num_players == 2:
            for i in range(1, 3):
                positions[f'player{i}'] = [current_index, current_index + 1]
                current_index += 2

        if self.num_players == 3:
            for i in range(1, 4):
                positions[f'player{i}'] = [current_index, current_index + 1]
                current_index += 2

        # # 덱 ( 덱 카드 인식 안함 )
        # positions['deck'] = [current_index]
        # current_index += 1

        # 커뮤니티 카드
        positions['community'] = list(range(current_index, current_index + 5))
        current_index += 5

        # 플레이어 4, 5의 카드 (선택적)
        if self.num_players == 4:
            positions['player4'] = [current_index, current_index + 1]
            current_index += 2
        if self.num_players == 5:
            positions['player5'] = [current_index, current_index + 1]
            current_index += 2
            positions['player4'] = [current_index, current_index + 1]
            current_index += 2

        self.card_positions = positions
        self.total_cards = current_index

    def extract_card_coordinates(self):
        """카드 좌표만 추출하여 저장"""
        try:
            image = self.picam2.capture_array()
            image_path = f"assets/test_image/capture.jpg"
            cv2.imwrite(image_path, image)
            
            card_contours = self.detect_card_edges(image)
            
            if card_contours and len(card_contours) >= self.total_cards:
                card_contours = self.sort_contours_reading_order(card_contours)
                self.card_coordinates = card_contours[:self.total_cards]
                self.last_coordinate_extraction = time.time()
                print(f"카드 좌표 추출 완료: {len(self.card_coordinates)}개")
                return True
            else:
                print(f"카드 좌표 추출 실패: {len(card_contours) if card_contours else 0}개 발견")
                return False
        except Exception as e:
            print(f"좌표 추출 오류: {e}")
            return False

    def validate_coordinates(self):
        """좌표가 여전히 유효한지 확인"""
        if not hasattr(self, 'card_coordinates') or self.card_coordinates is None:
            return False
        
        # 좌표 개수 확인
        if len(self.card_coordinates) < self.total_cards:
            return False
        
        # 시간 기반 유효성 검사 (30초마다 재추출)
        if time.time() - self.last_coordinate_extraction > 30:
            return False
        
        return True

    def should_re_extract_coordinates(self):
        """좌표를 다시 추출해야 하는지 판단"""
        return not self.validate_coordinates()

    def detect_specific_cards(self, card_indices):
        """특정 인덱스의 카드만 인식"""
        if not self.validate_coordinates():
            print("좌표가 유효하지 않습니다. 좌표를 재추출합니다.")
            if not self.extract_card_coordinates():
                return None
        
        try:
            image = self.picam2.capture_array()
            detected_cards = ["Unknown"] * len(card_indices)
            
            # 필요한 카드만 처리
            warped_images = []
            valid_indices = []
            
            for i, idx in enumerate(card_indices):
                if idx < len(self.card_coordinates):
                    warped = self.four_point_transform(
                        image, 
                        self.card_coordinates[idx].reshape(4, 2), 
                        margin=40
                    )
                    warped_images.append(warped)
                    valid_indices.append(i)
            
            if not warped_images:
                return None
            
            # 멀티프로세싱으로 인식
            results = self.pool.starmap(
                process_card_worker,
                [(self.model_path, img) for img in warped_images]
            )
            
            # 결과를 올바른 위치에 저장
            for i, result in zip(valid_indices, results):
                detected_cards[i] = result
            
            # 첫 번째 카드 이미지 저장 (디버깅용)
            if warped_images:
                cv2.imwrite("assets/test_image/result.jpg", warped_images[0])
            
            return detected_cards
            
        except Exception as e:
            print(f"카드 인식 오류: {e}")
            return None

    def detect_player_cards(self, player_num):
        """특정 플레이어의 카드만 인식"""
        if f'player{player_num}' in self.card_positions:
            indices = self.card_positions[f'player{player_num}']
            return self.detect_specific_cards(indices)
        return None

    def detect_flop_cards(self):
        """플랍 카드 3장만 인식"""
        community_indices = self.card_positions['community'][:3]
        return self.detect_specific_cards(community_indices)

    def detect_turn_card(self):
        """턴 카드 1장만 인식"""
        turn_index = self.card_positions['community'][3]
        results = self.detect_specific_cards([turn_index])
        return results[0] if results else None

    def detect_river_card(self):
        """리버 카드 1장만 인식"""
        river_index = self.card_positions['community'][4]
        results = self.detect_specific_cards([river_index])
        return results[0] if results else None

    def detect_all_player_cards(self):
        """모든 플레이어의 카드를 인식"""
        all_player_indices = []
        for i in range(1, self.num_players + 1):
            if f'player{i}' in self.card_positions:
                all_player_indices.extend(self.card_positions[f'player{i}'])
        
        return self.detect_specific_cards(all_player_indices)

    def detect_cards(self):
        """기존 방식: 모든 카드를 한 번에 인식 (하위 호환성)"""
        if not self.validate_coordinates():
            if not self.extract_card_coordinates():
                return ["Unknown"] * self.total_cards
        
        all_indices = list(range(self.total_cards))
        return self.detect_specific_cards(all_indices)

    def detect_card_edges(self, image):
        MIN_CARD_AREA = 10000
        MIN_RATIO = 0.5
        MAX_RATIO = 1.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        cv2.imwrite("./grayimg.jpg", thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        card_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CARD_AREA:
                continue
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rect = self.order_points(approx.reshape(4, 2))
                w = np.linalg.norm(rect[0] - rect[1])
                h = np.linalg.norm(rect[0] - rect[3])
                ratio = w / h if h > 0 else 0
                if MIN_RATIO < ratio < MAX_RATIO:
                    card_contours.append(approx)
        return card_contours

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts, margin=20):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
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

    def sort_contours_reading_order(self, contours):
        # 각 윤곽선의 중심점 계산
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            centers.append((cy, cx, contour))
        
        # 행(row) 기준으로 먼저 정렬 (위에서 아래로)
        centers.sort(key=lambda x: x[0] // 60)  # 60픽셀 단위로 행 구분
        
        # 각 행 내에서 열(column) 기준으로 정렬 (왼쪽에서 오른쪽으로)
        sorted_contours = []
        current_row = centers[0][0] // 60
        row_contours = []
        
        for center in centers:
            if center[0] // 60 == current_row:
                row_contours.append(center)
            else:
                # 현재 행의 윤곽선들을 x좌표로 정렬
                row_contours.sort(key=lambda x: x[1])
                sorted_contours.extend([c[2] for c in row_contours])
                row_contours = [center]
                current_row = center[0] // 60
        
        # 마지막 행 처리
        if row_contours:
            row_contours.sort(key=lambda x: x[1])
            sorted_contours.extend([c[2] for c in row_contours])
        
        return sorted_contours

    def close(self):
        """리소스 정리"""
        self.picam2.stop()
        self.pool.close()
        self.pool.join()

# # 테스트 함수
# if __name__ == "__main__":
#     detector = CardDetector(num_players=3)
    
#     print("카드 좌표 추출 중...")
#     if detector.extract_card_coordinates():
#         print("좌표 추출 성공!")
        
#         print("플랍 카드 인식 중...")
#         flop_cards = detector.detect_flop_cards()
#         print(f"플랍 카드: {flop_cards}")
        
#         print("플레이어 1 카드 인식 중...")
#         player1_cards = detector.detect_player_cards(1)
#         print(f"플레이어 1 카드: {player1_cards}")
        
#     else:
#         print("좌표 추출 실패!")
    
#     detector.close()
