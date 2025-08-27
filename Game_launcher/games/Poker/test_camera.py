#!/usr/bin/env python3
"""
카메라 테스트 스크립트
"""

import cv2
import time

def test_camera():
    print("카메라 테스트 시작...")
    
    # 기본 카메라 열기
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return False
    
    print(f"카메라 열기 성공")
    print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # 여러 번 프레임 읽기 시도
    for i in range(5):
        print(f"\n프레임 읽기 시도 {i+1}/5:")
        ret, frame = cap.read()
        print(f"  결과: {ret}")
        if ret and frame is not None:
            print(f"  프레임 크기: {frame.shape}")
            print(f"  프레임 타입: {frame.dtype}")
            break
        else:
            print(f"  프레임 읽기 실패")
            time.sleep(0.5)
    
    cap.release()
    print("\n카메라 테스트 완료")
    return ret

if __name__ == "__main__":
    test_camera()
