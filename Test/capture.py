#!/usr/bin/env python3
import time
import cv2
from picamera2 import Picamera2

def capture_and_save(filename: str) -> None:
    """
    Pi 카메라로 이미지를 캡처해 지정한 파일명으로 저장합니다.
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration({
        "format": "XRGB8888",
        "size": (1280, 720)
    })
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # 버퍼 안정화
    frame = picam2.capture_array()
    picam2.stop()

    # RGB → BGR 변환 후 저장
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
    print(f"Captured image saved to: {filename}")

if __name__ == "__main__":
    ts = int(time.time())
    fname = f"capture_{ts}.png"
    capture_and_save(fname)

capture_and_save() 에 파일명을 넘기면 그 이름으로 현재 디렉토리에 PNG가 저장됩니다.

time.sleep(0.5) 는 카메라 버퍼 안정화를 위해 넣은 것이고, 필요 없으면 제거해도 됩니다.


