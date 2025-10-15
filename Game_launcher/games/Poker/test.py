"""
카드 인식 테스트 모듈
Jetson Nano 웹캠으로  최신 프레임을 가져와서 YOLO 모델을 사용하여 포커 카드를 인식합니다.
좌표만 미리 추출하고 필요할 때만 카드 인식
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import multiprocessing as mp
from functools import partial
import signal
import sys
import subprocess
import os

def get_camera_capabilities(device_id=0):
    """카메라가 지원하는 해상도와 포맷을 확인"""
    try:
        device_path = f"/dev/video{device_id}"
        if not os.path.exists(device_path):
            print(f"카메라 디바이스 {device_path}가 존재하지 않습니다.")
            return None
        
        # v4l2-ctl 명령어가 있는지 확인
        try:
            result = subprocess.run(['which', 'v4l2-ctl'], capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                print("v4l2-ctl 명령어를 찾을 수 없습니다. 대체 방법을 사용합니다.")
                return None
        except:
            print("v4l2-ctl 명령어를 찾을 수 없습니다. 대체 방법을 사용합니다.")
            return None
        
        print(f"카메라 {device_path} 지원 해상도 확인 중...")
        result = subprocess.run(['v4l2-ctl', '--device', device_path, '--list-formats-ext'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("카메라 지원 해상도:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"해상도 확인 실패: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("카메라 해상도 확인 타임아웃")
        return None
    except Exception as e:
        print(f"카메라 해상도 확인 중 오류: {e}")
        return None


def diagnose_camera_state(cap, device_id):
    """카메라 상태를 상세히 진단"""
    try:
        print(f"🔍 카메라 {device_id} 상태 진단:")
        
        # 기본 속성들 확인
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        
        # FOURCC를 문자열로 변환
        fourcc_str = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"  현재 해상도: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  버퍼 크기: {buffer_size}")
        print(f"  코덱: {fourcc_str}")
        
        # 카메라가 실제로 프레임을 제공하는지 테스트 (여러 번 시도)
        print("  프레임 읽기 테스트...")
        
        # 버퍼 정리
        for _ in range(5):
            cap.grab()
        
        # 프레임 읽기 시도 (최대 5번)
        for frame_attempt in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"  프레임 읽기 성공: {frame.shape} (시도 {frame_attempt + 1})")
                return True
            else:
                print(f"  프레임 읽기 실패 (시도 {frame_attempt + 1})")
                if frame_attempt < 4:
                    time.sleep(0.1)
        
        print("  프레임 읽기 최종 실패")
        return False
            
    except Exception as e:
        print(f"  카메라 진단 중 오류: {e}")
        return False

def force_camera_resolution(cap, target_width, target_height, max_attempts=5):
    """카메라 해상도를 강제로 설정하는 함수"""
    print(f"🎯 해상도 {target_width}x{target_height} 강제 설정 시도...")
    
    for attempt in range(max_attempts):
        try:
            # 현재 해상도 확인
            current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  시도 {attempt + 1}: 현재 해상도 {current_width}x{current_height}")
            
            # 이미 목표 해상도라면 성공
            if abs(current_width - target_width) <= 5 and abs(current_height - target_height) <= 5:
                print(f"✅ 이미 목표 해상도 {target_width}x{target_height}로 설정됨!")
                return True
            
            # 해상도 설정 전에 카메라 상태 안정화
            if attempt > 0:
                print("  카메라 상태 안정화 중...")
                # 버퍼 정리
                for _ in range(5):
                    cap.grab()
                time.sleep(0.2)
            
            # 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            
            # 설정 적용 대기 (시도 횟수에 따라 대기 시간 증가)
            wait_time = 0.2 + (attempt * 0.1)
            time.sleep(wait_time)
            
            # 설정 확인
            new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"  설정 후 해상도: {new_width}x{new_height}")
            
            # 목표 해상도와 일치하는지 확인 (약간의 오차 허용)
            if abs(new_width - target_width) <= 5 and abs(new_height - target_height) <= 5:
                print(f"✅ 해상도 {target_width}x{target_height} 설정 성공!")
                return True
            else:
                print(f"❌ 해상도 설정 실패 (목표: {target_width}x{target_height}, 실제: {new_width}x{new_height})")
                
                # 추가적인 강제 설정 시도
                if attempt < max_attempts - 1:
                    print("  추가 설정 시도...")
                    # 더 강력한 버퍼 정리
                    for _ in range(10):
                        cap.grab()
                    time.sleep(0.2)
                    
                    # 다시 설정 (더 긴 대기 시간)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                    time.sleep(0.5)
                    
        except Exception as e:
            print(f"  해상도 설정 중 오류: {e}")
            time.sleep(0.2)
    
    print(f"⚠️ 해상도 {target_width}x{target_height} 설정 최종 실패")
    return False

def kill_camera_processes():
    """카메라를 사용하는 프로세스들을 종료"""
    try:
        print("카메라 사용 프로세스 확인 중...")
        
        # 카메라를 사용하는 프로세스 찾기
        result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            print("카메라를 사용하는 프로세스 발견:")
            print(result.stdout)
            
            # 프로세스 종료 시도
            lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        try:
                            print(f"프로세스 {pid} 종료 시도...")
                            subprocess.run(['kill', '-TERM', pid], timeout=3)
                            time.sleep(0.5)
                            # 강제 종료 시도
                            subprocess.run(['kill', '-KILL', pid], timeout=3)
                            print(f"프로세스 {pid} 종료 완료")
                        except:
                            print(f"프로세스 {pid} 종료 실패")
        else:
            print("카메라를 사용하는 프로세스 없음")
            
        time.sleep(1)  # 프로세스 종료 대기
        return True
        
    except Exception as e:
        print(f"프로세스 종료 중 오류: {e}")
        return False

def reset_camera_device(device_id=0):
    """시스템 레벨에서 카메라 디바이스를 리셋"""
    try:
        device_path = f"/dev/video{device_id}"
        if os.path.exists(device_path):
            print(f"카메라 디바이스 {device_path} 리셋 시도...")
            
            # 1단계: 카메라 사용 프로세스 종료
            kill_camera_processes()
            
            # v4l2-ctl 명령어가 있는지 확인
            try:
                result = subprocess.run(['which', 'v4l2-ctl'], capture_output=True, text=True, timeout=2)
                if result.returncode != 0:
                    print("v4l2-ctl 명령어를 찾을 수 없습니다. 리셋을 건너뜁니다.")
                    return True
            except:
                print("v4l2-ctl 명령어를 찾을 수 없습니다. 리셋을 건너뜁니다.")
                return True
            
            # 2단계: 여러 리셋 방법 시도 (방법 1 제거)
            reset_methods = [
                # 방법 1: 포맷 설정으로 리셋
                ['v4l2-ctl', '--device', device_path, '--set-fmt-video=width=640,height=480,pixelformat=MJPG'],
                # 방법 2: 더 간단한 포맷 설정
                ['v4l2-ctl', '--device', device_path, '--set-fmt-video=pixelformat=MJPG'],
                # 방법 3: 카메라 설정 초기화
                ['v4l2-ctl', '--device', device_path, '--set-ctrl=brightness=128'],
                # 방법 4: 강제 해상도 설정
                ['v4l2-ctl', '--device', device_path, '--set-fmt-video=width=1920,height=1080,pixelformat=MJPG'],
            ]
            
            for i, method in enumerate(reset_methods):
                try:
                    print(f"리셋 방법 {i+1} 시도...")
                    result = subprocess.run(method, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"리셋 방법 {i+1} 성공!")
                        time.sleep(0.5)
                        return True
                    else:
                        print(f"리셋 방법 {i+1} 실패: {result.stderr}")
                except Exception as e:
                    print(f"리셋 방법 {i+1} 실행 중 오류: {e}")
            
            print("모든 리셋 방법 실패")
            return False
        else:
            print(f"카메라 디바이스 {device_path}가 존재하지 않습니다.")
            return False
    except subprocess.TimeoutExpired:
        print("카메라 리셋 명령어 타임아웃")
        return False
    except Exception as e:
        print(f"카메라 리셋 중 오류: {e}")
        return False

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

class FrameCapture:
    def __init__(self, device_id=0, codec='MJPG'):
        self.device_id = device_id
        self.codec = codec  # 'MJPG' 또는 'YUYV'
        self.cap = None
        self._initialize_camera()
    
    def _force_close_camera(self):
        """카메라를 강제로 닫고 리소스 정리"""
        if self.cap:
            try:
                print(f"카메라 {self.device_id} 강제 해제 시작...")
                
                if self.cap.isOpened():
                    # 1단계: 버퍼 정리
                    print("  버퍼 정리 중...")
                    for i in range(10):
                        try:
                            self.cap.grab()
                        except:
                            break
                    
                    # 2단계: 카메라 설정 초기화
                    try:
                        print("  카메라 설정 초기화 중...")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        pass
                    
                    # 3단계: 카메라 해제
                    print("  카메라 해제 중...")
                    self.cap.release()
                
                # 4단계: 객체 정리
                self.cap = None
                
                # 5단계: 시스템 리소스 해제 대기
                time.sleep(0.5)
                
                print(f"카메라 {self.device_id} 강제 해제 완료")
                
            except Exception as e:
                print(f"카메라 강제 닫기 중 오류: {e}")
                self.cap = None
    
    def _initialize_camera(self):
        """카메라 초기화 - V4L2 백엔드 사용"""
        max_init_attempts = 3
        
        for init_attempt in range(max_init_attempts):
            try:
                print(f"카메라 초기화 시작... (시도 {init_attempt + 1}/{max_init_attempts})")
                
                # 이전 시도에서 카메라가 열려있다면 닫기
                if init_attempt > 0:
                    self._force_close_camera()
                    time.sleep(0.5)  # 리소스 해제 대기
                
                # 카메라 디바이스 리셋 시도 (선택적)
                reset_camera_device(self.device_id)
                
                # 첫 번째 시도에서만 카메라 지원 해상도 확인 (디버깅용)
                if init_attempt == 0:
                    get_camera_capabilities(self.device_id)
                
                # V4L2 백엔드로만 카메라 열기 (MJPG 코덱 우선)
                camera_opened = False
                
                # 방법 1: V4L2 백엔드로 인덱스 사용
                print("카메라 열기 시도 1: V4L2 백엔드 (인덱스)")
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
                if self.cap.isOpened():
                    camera_opened = True
                    print("✅ 카메라 열기 성공 (방법 1)")
                
                if not camera_opened:
                    # 방법 2: V4L2 백엔드로 경로 사용
                    print("카메라 열기 시도 2: V4L2 백엔드 (경로)")
                    device_path = f"/dev/video{self.device_id}"
                    self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                    if self.cap.isOpened():
                        camera_opened = True
                        print("✅ 카메라 열기 성공 (방법 2)")
                
                if not camera_opened:
                    if init_attempt < max_init_attempts - 1:
                        print(f"카메라 연결 실패, 재시도 중... ({init_attempt + 1}/{max_init_attempts})")
                        continue
                    else:
                        raise RuntimeError(f"웹캠 {self.device_id} 연결 실패")
                
                # 카메라가 완전히 열릴 때까지 대기
                time.sleep(0.3)
                
                # MJPG 코덱 강제 설정 (YUYV 완전 차단)
                print("🎨 MJPG 코덱 강제 설정 중...")
                
                mjpg_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                codec_set_success = False
                
                for codec_attempt in range(15):
                    # MJPG 코덱 설정
                    self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
                    time.sleep(0.3)
                    
                    # 설정된 코덱 확인
                    current_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
                    current_fourcc_str = "".join([chr((int(current_fourcc) >> 8 * i) & 0xFF) for i in range(4)])
                    
                    if current_fourcc_str == 'MJPG':
                        print(f"✅ MJPG 코덱 설정 성공! (시도 {codec_attempt + 1})")
                        codec_set_success = True
                        break
                    elif current_fourcc_str == 'YUYV':
                        print(f"❌ YUYV 코덱 감지! MJPG 재설정 중... (시도 {codec_attempt + 1})")
                        # YUYV가 감지되면 즉시 MJPG로 재설정
                        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
                        time.sleep(0.5)
                    else:
                        if codec_attempt < 5:
                            print(f"❌ MJPG 코덱 설정 실패, 재시도... ({current_fourcc_str})")
                        elif codec_attempt % 3 == 0:
                            print(f"❌ MJPG 코덱 설정 실패 (시도 {codec_attempt + 1}): {current_fourcc_str}")
                        time.sleep(0.4)
                
                if not codec_set_success:
                    print("⚠️ MJPG 코덱 설정 실패, 현재 코덱으로 진행")
                
                # 코덱 설정 후 충분한 대기
                time.sleep(0.5)
                
                # 버퍼 크기 최소화 설정
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 스마트 해상도 설정 (4K 우선, 실패 시 점진적 감소)
                print("🚀 스마트 해상도 설정 시작...")
                
                resolutions_to_try = [
                    (3840, 2160, "4K"),
                    (1920, 1080, "Full HD"),
                    (1280, 720, "HD"),
                    (640, 480, "기본")
                ]
                
                resolution_success = False
                actual_width = 640
                actual_height = 480
                
                for width, height, res_name in resolutions_to_try:
                    print(f"🎯 {res_name} 해상도({width}x{height}) 설정 시도...")
                    
                    if force_camera_resolution(self.cap, width, height, max_attempts=5):
                        actual_width = width
                        actual_height = height
                        print(f"🎉 {res_name} 해상도 설정 완료!")
                        resolution_success = True
                        break
                    else:
                        print(f"❌ {res_name} 해상도 설정 실패")
                
                if not resolution_success:
                    print("⚠️ 모든 해상도 설정 실패, 현재 해상도로 진행")
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                print(f"카메라 {self.device_id} 연결 성공: {actual_width}x{actual_height}")
                
                # 4K 해상도만 집중해서 설정
                print("🎯 4K 해상도 집중 설정 중...")
                
                # 4K 해상도 강제 설정
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                time.sleep(0.5)
                
                # MJPG 코덱 재강제 설정 (YUYV 완전 차단)
                print("🔧 MJPG 코덱 재강제 설정 중...")
                
                for recheck_attempt in range(10):
                    # MJPG 코덱 설정
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    time.sleep(0.4)
                    
                    # 설정된 코덱 확인
                    current_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
                    current_fourcc_str = "".join([chr((int(current_fourcc) >> 8 * i) & 0xFF) for i in range(4)])
                    
                    if current_fourcc_str == 'MJPG':
                        print(f"✅ MJPG 코덱 재설정 성공! (시도 {recheck_attempt + 1})")
                        break
                    elif current_fourcc_str == 'YUYV':
                        print(f"❌ YUYV 코덱 감지! MJPG 재설정 중... (시도 {recheck_attempt + 1})")
                        # YUYV가 감지되면 즉시 MJPG로 재설정
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                        time.sleep(0.6)
                    else:
                        if recheck_attempt < 3:
                            print(f"❌ MJPG 코덱 재설정 실패, 재시도... ({current_fourcc_str})")
                        else:
                            print(f"❌ MJPG 코덱 재설정 실패 (시도 {recheck_attempt + 1}): {current_fourcc_str}")
                        time.sleep(0.6)
                
                # 최종 설정 확인
                final_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                final_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                final_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
                final_fourcc_str = "".join([chr((int(final_fourcc) >> 8 * i) & 0xFF) for i in range(4)])
                
                print(f"✅ 최종 설정: {final_width}x{final_height} ({final_fourcc_str})")
                
                # 4K 해상도가 아니면 재초기화 시도
                if int(final_width) != 3840 or int(final_height) != 2160:
                    print("⚠️ 4K 해상도 설정 실패, 카메라 재초기화 시도...")
                    raise RuntimeError("4K 해상도 설정 실패")
                
                # 카메라 상태 진단
                diagnose_camera_state(self.cap, self.device_id)
                
                # 카메라 초기화를 위한 추가 대기
                time.sleep(0.5)
                
                # 카메라가 실제로 프레임을 제공할 준비가 될 때까지 대기
                print("📸 프레임 읽기 준비 테스트...")
                
                # 카메라 스트림 시작 강제화 (여러 방법 시도)
                print("📸 카메라 스트림 시작 강제화...")
                
                # 방법 1: 강력한 버퍼 정리
                print("  방법 1: 강력한 버퍼 정리 시도...")
                for _ in range(10):
                    self.cap.grab()
                
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ 방법 1 성공! 프레임 크기: {frame.shape}")
                    if frame.shape[1] == 3840 and frame.shape[0] == 2160:
                        print("✅ 4K 프레임 읽기 준비 완료")
                        return
                
                # 방법 2: 카메라 재설정 후 시도
                print("  방법 2: 카메라 재설정 후 시도...")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                time.sleep(1.0)
                
                for _ in range(5):
                    self.cap.grab()
                
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"✅ 방법 2 성공! 프레임 크기: {frame.shape}")
                    if frame.shape[1] == 3840 and frame.shape[0] == 2160:
                        print("✅ 4K 프레임 읽기 준비 완료")
                        return
                
                # 방법 3: 카메라 재연결 시도
                print("  방법 3: 카메라 재연결 시도...")
                self.cap.release()
                time.sleep(0.5)
                
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                    time.sleep(1.0)
                    
                    for _ in range(5):
                        self.cap.grab()
                    
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ 방법 3 성공! 프레임 크기: {frame.shape}")
                        if frame.shape[1] == 3840 and frame.shape[0] == 2160:
                            print("✅ 4K 프레임 읽기 준비 완료")
                            return
                
                print("❌ 모든 스트림 시작 방법 실패")
                print("⚠️ 프레임 읽기 실패, 카메라 재초기화 시도...")
                raise RuntimeError("프레임 읽기 실패")
                
                print("카메라 초기화 완료")
                return  # 성공적으로 초기화 완료
                
            except Exception as e:
                print(f"카메라 초기화 시도 {init_attempt + 1} 실패: {e}")
                if init_attempt < max_init_attempts - 1:
                    print("다음 시도를 위해 잠시 대기...")
                    time.sleep(1)
                    continue
                else:
                    print("모든 초기화 시도 실패")
                    raise

    def read(self):
        """프레임 읽기 - 안전하고 안정적인 방식으로 개선"""
        if not self.cap or not self.cap.isOpened():
            print("카메라가 열려있지 않습니다.")
            return (False, None)
        
        try:
            # 카메라가 실제로 준비되었는지 확인
            if not self._wait_for_camera_ready():
                print("카메라가 준비되지 않았습니다.")
                return (False, None)
            
            # 최신 프레임 읽기 (버퍼 비우기 없이)
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                return (False, None)
            
            # 프레임 유효성 검사
            if frame.size == 0:
                print("빈 프레임을 받았습니다.")
                return (False, None)
            
            return (True, frame)
            
        except Exception as e:
            print(f"프레임 읽기 중 오류: {e}")
            return (False, None)
    
    def _wait_for_camera_ready(self, max_attempts=10, delay=0.1):
        """카메라가 프레임을 제공할 준비가 될 때까지 대기"""
        for attempt in range(max_attempts):
            try:
                # 프레임 읽기 시도 (실제로는 저장하지 않음)
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    return True
                time.sleep(delay)
            except Exception as e:
                print(f"카메라 준비 확인 중 오류 (시도 {attempt + 1}): {e}")
                time.sleep(delay)
        
        print(f"카메라 준비 대기 시간 초과 ({max_attempts * delay}초)")
        return False
    
    def diagnose_camera(self):
        """카메라 상태 진단"""
        print(f"=== 카메라 {self.device_id} 진단 ===")
        
        if not self.cap:
            print("카메라 객체가 None입니다.")
            return False
        
        if not self.cap.isOpened():
            print("카메라가 열려있지 않습니다.")
            return False
        
        try:
            # 카메라 속성 확인
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
            
            # FOURCC를 문자열로 변환
            fourcc_str = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            print(f"해상도: {width}x{height}")
            print(f"FPS: {fps}")
            print(f"버퍼 크기: {buffer_size}")
            print(f"코덱: {fourcc_str}")
            
            # 프레임 읽기 테스트
            print("프레임 읽기 테스트 중...")
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"프레임 읽기 성공: {frame.shape}")
                return True
            else:
                print("프레임 읽기 실패")
                return False
                
        except Exception as e:
            print(f"카메라 진단 중 오류: {e}")
            return False

    def release(self):
        """카메라 해제 - 버퍼 정리 후 해제"""
        if self.cap and self.cap.isOpened():
            try:
                print(f"카메라 {self.device_id} 해제 중...")
                
                # 1단계: 남아있는 프레임 버퍼들을 모두 비우기
                print("  프레임 버퍼 정리 중...")
                buffer_count = 0
                max_buffer_clear = 20  # 최대 20개 프레임까지 버퍼에서 제거
                
                for i in range(max_buffer_clear):
                    try:
                        # grab()으로 프레임을 메모리에 저장하지 않고 버퍼만 비움
                        if self.cap.grab():
                            buffer_count += 1
                        else:
                            # 더 이상 읽을 프레임이 없으면 중단
                            break
                    except Exception as e:
                        print(f"    버퍼 정리 중 오류 (프레임 {i+1}): {e}")
                        break
                
                print(f"  {buffer_count}개 프레임 버퍼 정리 완료")
                
                # 2단계: 카메라 설정을 원래대로 복원
                try:
                    # 해상도를 기본값으로 리셋
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_SETTINGS, 0)
                    print("  카메라 설정 복원 완료")
                except Exception as e:
                    print(f"  카메라 설정 복원 실패: {e}")
                
                # 3단계: 카메라 해제
                self.cap.release()
                
                # 4단계: 카메라 객체를 None으로 설정
                self.cap = None
                
                # 5단계: 시스템 리소스 정리를 위한 짧은 대기
                time.sleep(0.1)
                
                print(f"카메라 {self.device_id} 연결 해제 완료")
                
            except Exception as e:
                print(f"카메라 해제 중 오류: {e}")
                # 오류 발생 시에도 강제로 None 설정
                self.cap = None
        else:
            print(f"카메라 {self.device_id}는 이미 해제되었습니다.")
    
    def force_release(self):
        """강제로 카메라 해제 (상태와 관계없이) - 버퍼 정리 포함"""
        try:
            if self.cap:
                print(f"카메라 {self.device_id} 강제 해제 중...")
                
                # 버퍼 정리 시도
                try:
                    if self.cap.isOpened():
                        print("  프레임 버퍼 강제 정리 중...")
                        for i in range(10):  # 최대 10개 프레임
                            try:
                                self.cap.grab()
                            except:
                                break
                        print("  버퍼 정리 완료")
                except Exception as e:
                    print(f"  버퍼 정리 중 오류: {e}")
                
                # 카메라 해제
                self.cap.release()
                self.cap = None
                print(f"카메라 {self.device_id} 강제 해제 완료")
        except Exception as e:
            print(f"카메라 강제 해제 중 오류: {e}")
            self.cap = None
    
    def is_ready(self):
        """카메라가 사용 가능한 상태인지 확인"""
        return self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        """소멸자에서도 카메라 해제 보장"""
        try:
            if hasattr(self, 'cap') and self.cap:
                print(f"FrameCapture 소멸자에서 카메라 {self.device_id} 해제")
                self.force_release()
        except Exception as e:
            print(f"소멸자에서 카메라 해제 중 오류: {e}")
            # 오류 발생 시에도 강제로 None 설정
            if hasattr(self, 'cap'):
                self.cap = None

class CardDetector:
    def __init__(self, num_players=5, device_id=0, codec='MJPG'):
        
        self.cap = FrameCapture(device_id, codec)
        
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
            # 카메라 상태 확인
            if not self.cap.is_ready():
                print("카메라가 준비되지 않았습니다.")
                return False
            
            # 카메라 진단 실행
            print("카메라 상태 진단 중...")
            if not self.cap.diagnose_camera():
                print("카메라 진단 실패")
                return False
            
            ret, image = self.cap.read()
            if not ret:
                print("카메라에서 이미지를 읽을 수 없습니다.")
                return False
            
            # 이미지 저장 (디버깅용)
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
            # 카메라 상태 확인
            if not self.cap.is_ready():
                print("카메라가 준비되지 않았습니다.")
                return None
            
            ret, image = self.cap.read()
            if not ret:
                print("카메라에서 이미지를 읽을 수 없습니다.")
                return None
            
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
            # if warped_images:
            #     cv2.imwrite("assets/test_image/result.jpg", warped_images[0])
            
            return detected_cards
            
        except Exception as e:
            print(f"카드 인식 오류: {e}")
            return None

    def detect_player_cards(self, player_num):
        """특정 플레이어의 카드만 인식"""
        # 플레이어 번호에 따라 실제 카드 위치 계산
        if player_num == 1:
            player_indices = [0, 1]
        elif player_num == 2:
            player_indices = [2, 3]
        elif player_num == 3:
            player_indices = [4, 5]
        elif player_num == 4:
            # 플레이어 4의 위치는 플레이어 수에 따라 다름
            if self.num_players == 4:
                player_indices = [11, 12]  # 커뮤니티 카드(5장) 이후
            elif self.num_players == 5:
                player_indices = [13, 14]  # 커뮤니티 카드(5장) + P5(2장) 이후
        elif player_num == 5:
            if self.num_players == 5:
                player_indices = [11, 12]  # 커뮤니티 카드(5장) 이후
        else:
            return None
        
        return self.detect_specific_cards(player_indices)

    def detect_flop_cards(self):
        """플랍 카드 3장만 인식"""
        # 플레이어 수에 따라 커뮤니티 카드의 실제 시작 위치 계산
        if self.num_players == 2:
            community_start = 4
        else:
            community_start = 6
        
        # 플랍 카드 3장의 인덱스
        flop_indices = [community_start, community_start + 1, community_start + 2]
        return self.detect_specific_cards(flop_indices)

    def detect_turn_card(self):
        """턴 카드 1장만 인식"""
        # 플레이어 수에 따라 커뮤니티 카드의 실제 시작 위치 계산
        if self.num_players == 2:
            community_start = 4
        else:
            community_start = 6
        
        # 턴 카드 인덱스 (커뮤니티 카드 4번째)
        turn_index = community_start + 3
        results = self.detect_specific_cards([turn_index])
        return results[0] if results else None

    def detect_river_card(self):
        """리버 카드 1장만 인식"""
        # 플레이어 수에 따라 커뮤니티 카드의 실제 시작 위치 계산
        if self.num_players == 2:
            community_start = 4
        else:
            community_start = 6
        
        # 리버 카드 인덱스 (커뮤니티 카드 5번째)
        river_index = community_start + 4
        results = self.detect_specific_cards([river_index])
        return results[0] if results else None

    def detect_all_player_cards(self):
        """모든 플레이어의 카드를 인식"""
        all_player_indices = []
        
        # 플레이어 1, 2, 3은 항상 같은 위치
        all_player_indices.extend([0, 1])  # P1
        all_player_indices.extend([2, 3])  # P2
        all_player_indices.extend([4, 5])  # P3
        
        # 플레이어 4, 5는 플레이어 수에 따라 위치가 다름
        if self.num_players >= 4:
            if self.num_players == 4:
                all_player_indices.extend([11, 12])  # P4
            elif self.num_players == 5:
                all_player_indices.extend([11, 12])  # P5
                all_player_indices.extend([13, 14])  # P4
        
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
        
        # 노이즈 제거를 위한 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 대비 향상을 위한 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # 다중 임계값 처리로 흰색 영역을 더 엄격하게 구분
        # 1. Otsu 알고리즘으로 기본 임계값 계산
        otsu_threshold, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu 임계값: {otsu_threshold}")
        
        # 2. 더 엄격한 임계값 적용 (Otsu 임계값보다 높게)
        #strict_threshold = min(otsu_threshold -20, 240)  # Otsu + 20, 최대 240
        _, thresh_strict = cv2.threshold(enhanced, 0.5*otsu_threshold, otsu_threshold, cv2.THRESH_BINARY)
        #print(f"엄격한 임계값: {strict_threshold}")
        
        # 3. 적응형 임계값도 시도
        thresh_adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 4. 세 가지 결과를 조합 (AND 연산으로 더 엄격하게)
        thresh = cv2.bitwise_and(thresh_otsu, thresh_strict)
        thresh = cv2.bitwise_and(thresh, thresh_adaptive)
        
        # 모폴로지 연산으로 노이즈 제거 및 흰색 영역 정리
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 작은 구멍 메우기
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # 작은 노이즈 제거
        
        # 디버깅을 위한 중간 결과 저장
        cv2.imwrite("./assets/test_image/enhanced.jpg", enhanced)
        cv2.imwrite("./assets/test_image/thresh_otsu.jpg", thresh_otsu)
        cv2.imwrite("./assets/test_image/thresh_strict.jpg", thresh_strict)
        cv2.imwrite("./assets/test_image/thresh_adaptive.jpg", thresh_adaptive)
        cv2.imwrite("./assets/test_image/grayimg.jpg", thresh)
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
        centers.sort(key=lambda x: x[0] // 100)  # 60픽셀 단위로 행 구분
        
        # 각 행 내에서 열(column) 기준으로 정렬 (왼쪽에서 오른쪽으로)
        sorted_contours = []
        current_row = centers[0][0] // 100
        row_contours = []
        
        for center in centers:
            if center[0] // 100 == current_row:
                print(center[0])
                row_contours.append(center)
            else:
                # 현재 행의 윤곽선들을 x좌표로 정렬
                row_contours.sort(key=lambda x: x[1])
                sorted_contours.extend([c[2] for c in row_contours])
                row_contours = [center]
                current_row = center[0] // 100
        
        # 마지막 행 처리
        if row_contours:
            row_contours.sort(key=lambda x: x[1])
            sorted_contours.extend([c[2] for c in row_contours])
        
        return sorted_contours

    def close(self):
        """리소스 정리 (간단한 버전)"""
        try:
            print("CardDetector 리소스 정리 중...")
            
            # 프로세스 풀 정리
            if hasattr(self, 'pool') and self.pool:
                print("프로세스 풀 정리 중...")
                try:
                    self.pool.terminate()
                    self.pool.join()
                except Exception as e:
                    print(f"프로세스 풀 정리 중 오류: {e}")
                finally:
                    self.pool = None
            
            # 카메라는 프로그램 종료 시 자동으로 해제되므로 간단히 처리
            if hasattr(self, 'cap') and self.cap:
                print("카메라 연결 해제 중...")
                try:
                    self.cap.release()
                except Exception as e:
                    print(f"카메라 해제 중 오류: {e}")
                finally:
                    self.cap = None
            
            print("리소스 정리 완료")
            
        except Exception as e:
            print(f"리소스 정리 중 오류: {e}")
        finally:
            # 강제로 None 설정
            self.pool = None
            self.cap = None
            
            # 가비지 컬렉션 강제 실행
            import gc
            gc.collect()
    
    def __del__(self):
        """소멸자에서도 리소스 정리 보장"""
        try:
            print("CardDetector 소멸자에서 리소스 정리")
            self.close()
        except Exception as e:
            print(f"소멸자에서 리소스 정리 중 오류: {e}")
            # 오류 발생 시에도 강제로 None 설정
            if hasattr(self, 'cap'):
                self.cap = None
            if hasattr(self, 'pool'):
                self.pool = None

# 전역 변수로 detector 저장 (시그널 핸들러에서 접근)
global_detector = None

def signal_handler(signum, frame):
    """시그널 핸들러 - 프로그램 강제 종료 시 카메라 해제"""
    print(f"\n시그널 {signum}을 받았습니다. 카메라를 안전하게 해제합니다.")
    
    if global_detector:
        try:
            print("카메라 연결 해제 중...")
            global_detector.cap.force_release()
            print("카메라 연결 해제 완료")
        except Exception as e:
            print(f"카메라 해제 중 오류: {e}")
    
    print("프로그램을 종료합니다.")
    sys.exit(0)

def cleanup_resources():
    """리소스 정리 함수"""
    global global_detector
    if global_detector:
        try:
            print("프로그램 종료 시 리소스 정리 중...")
            global_detector.close()
            print("리소스 정리 완료")
        except Exception as e:
            print(f"리소스 정리 중 오류: {e}")
        finally:
            global_detector = None

# 프로그램 종료 시 자동으로 리소스 정리
import atexit
atexit.register(cleanup_resources)

def main():
    """메인 실행 함수 - 버튼 입력으로 사진 캡처"""
    print("포커 카드 인식 시스템 시작")
    print("사용법:")
    print("  'c' - 카드 좌표 추출")
    print("  '1' - 플레이어 1 카드 인식")
    print("  '2' - 플레이어 2 카드 인식")
    print("  '3' - 플레이어 3 카드 인식")
    print("  '4' - 플레이어 4 카드 인식")
    print("  '5' - 플레이어 5 카드 인식")
    print("  'f' - 플랍 카드 인식")
    print("  't' - 턴 카드 인식")
    print("  'r' - 리버 카드 인식")
    print("  'a' - 모든 플레이어 카드 인식")
    print("  'd' - 카메라 진단")
    print("  'q' - 종료")
    
    # 플레이어 수 입력 받기
    while True:
        try:
            num_players = int(input("플레이어 수를 입력하세요 (2-5): "))
            if 2 <= num_players <= 5:
                break
            else:
                print("플레이어 수는 2-5 사이여야 합니다.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")
    
    # 카드 디텍터 초기화
    global global_detector
    try:
        global_detector = CardDetector(num_players=num_players)
        
        # 시그널 핸들러 설정 (프로그램 강제 종료 시 카메라 해제)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while True:
            command = input("\n명령을 입력하세요: ").lower().strip()
            
            if command == 'q':
                print("프로그램을 종료합니다.")
                break
            elif command == 'd':
                print("카메라 진단을 실행합니다...")
                if global_detector.cap.diagnose_camera():
                    print("카메라 진단 성공")
                else:
                    print("카메라 진단 실패")
            elif command == 'c':
                print("카드 좌표를 추출합니다...")
                if global_detector.extract_card_coordinates():
                    print("카드 좌표 추출 성공!")
                else:
                    print("카드 좌표 추출 실패!")
            elif command == '1':
                print("플레이어 1 카드를 인식합니다...")
                cards = global_detector.detect_player_cards(1)
                if cards:
                    print(f"플레이어 1 카드: {cards}")
                else:
                    print("카드 인식 실패!")
            elif command == '2':
                print("플레이어 2 카드를 인식합니다...")
                cards = global_detector.detect_player_cards(2)
                if cards:
                    print(f"플레이어 2 카드: {cards}")
                else:
                    print("카드 인식 실패!")
            elif command == '3':
                print("플레이어 3 카드를 인식합니다...")
                cards = global_detector.detect_player_cards(3)
                if cards:
                    print(f"플레이어 3 카드: {cards}")
                else:
                    print("카드 인식 실패!")
            elif command == '4':
                if num_players >= 4:
                    print("플레이어 4 카드를 인식합니다...")
                    cards = global_detector.detect_player_cards(4)
                    if cards:
                        print(f"플레이어 4 카드: {cards}")
                    else:
                        print("카드 인식 실패!")
                else:
                    print("플레이어 4는 존재하지 않습니다.")
            elif command == '5':
                if num_players == 5:
                    print("플레이어 5 카드를 인식합니다...")
                    cards = global_detector.detect_player_cards(5)
                    if cards:
                        print(f"플레이어 5 카드: {cards}")
                    else:
                        print("카드 인식 실패!")
                else:
                    print("플레이어 5는 존재하지 않습니다.")
            elif command == 'f':
                print("플랍 카드를 인식합니다...")
                cards = global_detector.detect_flop_cards()
                if cards:
                    print(f"플랍 카드: {cards}")
                else:
                    print("카드 인식 실패!")
            elif command == 't':
                print("턴 카드를 인식합니다...")
                card = global_detector.detect_turn_card()
                if card:
                    print(f"턴 카드: {card}")
                else:
                    print("카드 인식 실패!")
            elif command == 'r':
                print("리버 카드를 인식합니다...")
                card = global_detector.detect_river_card()
                if card:
                    print(f"리버 카드: {card}")
                else:
                    print("카드 인식 실패!")
            elif command == 'a':
                print("모든 플레이어 카드를 인식합니다...")
                cards = global_detector.detect_all_player_cards()
                if cards:
                    print(f"모든 플레이어 카드: {cards}")
                else:
                    print("카드 인식 실패!")
            else:
                print("알 수 없는 명령입니다. 다시 시도하세요.")
    
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
    finally:
        # 간단한 정리 (프로그램 종료 시 자동으로 처리됨)
        if global_detector:
            print("프로그램을 종료합니다.")
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()