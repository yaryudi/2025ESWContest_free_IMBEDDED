import subprocess
import sys
import os
import time

def main():
    print("터치패드 시스템을 시작합니다...")
    
    # 개선된 캘리브레이션 실행 (드래그 지원)
    print("개선된 터치패드 캘리브레이션을 시작합니다...")
    calibration_process = None
    try:
        # Popen을 사용하여 개선된 캘리브레이션 GUI를 백그라운드에서 실행
        calibration_process = subprocess.Popen([sys.executable, 'calibration.py'])

        # 캘리브레이션 완료 신호 파일 대기
        signal_file = '.calibration_complete'
        print("캘리브레이션 완료를 기다리는 중...")
        while not os.path.exists(signal_file):
            if calibration_process.poll() is not None:
                print("캘리브레이션 프로세스가 예기치 않게 종료되었습니다.")
                return
            time.sleep(1)
        
        print("캘리브레이션이 성공적으로 완료되었습니다!")
        os.remove(signal_file) # 신호 파일 삭제

    except Exception as e:
        print(f"캘리브레이션 실행 중 오류 발생: {e}")
        if calibration_process:
            calibration_process.terminate()
        return

    # 게임 런처 실행
    print("게임 런처를 시작합니다...")
    try:
        # Popen을 사용하여 런처를 별도 프로세스로 실행
        subprocess.Popen([sys.executable, 'launcher.py'])
        print("게임 런처가 실행되었습니다.")
    except Exception as e:
        print(f"게임 런처 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 