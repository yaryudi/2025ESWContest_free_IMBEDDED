import subprocess
import sys
import os

def main():
    print("터치패드 시스템을 시작합니다...")
    
    # 캘리브레이션 실행
    print("터치패드 캘리브레이션을 시작합니다...")
    try:
        calibration_process = subprocess.run([sys.executable, 'calibration_gui.py'], 
                                           capture_output=True, text=True)
        
        if calibration_process.returncode == 0:
            print("캘리브레이션이 성공적으로 완료되었습니다!")
        else:
            print("캘리브레이션에 실패했습니다.")
            print("오류:", calibration_process.stderr)
            return
            
    except Exception as e:
        print(f"캘리브레이션 실행 중 오류 발생: {e}")
        return

if __name__ == "__main__":
    main() 