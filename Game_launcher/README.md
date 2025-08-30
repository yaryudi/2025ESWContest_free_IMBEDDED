# 🎯 드래그 지원 터치 시스템 개선

## 📋 개요

기존 터치 시스템의 드래그 인식 문제를 해결하기 위해 새로운 터치 매니저를 구현했습니다. 이제 터치 입력에서 드래그 동작을 정확하게 인식할 수 있습니다.

## 🔧 주요 개선사항

### 1. **터치 상태 관리**
- **터치 시작** (`TOUCH_START`)
- **터치 이동** (`TOUCH_MOVE`) 
- **터치 종료** (`TOUCH_END`)
- **드래그 시작** (`DRAG_START`)
- **드래그 이동** (`DRAG_MOVE`)
- **드래그 종료** (`DRAG_END`)

### 2. **노이즈 필터링**
- 터치 히스토리를 통한 노이즈 제거
- 최근 3개 포인트의 평균값 사용
- 안정적인 터치 좌표 추적


## 🚀 사용법

### 1. **기본 실행**
```bash
cd Game_launcher
python main.py
```

### 2. **개선된 캘리브레이션 직접 실행**
```bash
python improved_calibration_avg.py
```

### 3. **터치 매니저 테스트**
```python
from touch_manager import TouchManager, TouchState

# 터치 매니저 초기화
touch_mgr = TouchManager(
    drag_threshold=15,      # 드래그 시작 임계값
    touch_timeout=0.05,     # 터치 타임아웃
    min_drag_distance=8,    # 최소 드래그 거리
    max_touch_points=1      # 최대 터치 포인트 수
)

# 콜백 함수 설정
def on_drag_start(event):
    print(f"드래그 시작: {event.current_point.x}, {event.current_point.y}")

def on_drag_move(event):
    print(f"드래그 이동: 거리={event.drag_distance:.1f}")

def on_click(event):
    print(f"클릭: {event.current_point.x}, {event.current_point.y}")

touch_mgr.on_drag_start = on_drag_start
touch_mgr.on_drag_move = on_drag_move
touch_mgr.on_click = on_click
```

## ⚙️ 설정 매개변수

### TouchManager 설정
```python
TouchManager(
    drag_threshold=15,      # 드래그 시작 임계값 (센서 단위)
    touch_timeout=0.05,     # 터치 타임아웃 (초)
    min_drag_distance=8,    # 최소 드래그 거리 (센서 단위)
    max_touch_points=1      # 최대 터치 포인트 수
)
```

### 권장 설정값
- **높은 정밀도**: `drag_threshold=10, min_drag_distance=5`
- **중간 정밀도**: `drag_threshold=15, min_drag_distance=8` (기본값)
- **낮은 정밀도**: `drag_threshold=20, min_drag_distance=12`


---

**개선된 터치 시스템으로 더욱 직관적이고 반응성 좋은 게임 경험을 제공**
