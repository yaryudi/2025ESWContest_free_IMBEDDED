import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

class TouchState(Enum):
    """터치 상태 열거형"""
    NONE = "none"
    TOUCH_START = "touch_start"
    TOUCH_MOVE = "touch_move"
    TOUCH_END = "touch_end"
    DRAG_START = "drag_start"
    DRAG_MOVE = "drag_move"
    DRAG_END = "drag_end"

@dataclass
class TouchPoint:
    """터치 포인트 데이터 클래스"""
    x: int
    y: int
    pressure: int
    timestamp: float

@dataclass
class TouchEvent:
    """터치 이벤트 데이터 클래스"""
    state: TouchState
    current_point: TouchPoint
    start_point: Optional[TouchPoint] = None
    previous_point: Optional[TouchPoint] = None
    drag_distance: float = 0.0
    duration: float = 0.0

class TouchManager:
    """고급 터치 입력 관리자 - 드래그 인식 지원"""
    
    def __init__(self, 
                 drag_threshold: int = 10,      # 드래그 시작 임계값 (픽셀)
                 touch_timeout: float = 0.1,    # 터치 타임아웃 (초)
                 min_drag_distance: int = 5,    # 최소 드래그 거리
                 max_touch_points: int = 1):    # 최대 터치 포인트 수
        
        self.drag_threshold = drag_threshold
        self.touch_timeout = touch_timeout
        self.min_drag_distance = min_drag_distance
        self.max_touch_points = max_touch_points
        
        # 터치 상태 관리
        self.current_state = TouchState.NONE
        self.touch_start_time = 0.0
        self.touch_start_point: Optional[TouchPoint] = None
        self.current_point: Optional[TouchPoint] = None
        self.previous_point: Optional[TouchPoint] = None
        
        # 콜백 함수들
        self.on_touch_start: Optional[Callable[[TouchEvent], None]] = None
        self.on_touch_move: Optional[Callable[[TouchEvent], None]] = None
        self.on_touch_end: Optional[Callable[[TouchEvent], None]] = None
        self.on_drag_start: Optional[Callable[[TouchEvent], None]] = None
        self.on_drag_move: Optional[Callable[[TouchEvent], None]] = None
        self.on_drag_end: Optional[Callable[[TouchEvent], None]] = None
        self.on_click: Optional[Callable[[TouchEvent], None]] = None
        
        # 터치 히스토리 (노이즈 필터링용)
        self.touch_history: List[TouchPoint] = []
        self.max_history_size = 5
        
    def process_frame(self, frame: np.ndarray, threshold: int = 20) -> Optional[TouchEvent]:
        """
        프레임 데이터를 처리하여 터치 이벤트 생성
        
        Args:
            frame: 터치 센서 프레임 데이터
            threshold: 터치 감지 임계값
            
        Returns:
            TouchEvent 또는 None (터치가 없을 경우)
        """
        # 터치 포인트 감지
        touch_points = self._detect_touch_points(frame, threshold)
        
        if not touch_points:
            # 터치가 없으면 종료 처리
            if self.current_state != TouchState.NONE:
                return self._handle_touch_end()
            return None
        
        # 가장 강한 터치 포인트 선택
        primary_touch = max(touch_points, key=lambda p: p.pressure)
        
        # 터치 히스토리에 추가
        self._add_to_history(primary_touch)
        
        # 노이즈 필터링된 터치 포인트 계산
        filtered_point = self._get_filtered_touch_point()
        
        # 터치 상태 업데이트
        return self._update_touch_state(filtered_point)
    
    def _detect_touch_points(self, frame: np.ndarray, threshold: int) -> List[TouchPoint]:
        """프레임에서 터치 포인트들을 감지"""
        touch_points = []
        
        # 행/열 최대값 교집합으로 노이즈 제거
        row_max = frame.max(axis=1, keepdims=True)
        col_max = frame.max(axis=0, keepdims=True)
        mask = (frame == row_max) & (frame == col_max)
        filtered_frame = frame * mask
        
        # 임계값 이상인 모든 포인트 찾기
        high_pressure_points = np.where(filtered_frame >= threshold)
        
        for i in range(len(high_pressure_points[0])):
            y, x = high_pressure_points[0][i], high_pressure_points[1][i]
            pressure = filtered_frame[y, x]
            
            touch_point = TouchPoint(
                x=int(x), 
                y=int(y), 
                pressure=int(pressure),
                timestamp=time.time()
            )
            touch_points.append(touch_point)
        
        return touch_points
    
    def _add_to_history(self, touch_point: TouchPoint):
        """터치 히스토리에 포인트 추가"""
        self.touch_history.append(touch_point)
        if len(self.touch_history) > self.max_history_size:
            self.touch_history.pop(0)
    
    def _get_filtered_touch_point(self) -> TouchPoint:
        """히스토리를 기반으로 노이즈 필터링된 터치 포인트 반환"""
        if not self.touch_history:
            return None
            
        # 최근 포인트들의 평균 계산 (극한 반응 속도)
        recent_points = self.touch_history[-2:]  # 최근 2개 포인트로 극한 줄임 (극한 반응 속도)
        
        # 압력이 임계값 이상인 포인트만 필터링 (극한 민감하게 설정)
        high_pressure_points = [p for p in recent_points if p.pressure > 5]  # 압력 임계값 극한으로 낮춤
        
        if not high_pressure_points:
            # 압력이 낮아도 최근 포인트가 있으면 사용 (매우 민감한 터치 감지)
            if recent_points:
                recent_point = recent_points[-1]
                return TouchPoint(
                    x=recent_point.x,
                    y=recent_point.y,
                    pressure=recent_point.pressure,
                    timestamp=time.time()
                )
            return None  # 포인트가 전혀 없는 경우만 터치 없음으로 처리
        
        avg_x = int(np.mean([p.x for p in high_pressure_points]))
        avg_y = int(np.mean([p.y for p in high_pressure_points]))
        avg_pressure = int(np.mean([p.pressure for p in high_pressure_points]))
        
        return TouchPoint(
            x=avg_x,
            y=avg_y,
            pressure=avg_pressure,
            timestamp=time.time()
        )
    
    def _update_touch_state(self, touch_point: TouchPoint) -> Optional[TouchEvent]:
        """터치 상태를 업데이트하고 적절한 이벤트 생성"""
        current_time = time.time()
        
        if self.current_state == TouchState.NONE:
            # 새로운 터치 시작
            return self._handle_touch_start(touch_point, current_time)
        
        elif self.current_state in [TouchState.TOUCH_START, TouchState.TOUCH_MOVE, TouchState.DRAG_START, TouchState.DRAG_MOVE]:
            # 기존 터치 계속 (드래그 상태 포함)
            return self._handle_touch_continue(touch_point, current_time)
        
        return None
    
    def _handle_touch_start(self, touch_point: TouchPoint, current_time: float) -> TouchEvent:
        """터치 시작 처리"""
        self.current_state = TouchState.TOUCH_START
        self.touch_start_time = current_time
        self.touch_start_point = touch_point
        self.current_point = touch_point
        self.previous_point = None
        
        event = TouchEvent(
            state=TouchState.TOUCH_START,
            current_point=touch_point,
            start_point=touch_point,
            duration=0.0
        )
        
        if self.on_touch_start:
            self.on_touch_start(event)
        
        return event
    
    def _handle_touch_continue(self, touch_point: TouchPoint, current_time: float) -> TouchEvent:
        """터치 계속 처리 (이동 또는 드래그)"""
        self.previous_point = self.current_point
        self.current_point = touch_point
        
        # 드래그 거리 계산
        if self.touch_start_point:
            drag_distance = self._calculate_distance(self.touch_start_point, touch_point)
        else:
            drag_distance = 0.0
        
        # 드래그 임계값 확인
        print(f"DEBUG: 드래그 거리: {drag_distance:.1f}, 임계값: {self.drag_threshold}, 현재 상태: {self.current_state}")
        if drag_distance >= self.drag_threshold:
            # 드래그 상태로 전환
            if self.current_state in [TouchState.TOUCH_START, TouchState.TOUCH_MOVE]:
                print(f"DEBUG: 드래그 시작으로 전환! 거리: {drag_distance:.1f}")
                self.current_state = TouchState.DRAG_START
                event = TouchEvent(
                    state=TouchState.DRAG_START,
                    current_point=touch_point,
                    start_point=self.touch_start_point,
                    previous_point=self.previous_point,
                    drag_distance=drag_distance,
                    duration=current_time - self.touch_start_time
                )
                if self.on_drag_start:
                    self.on_drag_start(event)
                return event
            
            elif self.current_state in [TouchState.DRAG_START, TouchState.DRAG_MOVE]:
                self.current_state = TouchState.DRAG_MOVE
                event = TouchEvent(
                    state=TouchState.DRAG_MOVE,
                    current_point=touch_point,
                    start_point=self.touch_start_point,
                    previous_point=self.previous_point,
                    drag_distance=drag_distance,
                    duration=current_time - self.touch_start_time
                )
                if self.on_drag_move:
                    self.on_drag_move(event)
                return event
        
        else:
            # 일반 터치 이동
            self.current_state = TouchState.TOUCH_MOVE
            event = TouchEvent(
                state=TouchState.TOUCH_MOVE,
                current_point=touch_point,
                start_point=self.touch_start_point,
                previous_point=self.previous_point,
                drag_distance=drag_distance,
                duration=current_time - self.touch_start_time
            )
            if self.on_touch_move:
                self.on_touch_move(event)
            return event
    
    def _handle_touch_end(self) -> TouchEvent:
        """터치 종료 처리"""
        if not self.current_point:
            return None
            
        current_time = time.time()
        duration = current_time - self.touch_start_time
        
        # 드래그 거리 계산
        if self.touch_start_point:
            drag_distance = self._calculate_distance(self.touch_start_point, self.current_point)
        else:
            drag_distance = 0.0
        
        # 이벤트 생성
        if self.current_state in [TouchState.DRAG_START, TouchState.DRAG_MOVE]:
            # 드래그 종료
            event = TouchEvent(
                state=TouchState.DRAG_END,
                current_point=self.current_point,
                start_point=self.touch_start_point,
                previous_point=self.previous_point,
                drag_distance=drag_distance,
                duration=duration
            )
            if self.on_drag_end:
                self.on_drag_end(event)
        else:
            # 일반 터치 종료
            event = TouchEvent(
                state=TouchState.TOUCH_END,
                current_point=self.current_point,
                start_point=self.touch_start_point,
                previous_point=self.previous_point,
                drag_distance=drag_distance,
                duration=duration
            )
            if self.on_touch_end:
                self.on_touch_end(event)
            
            # 클릭 이벤트 확인 (짧은 터치 + 작은 이동) - 캘리브레이션용으로 완화
            if duration < 2.0 and drag_distance < self.min_drag_distance:
                click_event = TouchEvent(
                    state=TouchState.TOUCH_END,
                    current_point=self.current_point,
                    start_point=self.touch_start_point,
                    drag_distance=drag_distance,
                    duration=duration
                )
                if self.on_click:
                    self.on_click(click_event)
        
        # 상태 리셋
        self.current_state = TouchState.NONE
        self.touch_start_point = None
        self.current_point = None
        self.previous_point = None
        self.touch_history.clear()
        
        return event
    
    def _calculate_distance(self, point1: TouchPoint, point2: TouchPoint) -> float:
        """두 터치 포인트 간의 거리 계산"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def reset(self):
        """터치 매니저 상태 리셋"""
        self.current_state = TouchState.NONE
        self.touch_start_point = None
        self.current_point = None
        self.previous_point = None
        self.touch_history.clear()
        self.touch_start_time = 0.0
