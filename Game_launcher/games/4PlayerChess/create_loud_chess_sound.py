#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
매우 크고 명확한 체스 말 놓는 소리 생성 스크립트
더 강력하고 탁한 느낌의 체스 소리를 생성합니다.
"""

import wave
import struct
import math
import random
import os

def generate_loud_chess_sound():
    """매우 크고 명확한 체스 말 놓는 소리를 생성합니다."""
    
    # 오디오 파라미터
    sample_rate = 44100  # Hz
    duration = 0.5  # 0.5초 (더 길게)
    num_samples = int(sample_rate * duration)
    
    # 기본 주파수 (더 낮은 주파수로 탁한 느낌)
    base_freq = 120  # Hz (더 낮게)
    
    # 여러 주파수를 조합하여 강력한 타격음 생성
    frequencies = [base_freq, base_freq * 1.5, base_freq * 2, base_freq * 2.5, base_freq * 3, base_freq * 4]
    amplitudes = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        
        # 사인파 조합 (더 강력하게)
        sound = 0
        for freq, amp in zip(frequencies, amplitudes):
            sound += amp * math.sin(2 * math.pi * freq * t)
        
        # 더 급격한 감쇠 효과 (더 탁한 느낌)
        decay = math.exp(-2 * t)  # 더 천천히 감쇠
        sound *= decay
        
        # 더 많은 노이즈 추가 (더 자연스러운 타격음)
        noise = (random.random() - 0.5) * 0.3
        sound += noise * decay
        
        # 볼륨 조정 (매우 크게)
        sound *= 0.9  # 안전한 범위로 조정
        
        # 클리핑 방지
        sound = max(-0.95, min(0.95, sound))
        
        # 16비트 정수로 변환
        sample = int(sound * 32767)
        samples.append(sample)
    
    return samples, sample_rate

def save_wav_file(samples, sample_rate, filename):
    """WAV 파일로 저장합니다."""
    
    with wave.open(filename, 'w') as wav_file:
        # 파라미터 설정
        wav_file.setnchannels(1)  # 모노
        wav_file.setsampwidth(2)  # 16비트
        wav_file.setframerate(sample_rate)
        
        # 샘플 데이터 쓰기
        for sample in samples:
            wav_file.writeframes(struct.pack('<h', sample))

def create_loud_chess_sound():
    """매우 큰 체스 소리 파일을 생성합니다."""
    
    # sounds 디렉토리 생성
    sounds_dir = "resources/sounds"
    os.makedirs(sounds_dir, exist_ok=True)
    
    # 소리 생성
    samples, sample_rate = generate_loud_chess_sound()
    
    # 파일 저장
    filename = os.path.join(sounds_dir, "chess_move.wav")
    save_wav_file(samples, sample_rate, filename)
    
    print(f"매우 큰 체스 소리가 생성되었습니다: {filename}")
    return filename

if __name__ == "__main__":
    create_loud_chess_sound() 