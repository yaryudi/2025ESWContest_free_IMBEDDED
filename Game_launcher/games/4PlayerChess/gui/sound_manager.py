#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
체스 게임 사운드 관리 모듈
체스 말을 놓을 때 탁하는 소리와 중세 BGM을 재생합니다.
"""

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QSoundEffect, QMediaPlayer, QMediaContent
import os

class SoundManager:
    """체스 게임의 사운드를 관리하는 클래스"""
    
    def __init__(self):
        """사운드 매니저를 초기화합니다."""
        self.move_sound = None
        self.check_sound = None
        self.checkmate_sound = None
        self.bgm_player = None
        self.sound_enabled = True
        self.bgm_enabled = True
        self.bgm_volume = 0.8  # BGM 볼륨을 80%로 설정 (더 크게)
        self.load_sounds()
    
    def load_sounds(self):
        """사운드 파일들을 로드합니다."""
        try:
            # 체스 말 놓는 소리 로드
            sound_path = os.path.join("resources", "sounds", "chess_move.wav")
            if os.path.exists(sound_path):
                self.move_sound = QSoundEffect()
                self.move_sound.setSource(QUrl.fromLocalFile(sound_path))
                self.move_sound.setVolume(1.0)  # 볼륨을 100%로 설정 (최대)
                print(f"체스 사운드 로드됨: {sound_path}")
            else:
                print(f"사운드 파일을 찾을 수 없습니다: {sound_path}")
            
            # 체크 사운드 로드
            check_path = os.path.join("resources", "sounds", "check.wav")
            if os.path.exists(check_path):
                self.check_sound = QSoundEffect()
                self.check_sound.setSource(QUrl.fromLocalFile(check_path))
                self.check_sound.setVolume(1.0)
                print(f"체크 사운드 로드됨: {check_path}")
            else:
                print(f"체크 사운드 파일을 찾을 수 없습니다: {check_path}")
            
            # 체크메이트 사운드 로드
            checkmate_path = os.path.join("resources", "sounds", "checkmate.wav")
            if os.path.exists(checkmate_path):
                self.checkmate_sound = QSoundEffect()
                self.checkmate_sound.setSource(QUrl.fromLocalFile(checkmate_path))
                self.checkmate_sound.setVolume(1.0)
                print(f"체크메이트 사운드 로드됨: {checkmate_path}")
            else:
                print(f"체크메이트 사운드 파일을 찾을 수 없습니다: {checkmate_path}")
            
            # 중세 BGM 로드
            bgm_path = os.path.join("resources", "sounds", "medieval_bgm.wav")
            if os.path.exists(bgm_path):
                self.bgm_player = QMediaPlayer()
                self.bgm_player.setMedia(QMediaContent(QUrl.fromLocalFile(bgm_path)))
                self.bgm_player.setVolume(int(self.bgm_volume * 100))  # QMediaPlayer는 0-100 범위
                # BGM 루프 설정
                self.bgm_player.mediaStatusChanged.connect(self._on_bgm_status_changed)
                print(f"중세 BGM 로드됨: {bgm_path}")
            else:
                print(f"BGM 파일을 찾을 수 없습니다: {bgm_path}")
        except Exception as e:
            print(f"사운드 로드 중 오류 발생: {e}")
    
    def _on_bgm_status_changed(self, status):
        """BGM 상태가 변경될 때 호출됩니다. 루프 재생을 위해 사용됩니다."""
        if status == QMediaPlayer.EndOfMedia and self.bgm_enabled:
            # BGM이 끝나면 다시 시작
            self.bgm_player.setPosition(0)
            self.bgm_player.play()
    
    def play_move_sound(self):
        """체스 말을 놓을 때 소리를 재생합니다."""
        if self.sound_enabled and self.move_sound and self.move_sound.isLoaded():
            self.move_sound.play()
    
    def play_check_sound(self):
        """체크 경고음을 재생합니다."""
        if self.sound_enabled and self.check_sound and self.check_sound.isLoaded():
            self.check_sound.play()
    
    def play_checkmate_sound(self):
        """체크메이트 승리음을 재생합니다."""
        if self.sound_enabled and self.checkmate_sound and self.checkmate_sound.isLoaded():
            self.checkmate_sound.play()
    
    def start_bgm(self):
        """BGM을 시작합니다."""
        if self.bgm_enabled and self.bgm_player:
            self.bgm_player.play()
            print("중세 BGM 시작")
    
    def stop_bgm(self):
        """BGM을 정지합니다."""
        if self.bgm_player:
            self.bgm_player.stop()
            print("중세 BGM 정지")
    
    def pause_bgm(self):
        """BGM을 일시정지합니다."""
        if self.bgm_player:
            self.bgm_player.pause()
            print("중세 BGM 일시정지")
    
    def resume_bgm(self):
        """BGM을 재개합니다."""
        if self.bgm_enabled and self.bgm_player:
            self.bgm_player.play()
            print("중세 BGM 재개")
    
    def set_sound_enabled(self, enabled):
        """사운드 활성화/비활성화를 설정합니다."""
        self.sound_enabled = enabled
    
    def set_bgm_enabled(self, enabled):
        """BGM 활성화/비활성화를 설정합니다."""
        self.bgm_enabled = enabled
        if enabled and self.bgm_player:
            self.bgm_player.play()
        elif not enabled and self.bgm_player:
            self.bgm_player.stop()
    
    def is_sound_enabled(self):
        """사운드가 활성화되어 있는지 확인합니다."""
        return self.sound_enabled
    
    def is_bgm_enabled(self):
        """BGM이 활성화되어 있는지 확인합니다."""
        return self.bgm_enabled
    
    def set_volume(self, volume):
        """사운드 볼륨을 설정합니다. (0.0 ~ 1.0)"""
        if self.move_sound:
            self.move_sound.setVolume(max(0.0, min(1.0, volume)))
    
    def set_bgm_volume(self, volume):
        """BGM 볼륨을 설정합니다. (0.0 ~ 1.0)"""
        self.bgm_volume = max(0.0, min(1.0, volume))
        if self.bgm_player:
            self.bgm_player.setVolume(int(self.bgm_volume * 100))
    
    def get_bgm_volume(self):
        """BGM 볼륨을 반환합니다."""
        return self.bgm_volume 