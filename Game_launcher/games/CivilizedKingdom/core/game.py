# game.py
import json
from datetime import datetime
from typing import List
import os

class Game:
    """한 게임의 정보(생성시간)를 담는 객체."""
    def __init__(self, created_at: str):
        self.created_at = created_at
        self.index = None  # GameManager가 로드할 때 설정

    @property
    def display_text(self) -> str:
        return f"{self.index}: {self.created_at}"

class GameManager:
    """games.json 파일을 로드/저장하고 Game 객체 리스트를 관리."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.games: List[Game] = []
        self.load_games()

    def load_games(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        self.games = []
        for idx, entry in enumerate(data, start=1):
            if isinstance(entry, str):
                g = Game(entry)
                g.index = idx
                self.games.append(g)

        if len(self.games) != len(data):
            self.save_games()

    def save_games(self):
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump([g.created_at for g in self.games], f, ensure_ascii=False, indent=4)

    def add_game(self) -> Game:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        g = Game(now)
        self.games.append(g)
        for idx, game in enumerate(self.games, start=1):
            game.index = idx
        self.save_games()
        return g

    def delete_game(self, index: int):
        if 0 <= index < len(self.games):
            del self.games[index]
            for idx, game in enumerate(self.games, start=1):
                game.index = idx
            self.save_games()
