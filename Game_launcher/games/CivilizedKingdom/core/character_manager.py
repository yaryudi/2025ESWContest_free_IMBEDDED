# -*- coding: utf-8 -*-
# 
# D&D 캐릭터 매니저 GUI 백업본
# 1차 수정: 2025-05-17 21:24
# 2차 수정: 2025-05-17 22:22    캐릭터 클래스 12종류로 확장
# 3차 수정: 2025-05-17 22:29    종족 정보, 세부 종족, 배경 추가, 그리드 형태로 변경
# 4차 수정: 2025-05-17 23:24    7종의 클래스에 대해 레벨에 따라 마법 선택 기능              (레벨 업 시 오류 발생)
# 5차 수정: 2025-05-18 00:17    캐릭터 목록 창 수정, 레벨업 시 능력치 향상 기능 추가 시도   (실패, 계속 수정중. 캐릭터 목록 창에서 스펠 목록, 직업 등 로드 불가)
# 6차 수정: 2025-05-18 15:39    캐릭터 생성 버그 픽스, 캐릭터 목록 창 위치 변경 등 (레벨업 오류 수정중)

from dnd_character import Character
from dnd_character.classes import (
    Barbarian, Bard, Cleric, Druid, Fighter, Monk,
    Paladin, Ranger, Rogue, Sorcerer, Warlock, Wizard
)
from dnd_character.spellcasting import spells_for_class_level
import json
import os
import uuid
from typing import Dict, List, Optional

def convert_for_json(obj):
    """재귀적으로 JSON 직렬화 불가 객체를 문자열로 변환"""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(i) for i in obj]
    elif isinstance(obj, (uuid.UUID,)):  # UUID는 문자열로
        return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # 나머지 커스텀 객체(Experience 등)는 문자열로 변환
        return str(obj)

class CharacterManager:
    def __init__(self, save_dir="characters"):
        # 항상 Character_devel/characters 폴더를 사용
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "characters")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 문자열 → 클래스 맵핑
        self.class_map = {
            'barbarian': Barbarian,
            'bard':      Bard,
            'cleric':    Cleric,
            'druid':     Druid,
            'fighter':   Fighter,
            'monk':      Monk,
            'paladin':   Paladin,
            'ranger':    Ranger,
            'rogue':     Rogue,
            'sorcerer':  Sorcerer,
            'warlock':   Warlock,
            'wizard':    Wizard,
        }
        
        # 종족과 서브종족 정보
        self.race_subrace_map = {
            'Human': ['None', 'Variant Human'],
            'Elf': ['High Elf', 'Wood Elf', 'Dark Elf (Drow)'],
            'Dwarf': ['Hill Dwarf', 'Mountain Dwarf'],
            'Halfling': ['Lightfoot', 'Stout'],
            'Gnome': ['Rock Gnome', 'Forest Gnome'],
            'Half-Elf': ['None'],
            'Half-Orc': ['None'],
            'Tiefling': ['None'],
            'Dragonborn': ['None']
        }
        
        # 배경 정보
        self.backgrounds = [
            'Acolyte', 'Criminal', 'Folk Hero', 'Noble', 'Sage',
            'Soldier', 'Entertainer', 'Guild Artisan', 'Hermit',
            'Outlander', 'Urchin', 'Charlatan', 'Haunted One'
        ]

        self.characters = {}
        self.load_characters()
    
    def create_character(
        self,
        name: str,
        character_class: str,
        race: str,
        subrace: str,
        background: str,
        abilities: Dict[str, int],
        spells: Optional[List[str]] = None,
        game_id: str = None
    ) -> None:
        """캐릭터 생성 (name+game_id 조합이 유일하면 생성 허용)"""
        # 이름+game_id 조합이 이미 존재하면 에러
        for c in self.characters.values():
            if c['name'] == name and c.get('game_id') == game_id:
                raise ValueError(f"이미 존재하는 캐릭터 이름입니다: {name}")
        character = {
            "name": name,
            "class": character_class,
            "race": race,
            "subrace": subrace,
            "background": background,
            "abilities": abilities,
            "spells": spells or [],
            "level": 1,
            "experience": 0,
            "game_id": game_id
        }
        self.characters[f"{name}__{game_id}"] = character
        self.save_characters()
        
    def level_up(self, name: str) -> None:
        """캐릭터 레벨업"""
        if name not in self.characters:
            raise ValueError(f"존재하지 않는 캐릭터입니다: {name}")
            
        character = self.characters[name]
        character["level"] += 1
        
        # 레벨업에 따른 스펠 슬롯 증가
        if character["class"] in ["마법사", "성직자"]:
            # TODO: 레벨에 따른 스펠 슬롯 계산 로직 구현
            pass
            
        self.save_characters()
        
    def save_characters(self) -> None:
        """캐릭터 정보 저장"""
        os.makedirs("data/characters", exist_ok=True)
        with open("data/characters/characters.json", "w", encoding="utf-8") as f:
            json.dump(self.characters, f, ensure_ascii=False, indent=2)
            
    def load_characters(self) -> None:
        """캐릭터 정보 로드"""
        try:
            with open("data/characters/characters.json", "r", encoding="utf-8") as f:
                self.characters = json.load(f)
        except FileNotFoundError:
            self.characters = {}
            
    def list_characters(self) -> List[str]:
        """캐릭터 목록 반환"""
        return list(self.characters.keys())
        
    def get_character(self, name: str) -> Dict:
        """캐릭터 정보 반환"""
        if name not in self.characters:
            raise ValueError(f"존재하지 않는 캐릭터입니다: {name}")
        return self.characters[name]
    
    def save_character(self, character):
        """캐릭터를 JSON 파일로 저장합니다."""
        character_data = vars(character).copy()
        # 반드시 저장할 추가 정보들
        if hasattr(character, 'classs') and hasattr(character.classs, 'name'):
            character_data['class_name'] = character.classs.name
        if hasattr(character, 'spells_known'):
            character_data['spells_known'] = character.spells_known
        if hasattr(character, 'spells_prepared'):
            character_data['spells_prepared'] = character.spells_prepared
        if hasattr(character, 'cantrips_known'):
            character_data['cantrips_known'] = character.cantrips_known

        # UUID 등 직렬화 불가 객체를 문자열로 변환
        character_data = convert_for_json(character_data)

        file_path = os.path.join(self.save_dir, f"{character.name}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)
    
    def load_character(self, character_name):
        """저장된 캐릭터를 불러옵니다."""
        file_path = os.path.join(self.save_dir, f"{character_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                character_data = json.load(f)
            class_name = character_data.get('class_name', None)
            cls = self.class_map.get(str(class_name).lower(), Character) if class_name else Character
            base_kwargs = {k: character_data[k] for k in ('name', 'level') if k in character_data}
            character = cls(**base_kwargs)
            # 나머지 속성은 setattr로 할당
            for key, value in character_data.items():
                if key not in ('name', 'level'):
                    attr = getattr(type(character), key, None)
                    if isinstance(attr, property) and not attr.fset:
                        continue
                    try:
                        setattr(character, key, value)
                    except Exception:
                        pass
            # class_name, spells_xxx 명시적으로 할당
            if class_name:
                setattr(character, 'class_name', class_name)
            for spell_key in ['spells_known', 'spells_prepared', 'cantrips_known']:
                if spell_key in character_data:
                    setattr(character, spell_key, character_data[spell_key])
            return character
        return None

    def list_characters_by_game(self, game_id: str) -> list:
        """특정 게임에 속한 캐릭터 목록만 반환 (game_id 없는 캐릭터는 제외)"""
        return [name for name, char in self.characters.items() if char.get("game_id") == game_id]

    def delete_character(self, name: str, game_id: str) -> None:
        """name+game_id 조합으로 캐릭터 삭제"""
        key = f"{name}__{game_id}"
        if key in self.characters:
            del self.characters[key]
            self.save_characters()
