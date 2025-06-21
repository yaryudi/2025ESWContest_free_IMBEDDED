"""
포커 게임의 핸드 평가 모듈
카드 조합을 평가하고 승자를 결정하는 기능을 제공합니다.
"""

from itertools import combinations

class HandEvaluator:
    # 족보 점수 정의 (높은 순서대로)
    ROYAL_STRAIGHT_FLUSH = 10  # 로열 스트레이트 플러시
    STRAIGHT_FLUSH = 9         # 스트레이트 플러시
    FOUR_OF_A_KIND = 8        # 포카드
    FULL_HOUSE = 7            # 풀하우스
    FLUSH = 6                 # 플러시
    STRAIGHT = 5              # 스트레이트
    THREE_OF_A_KIND = 4       # 트리플
    TWO_PAIR = 3              # 투페어
    ONE_PAIR = 2              # 원페어
    HIGH_CARD = 1             # 하이카드

    # 족보 이름 정의 (한글)
    HAND_NAMES = {
        ROYAL_STRAIGHT_FLUSH: "로열 스트레이트 플러시",
        STRAIGHT_FLUSH: "스트레이트 플러시",
        FOUR_OF_A_KIND: "포카드",
        FULL_HOUSE: "풀하우스",
        FLUSH: "플러시",
        STRAIGHT: "스트레이트",
        THREE_OF_A_KIND: "트리플",
        TWO_PAIR: "투페어",
        ONE_PAIR: "원페어",
        HIGH_CARD: "하이카드"
    }

    @staticmethod
    def get_card_value(card):
        """카드의 숫자 값을 반환합니다.
        A=14, K=13, Q=12, J=11, 나머지는 숫자 그대로"""
        value = card[:-1]  # 마지막 문자(문양) 제외
        if value == 'A':
            return 14
        elif value == 'K':
            return 13
        elif value == 'Q':
            return 12
        elif value == 'J':
            return 11
        else:
            return int(value)

    @staticmethod
    def get_card_suit(card):
        """카드의 문양을 반환합니다."""
        return card[-1]

    @staticmethod
    def is_flush(hand):
        """플러시 여부를 확인합니다.
        모든 카드가 같은 문양이면 True를 반환합니다."""
        return len(set(HandEvaluator.get_card_suit(card) for card in hand)) == 1

    @staticmethod
    def is_straight(values):
        """스트레이트 여부를 확인합니다.
        일반적인 스트레이트(연속된 5장)와 A-5 스트레이트를 모두 체크합니다."""
        # 일반적인 스트레이트
        if len(set(values)) == 5 and max(values) - min(values) == 4:
            return True
        # A-5 스트레이트
        if set(values) == {14, 2, 3, 4, 5}:
            return True
        return False

    @staticmethod
    def get_value_counts(values):
        """카드 값의 출현 빈도를 계산합니다.
        예: [A, A, K, Q, Q] -> {14: 2, 13: 1, 12: 2}"""
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        return value_counts

    @staticmethod
    def evaluate_hand(hand):
        """5장의 카드로 구성된 핸드를 평가합니다.
        Returns:
            tuple: (족보 점수, 족보 이름, 카드 값 리스트)"""
        # 카드 정렬 (숫자 내림차순)
        sorted_cards = sorted(hand, key=lambda x: HandEvaluator.get_card_value(x), reverse=True)
        values = [HandEvaluator.get_card_value(card) for card in sorted_cards]
        
        # 플러시 체크
        is_flush = HandEvaluator.is_flush(hand)
        
        # 스트레이트 체크
        is_straight = HandEvaluator.is_straight(values)
        
        # 페어 체크
        value_counts = HandEvaluator.get_value_counts(values)
        counts = sorted(value_counts.values(), reverse=True)
        
        # 족보 판정 (높은 순서대로)
        if is_flush and is_straight:
            if values == [14, 13, 12, 11, 10]:  # 로열 스트레이트 플러시
                return HandEvaluator.ROYAL_STRAIGHT_FLUSH, "로열 스트레이트 플러시", values
            return HandEvaluator.STRAIGHT_FLUSH, "스트레이트 플러시", values
        
        if 4 in counts:  # 포카드
            # 포카드의 숫자와 나머지 가장 높은 카드 선택
            four_value = [v for v, count in value_counts.items() if count == 4][0]
            kicker = max([v for v in values if v != four_value])
            return HandEvaluator.FOUR_OF_A_KIND, "포카드", [four_value] * 4 + [kicker]
        
        if counts == [3, 2]:  # 풀하우스
            # 트리플의 숫자와 페어의 숫자 선택
            three_value = [v for v, count in value_counts.items() if count == 3][0]
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            return HandEvaluator.FULL_HOUSE, "풀하우스", [three_value] * 3 + [pair_value] * 2
        
        if is_flush:  # 플러시
            # 플러시인 경우 가장 높은 5장의 카드를 사용
            flush_values = sorted([HandEvaluator.get_card_value(card) for card in hand], reverse=True)[:5]
            return HandEvaluator.FLUSH, "플러시", flush_values
        
        if is_straight:  # 스트레이트
            return HandEvaluator.STRAIGHT, "스트레이트", values
        
        if 3 in counts:  # 트리플
            # 트리플의 숫자와 나머지 가장 높은 2장의 카드 선택
            three_value = [v for v, count in value_counts.items() if count == 3][0]
            kickers = sorted([v for v in values if v != three_value], reverse=True)[:2]
            return HandEvaluator.THREE_OF_A_KIND, "트리플", [three_value] * 3 + kickers
        
        if counts.count(2) == 2:  # 투페어
            # 두 개의 페어 중 높은 순서대로 선택하고, 나머지 가장 높은 카드 선택
            pairs = sorted([v for v, count in value_counts.items() if count == 2], reverse=True)
            kicker = max([v for v in values if v not in pairs])
            return HandEvaluator.TWO_PAIR, "투페어", pairs * 2 + [kicker]
        
        if 2 in counts:  # 원페어
            # 페어의 숫자와 나머지 가장 높은 3장의 카드 선택
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            kickers = sorted([v for v in values if v != pair_value], reverse=True)[:3]
            return HandEvaluator.ONE_PAIR, "원페어", [pair_value] * 2 + kickers
        
        # 하이카드
        return HandEvaluator.HIGH_CARD, "하이카드", sorted(values, reverse=True)[:5]

    @staticmethod
    def find_best_hand(cards):
        """7장의 카드 중 가장 높은 5장의 조합을 찾습니다.
        Args:
            cards: 7장의 카드 리스트
        Returns:
            tuple: (최고의 5장 조합, 족보 이름, 점수, 카드 값 리스트)"""
        all_combinations = list(combinations(cards, 5))
        best_score = -1
        best_hand = None
        best_hand_name = ""
        best_values = None

        for hand in all_combinations:
            score, hand_name, values = HandEvaluator.evaluate_hand(hand)
            # 족보 점수 > 킥커(값) 순으로 비교
            if (score > best_score) or (score == best_score and (best_values is None or values > best_values)):
                best_score = score
                best_hand = hand
                best_hand_name = hand_name
                best_values = values

        return best_hand, best_hand_name, best_score, best_values

    @staticmethod
    def determine_winner(players_hands, community_cards):
        """승자를 결정합니다.
        Args:
            players_hands: [(플레이어 인덱스, 플레이어 카드 리스트), ...]
            community_cards: 커뮤니티 카드 리스트
        Returns:
            tuple: (승자 인덱스 리스트, 최고 족보 이름, 최고 카드 값 리스트)"""
        best_score = -1
        winners = []
        best_hand_name = ""
        best_values = None

        for player_idx, player_cards in players_hands:
            # 플레이어의 카드와 커뮤니티 카드를 합쳐서 7장의 카드 생성
            all_cards = player_cards + community_cards
            # 7장의 카드 중 가장 높은 5장의 조합 찾기
            _, hand_name, score, values = HandEvaluator.find_best_hand(all_cards)
            
            if score > best_score:
                best_score = score
                winners = [player_idx]
                best_hand_name = hand_name
                best_values = values
            elif score == best_score:
                # 동점인 경우 하이카드 비교
                if values > best_values:
                    winners = [player_idx]
                    best_values = values
                elif values == best_values:
                    winners.append(player_idx)

        return winners, best_hand_name, best_values

