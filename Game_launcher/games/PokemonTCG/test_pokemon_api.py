from pokemontcgsdk import RestClient, Card

# API 키 설정
API_KEY = "43e0a380-a83f-4c14-9b6f-84322e31b7fd"
RestClient.configure(API_KEY)

def count_unique_pokemon_tools():
    print("포켓몬 도구(장착 가능) 카드 개수 검색:")
    # subtype을 'Pokémon Tool'로 한정
    cards = Card.where(q='subtypes:"Pokémon Tool"')
    unique_names = set()
    for card in cards:
        unique_names.add(card.name)
    print(f"중복 없는 포켓몬 도구 카드 수: {len(unique_names)}")

if __name__ == "__main__":
    count_unique_pokemon_tools() 