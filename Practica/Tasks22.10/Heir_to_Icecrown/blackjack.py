import random

# Колода: 2–10, J/Q/K = 10, A = 1 или 11
DECK = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4  # 52 карты

def create_deck():
    """Создаёт и перемешивает колоду."""
    deck = DECK.copy()
    random.shuffle(deck)
    return deck

def calculate_hand(hand):
    """Считает сумму руки с учётом тузов (11 или 1)."""
    total = sum(hand)
    aces = hand.count(11)
    # Если перебор и есть тузы — считаем их как 1
    while total > 21 and aces:
        total -= 10  # 11 превращаем в 1
        aces -= 1
    return total

def deal_card(deck):
    """Выдаёт одну карту из колоды."""
    return deck.pop()

def dragon_blackjack(player_name="Рыцарь"):
    """Игра в блэкджек против дракона Сарторакса."""
    print("\nСарторакс расправляет крылья.")
    print('"Я буду подчиняться тебе если победишь меня в Кровавом Блэкджеке"')
    print('"Да начнётся Кровавый Блэкджек!!!"')
    print("Правила: набери 21 или больше, чем дракон, но не переборщи!")
    print()

    deck = create_deck()

    # Раздача по 2 карты
    player_hand = [deal_card(deck), deal_card(deck)]
    dragon_hand = [deal_card(deck), deal_card(deck)]

    # Показываем руку игрока и одну карту дракона
    print(f"Твои карты: {player_hand} --> {calculate_hand(player_hand)}")
    print(f"Карты дракона: [?, {dragon_hand[1]}]")

    # === Ход игрока ===
    while True:
        current = calculate_hand(player_hand)
        if current >= 21:
            break

        action = input("\nВзять карту (h) или остановиться (s)? ").strip().lower()
        if action == 'h':
            card = deal_card(deck)
            player_hand.append(card)
            new_total = calculate_hand(player_hand)
            print(f"Ты взял: {card} --> Всего: {new_total}")
            if new_total > 21:
                print("💥 Перебор! Ты проиграл.")
                return False
        elif action == 's':
            break
        else:
            print("Неверный ввод. Введите 'h' или 's'.")

    # === Ход дракона (автоматический) ===
    print(f"\nДракон открывает карты: {dragon_hand} --> {calculate_hand(dragon_hand)}")
    while calculate_hand(dragon_hand) < 17:
        card = deal_card(deck)
        dragon_hand.append(card)
        print(f"Дракон берёт: {card} --> Всего: {calculate_hand(dragon_hand)}")

    # === Определение победителя ===
    player_total = calculate_hand(player_hand)
    dragon_total = calculate_hand(dragon_hand)

    print(f"\nИтог:")
    print(f"Ты: {player_total}")
    print(f"Сарторакс: {dragon_total}")

    if dragon_total > 21:
        print("Дракон перебрал! Ты победил!")
        return True
    elif player_total > dragon_total:
        print("Ты набрал больше! Победа!")
        return True
    elif player_total == dragon_total:
        print("Ничья... но дракон не терпит равных.")
        return False
    else:
        print("Дракон сильнее. Ты проиграл.")
        return False