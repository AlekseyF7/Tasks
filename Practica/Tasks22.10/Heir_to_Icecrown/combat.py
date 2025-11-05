from characters import Knight, Undead_Warrior, Dragon
import random


def     fight_undead_ambush(player: Knight):
    """Битва с волной восставших рыцарей (1 битва)"""

    # Создаём врагов
    undead_army = [Undead_Warrior() for _ in range(3)]
    print(f'\nПротив тебя - {len(undead_army)} восставших рыцаря!')

    while len(undead_army) > 0 and player.is_alive():
        # Показываемм статус
        print(f'\nСер {player.name}: {player.hp}/{player.max_hp} HP | Честь: {player.honor}')
        if len(undead_army) < 3:
            print(f'\nОтсалось {len(undead_army)} восставших рыцаря')

        # Выбор действия
        print("1. Атаковать мечом")
        if player.lich_sword:
            print("2. Использовать силу меча Артаса (ледяной удар)")

        choice = input("Ваш выбор: ").strip()

        # Атака игрока
        if choice == "2" and player.lich_sword:
            damage = player.use_lich_sword()
            undead_army[0].take_damage(damage)
            print(f"Меч Артаса наносит {damage} урона! {undead_army[0].name} скован льдом.")
        elif choice == "1":
            damage = player.get_attack_damage()
            undead_army[0].take_damage(damage)
            print(f"Ты атакуешь! Нанесено {damage} урона.")
        else:
            print("Ты колеблишься... и получаешь удар!")
            player.take_damage(15)
            continue

        # Проверка умер ли первый враг
        if undead_army[0].is_alive():
            print(f"{undead_army.pop(0).name} повержен!")

        # Атака оставшихся врагов
        for undead in undead_army[:2]:  # максимум 2 атакуют за ход
            if undead.is_alive():
                damage = undead.get_attack_damage()
                player.take_damage(damage)
                print(f"{undead.name} атакует! -{damage} HP")

        if player.is_alive() == False:
            print("\nТы падаешь в снег... Тьма поглощает тебя.")
            return False

    print("\nПоследний рыцарь смерти рассыпается в прах.")
    if player.lich_sword and player.honor < 40:
        print("Меч шепчет: «Ты начинаешь понимать... власть требует жертв».")
    else:
        print("Ты вытираешь кровь с лица. Путь ещё не окончен.")

    return True

def fight_dragon(player: Knight):
    dragon = Dragon()
    while player.is_alive() and dragon.is_alive():
        print(f"\n{player.name}: {player.hp}/{player.max_hp} HP")
        print(f"{dragon.name}: {dragon.hp}/{dragon.max_hp} HP")

        print("1. Атаковать мечом")
        if player.lich_sword:
            print("2. Использовать силу меча Артаса")
        print("3. Защититься (восстановить 66 HP)")

        choice = input("Ваш выбор: ").strip()

        # Действие игрока
        if choice == "1":
            damage = player.get_attack_damage()
            dragon.take_damage(damage)
            print(f"Ты наносишь {damage} урона!")
        elif choice == "2" and player.lich_sword:
            damage = player.use_lich_sword()
            dragon.take_damage(damage)
            print(f"Ледяной шторм! Нанесено {damage} урона!")
        elif choice == "3":
            player.heal(66)
            print("Ты прикрываешься щитом и восстанавливаешь здоровье.")
        else:
            print("Ты теряешь ход!")

            # Атака дракона (если жив)
        if dragon.is_alive():
            # Иногда дракон ревёт (сильная атака)
            if random.random() < 0.3:
                damage = dragon.roar()
                print("Сарторакс выпускает яростный рёв!")
                player.take_damage(damage)
            else:
                damage = dragon.get_attack_damage()
                print("Сарторакс атакует когтями!")
                player.take_damage(damage)

        if not player.is_alive():
            print("\nТы пал... но Сарторакс улетел в небо.")
            return False
    print("\nСарторакс падает на лёд. Он жив... но побеждён.")
    return True