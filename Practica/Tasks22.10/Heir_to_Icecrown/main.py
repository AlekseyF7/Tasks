# Основной цикл игры прописан здесь

from story import (
    scene_0_intro,
    scene_1_sword,
    scene_2_battle_one,
    scene_3_betrayal,
    scene_4_battle_two,
    scene_5_final,
)
from characters import Knight


def main():
    print('Наследник Ледяной Короны')

    print('===================')
    name_hero = str(input('Сперва скажи свое имя рыцарь!: '))
    player = Knight(name_hero)
    print('---------------')
    scene_0_intro(name_hero)

    print('---------------')
    scene_1_sword(player)

    print('---------------')
    if not scene_2_battle_one(player):
        print("Вы проиграли.")
        return

    print('---------------')
    scene_3_betrayal(name_hero)

    print('---------------')
    scene_4_battle_two(player)

    print('---------------')
    scene_5_final()

if __name__ == '__main__':
    main()