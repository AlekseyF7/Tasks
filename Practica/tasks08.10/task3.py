from random import randint

WinCounterBot = WinCounterPlayer = 0
while WinCounterBot != 3 and WinCounterPlayer != 3:
    print('камень---1\n'
          'нож---2\n'
          'бумага---3\n')
    ChoicePlayer = input()
    if all(ChoicePlayer != x for x in ('123')): continue
    ChoiceBot = str(randint(1, 3))
    if ChoiceBot == ChoicePlayer:
        print('ничья')
        continue
    else:
        if ((ChoicePlayer == '3' and ChoiceBot == '1')
                or (ChoicePlayer == '2' and ChoiceBot == '3')
                or (ChoicePlayer == '1' and ChoiceBot == '2')):
            print('Забрал раунд')
            WinCounterPlayer += 1
        else:
            print('Потерял раунд')
            WinCounterBot += 1
if WinCounterBot == 3:
    print('Игра проиграна')
else:
    print('Победа, победа вместо обеда!')

