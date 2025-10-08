import random

def main():
    num = random.randint(1,101)
    print('Игра угадай число')

    try_one = int(input('Первая попытка! Введите число: '))
    if try_one == num:
        print('Вы угадали число!')
    else:
        if try_one > num:
            print('Загаданное число меньше')
        else:
            print('Загаданное число больше')
            try_two = int(input('Вторая попытка! Введите число: '))
            if try_two == num:
                print('Вы угадали число!')
            else:
                if num % 2 == 0:
                    print('Загаданное число чётное. Осталась 1 попытка!')
                else:
                    print('Загаданное число нечётное. Осталась 1 попытка!')
                    try_three = int(input('Третья попытка! Введите число: '))
                    if try_three == num:
                        print("Вы угадали число!")
                    else:
                        print("Неправильно. Попытки кончились")

while True:
    main()
    restart = input("Хотите перезапустить программу? (y/n): ").lower()
    if restart != 'y':
        print("Выход из программы.")
        break
