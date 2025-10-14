import random

def trys(num, k):
    num, k = num, k
    try_ = int(input(f'{k} попытка. Введите число: '))
    if try_ == num:
        print('Вы угадали число!')
        return True
    else:
        if k == 1 and try_ > num:
            print('Загаданное число меньше')
        elif k ==1 and try_ < num:
            print('Загаданное число больше')
        if k == 2 and num % 2 == 0:
            print('Загаданное число чётное.')
        elif k == 2 and num % 2 != 0:
            print('Загаданное число нечётное.')
        if k >= 3:
            print(f"Попытки кончились. Загаданное число это: {num}")


def main():
    k = 0
    num = random.randint(1,3)
    print('Игра угадай число')
    print("Программа загадывает число и у Вас есть 3 попытки его угадать")
    while k < 3:
        k += 1
        func = trys(num, k)
        if func == True:
            break

while True:
    main()
    restart = input("Хотите попробовать еще раз? (y/n): ").lower()
    if restart != 'y':
        print("Выход из игры.")
        break
