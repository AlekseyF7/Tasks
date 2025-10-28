#задача про кондитерскую лавку
def print_pack_report(val:int = 30):
    for i in range(val, 0, -1):
        if i%5==0 and i%3==0:
            print(f'{i} - расфасуем по 3 или по 5')
        elif i%5==0:
            print(f'{i} - расфасуем по 5')
        elif i%3==0:
            print(f'{i} - расфасуем по 3')
        else:
            print(f'{i} - не заказываем!')

while True:
    try:
        print_pack_report(int(input('\nВведите число: ')))
    except:
        print('Вы ввели что-то не то, давайте еще раз')
