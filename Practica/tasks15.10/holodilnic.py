from decimal import Decimal
from datetime import datetime, date
import os

clear = lambda: os.system('cls')
DATE_FORMAT = '%Y-%m-%d'
goods = {
    'Пельмени Универсальные': [
        # Первая партия продукта 'Пельмени Универсальные':
        {'amount': Decimal('0.5'), 'expiration_date': date(2023, 7, 15)},
        # Вторая партия продукта 'Пельмени Универсальные':
        {'amount': Decimal('2'), 'expiration_date': date(2023, 8, 1)},
    ],
    'Пельмени Ариант': [
        # Первая партия продукта 'Пельмени Универсальные':
        {'amount': Decimal('0.8'), 'expiration_date': date(2025, 10, 13)}
    ],
    'Вода': [
        {'amount': Decimal('1.5'), 'expiration_date': None}
    ],
}


def parse_date(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, DATE_FORMAT).date()
    except:
        return None


def add(title: str = 'Аджика', amount: Decimal = Decimal('1'), expiration_date=None):
    if title in goods:
        goods[title] += [{'amount': amount, 'expiration_date': expiration_date}]
    else:
        goods[title] = [{'amount': amount, 'expiration_date': expiration_date}]


def add_by_note(s: str = 'Аджика 1 2025-10-13'):
    s = s.split(' ')
    expiration_date = parse_date(s[-1])

    if expiration_date:
        amount = s[-2]
        title = ' '.join(s[-3::-1])
    else:
        amount = s[-1]
        title = ' '.join(s[-2::-1])

    amount = Decimal(amount)
    add(title=title, amount=amount, expiration_date=expiration_date)
    return title, amount, expiration_date


def find(s: str = 'Пельмени') -> dict:
    s = s.lower()
    ans = {}
    for i in goods.keys():
        if i.lower().count(s):
            ans[i] = goods[i]
    return ans


def amount(s: str = 'Пельмени') -> Decimal:
    products = find(s)
    count = 0
    for product in products.keys():
        for i in products[product]:
            count += i['amount']

    return count


def put_food_in_refrigerator(error: bool = False):
    clear()
    if error:
        print('Произошла ошибка! Давай попробуем еще раз.')
    else:
        print('Я вижу, ты купил новые продукты!')
    print('Что будем класть? ')
    print('Напиши в формате: {НАЗВАНИЕ} {КОЛ_ВО} {ГОД-МЕСЯЦ-ДЕНЬ (опционально)}')
    s = input()

    try:
        title, amount, expiration_date = add_by_note(s)
    except:
        put_food_in_refrigerator(error=True)
    else:
        print(f'Отлично! Ты успешно добавил {title} в кол-ве {amount} кг')
        input('Нажмите enter, чтобы продолжить')
        menu()


def find_food_in_refrigerator():
    clear()
    print('Да ты голоден!')
    print('Что хочешь найти? ')
    print('Напиши название продукта')
    s = input('Ввод: ')
    response = find(s)
    total = amount(s)

    if not response:
        print('К сожалению, я ничего не нашел :(')
        input('Нажмите enter, чтобы продолжить')
        menu()
    else:
        print('Нашел!')
        print("-" * 50)

        for product in response.keys():
            print(f'📦 {product}:')
            count_of_part = 0
            for part in response[product]:
                count_of_part += 1
                if part['expiration_date']:
                    print(f'• Партия {count_of_part}: {part['amount']} кг, срок годности: {part['expiration_date']}')
                else:  # Если нет
                    print(f'• Партия {count_of_part}: {part['amount']} кг')

        print("-" * 50)
        print(f'Всего: {total} кг')
        input('Нажмите enter, чтобы продолжить')
        menu()


def menu(error: bool = False):
    while True:
        clear()
        if error:
            print('Не понял! Давайте еще раз.')
        else:
            print('Главное меню Холодильника.')
            print(f'На данный момент холодильник находиться {amount('')} кг еды!')
        print('Возможные операции:')
        print('1. Положить продукты в холодильник')
        print('2. Найти продукты в холодильнике')
        choice = input('Какое действие Вы хотите сделать? Напишите только цифру: ')

        if choice == '1':
            put_food_in_refrigerator()
        elif choice == '2':
            find_food_in_refrigerator()
        else:
            menu(error=True)


if __name__ == '__main__':
    menu()
