import random
import os

clear = lambda: os.system('cls')  # Стирает cmd

bank_accounts = {1234: 100, 9999: 505}  # Банковские счета

# Создать счет
def create_account():
    clear()
    print('Благодарим за создание счёта в Спёрбанк!')
    while True:
        acc_id = random.randint(1000, 9999)  # Генерим номер счета
        if acc_id not in bank_accounts.keys():
            bank_accounts[acc_id] = 0  # Создаем счет с балансом 0
            break
    print(f'Ваш счет под номером {acc_id} успешно зарегистрирован!')
    print(f'На данный момент на счету: {bank_accounts[acc_id]}')
    input('Нажмите enter, чтобы продолжить')
    menu()


# Проверить баланс
def check_balance():
    clear()
    try:
        acc_id = int(input('Введите номер счета, баланс которого Вы хотите узнать: '))
    except:
        print('К сожалению, такого счета не существует.')
        input('Нажмите enter, чтобы продолжить')
        menu()

    if acc_id not in bank_accounts.keys():
        print('К сожалению, такого счета не существует.')
    else:
        print(f'Баланс счета под номером {acc_id} составляет {bank_accounts[acc_id]} рублей.')
    input('Нажмите enter, чтобы продолжить')
    menu()


# Списать со счета
def write_off():
    clear()
    try:
        acc_id = int(input('Введите номер счета, с которого вы хотите снять деньги: '))
        summ = int(input('Введите сумму, которую вы хотите снять: '))
    except:
        print('К сожалению, Вы ввели неккоректное значение. Попробуйте еще раз.')
        input('Нажмите enter, чтобы продолжить')
        menu()

    if acc_id not in bank_accounts.keys():  # Существует ли счет
        print('К сожалению, такого счета не существует.')
    elif summ > bank_accounts[acc_id]:  # Хватает ли денег
        print('К сожалению, на счету недостаточно средств.')
    else:
        bank_accounts[acc_id] -= summ
        print(f'Вы успешно сняли со счета под номером {acc_id} средства в размере {summ} рублей')
        print(f'Текущий баланс: {bank_accounts[acc_id]}')

    input('Нажмите enter, чтобы продолжить')
    menu()


# Пополнить счет
def top_up():
    clear()
    try:
        acc_id = int(input('Введите номер счета, который вы хотите пополнить: '))
        summ = int(input('Введите сумму, на которую вы хотите пополнить: '))
    except:
        print('К сожалению, Вы ввели неккоректное значение. Попробуйте еще раз.')
        input('Нажмите enter, чтобы продолжить')
        menu()

    if acc_id not in bank_accounts.keys():  # Существует ли счет
        print('К сожалению, такого счета не существует.')
    else:
        bank_accounts[acc_id] += summ
        print(f'Вы успешно пополнили счет под номером {acc_id} средства в размере {summ} рублей')
        print(f'Текущий баланс: {bank_accounts[acc_id]}')

    input('Нажмите enter, чтобы продолжить')
    menu()

# Перевод с счета на счет
def money_transfer():
    clear()
    try:
        from_id = int(input('Введите номер счета отправителя: '))
        to_id = int(input('Введите номер счета получателя: '))
        summ = int(input('Введите сумму, перевода: '))
    except:
        print('К сожалению, Вы ввели неккоректное значение. Попробуйте еще раз.')
        input('Нажмите enter, чтобы продолжить')
        menu()

    if (from_id not in bank_accounts.keys()) or (to_id not in bank_accounts.keys()):  # Существуют ли оба счета
        print('К сожалению, такого счета не существует.')
    elif summ > bank_accounts[from_id]:  # Хватает ли денег у отправителя
        print('К сожалению, на счету отправителя недостаточно средств.')
    else:
        clear()
        bank_accounts[from_id] -= summ
        bank_accounts[to_id] += summ
        print(f'Вы успешно выполнили перевод со счета {from_id} на счет {to_id}. Сумма перевода составила {summ}')
        print(f'Текущий баланс счета {from_id}: {bank_accounts[from_id]}')
        print(f'Текущий баланс счета {to_id}: {bank_accounts[to_id]}')

    input('Нажмите enter, чтобы продолжить')
    menu()


# Основное меню банка
def menu(error: bool = False):  # error - передается true когда пользователь вводит что-то не то
    while True:
        clear()
        if error:  # Если пользователь написал что-то непонятное
            print('Не понял! Давайте еще раз.')
        else:
            print('Вы находитесь в главном меню банка Спёрбанк.')
            print(f'На данный момент в банке зарегистрированно {len(bank_accounts)} счета/ов')
        print('Возможные операции:')
        print('1. Перевод')
        print('2. Пополнение счета')
        print('3. Списание со счета')
        print('4. Проверка баланса счета')
        print('5. Создание счета')
        choice = input('Какое действие Вы хотите сделать? Напишите только цифру: ')

        if choice == '1':
            money_transfer()
        elif choice == '2':
            top_up()
        elif choice == '3':
            write_off()
        elif choice == '4':
            check_balance()
        elif choice == '5':
            create_account()
        else:
            menu(error=True)


menu()
