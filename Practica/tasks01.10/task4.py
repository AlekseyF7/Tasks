#задача про генератор случайных паролей с настройками
from random import *
from string import *

while True:
    g = ''
    print('Введите длину пароля: ')
    try: lenn = int(input())
    except ValueError:
        print('ВВОДИ ЦЕЛОЕ ЧИСЛО')
        continue

    print('Вводите 1 если нужен, 0 если не нужен')
    print('Должен ли быть верхний регистр?')
    while True:
        try:
            if int(input()) == 1:
                g += ascii_uppercase
                break
        except:
            print('НУЖЕН ЛИБО 0  ЛИБО 1')
    print('Должен ли быть нижний регистр?')
    while True:
        try:
            if int(input()) == 1:
                g += ascii_lowercase
                break
        except:
            print('НУЖЕН ЛИБО 0  ЛИБО 1')
    print('Должны ли быть цифры?')
    while True:
        try:
            if int(input()) == 1:
                g += digits
                break
        except:
            print('НУЖЕН ЛИБО 0  ЛИБО 1')
    print('Должны ли быть спец символы?')
    while True:
        try:
            if int(input()) == 1:
                g += '!@#$%^&*()'
                break
        except:
            print('НУЖЕН ЛИБО 0  ЛИБО 1')

    passs = ''.join(choice(g) for _ in range(lenn))
    passs = list(passs)
    shuffle(passs)
    passs = ''.join(passs)
    print(passs)
    print('ЕСЛИ ХОТИТЕ ЕЩЕ ОДИН ПАРОЛЬ--1 ЕСЛИ ХОТИТЕ ВЫЙТИ--ЛЮБОЙ ДРУГОЙ СИМВОЛ')
    if input() == '1':
        continue
    else:
        break