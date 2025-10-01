def language(text):
    # автоопределитель языка текста (ру или англ)
    ru_letters = 0
    eng_letters = 0
    for char in text:
        if 'а' <= char.lower() <= 'я' or char.lower() == 'ё':
            ru_letters += 1
        elif 'a' <= char.lower() <= 'z':
            eng_letters += 1
    if ru_letters > eng_letters:
        alfavit = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    else:
        alfavit = 'abcdefghijklmnopqrstuvwxyz'
    
    return alfavit

def caser_shifer(text, sdvig, mode):
    result = ""
    alphabet = language(text)

    if mode == 'de':
        sdvig = -sdvig
    
    for char in text:
        if char.lower() in alphabet:
            index = 0
            for i in range(len(alphabet)):
                if alphabet[i] == char.lower():
                    index = i
                    break
            new_index = (index + sdvig) % len(alphabet)
            if char.isupper():
                result += alphabet[new_index].upper()
            else:
                result += alphabet[new_index]
        else:
            result += char
    return result

def menu():
    while True:
        print('Меню выбора действий программы Шифр Цезаря')
        print('1. Зашифровать')
        print('2. Расшифровать')
        print('3. Выйти')
        choice = input('Выберите действие и впишите цифру: ')
        if choice == '1':
            text = input('Впишите текст: ')
            sdvig = int(input('Сдвиг: '))
            enshifr = caser_shifer(text, sdvig, 'en')
            print(f'Зашифрованый текст: {enshifr}')
        elif choice =='2':
            text = input('Впишите текст: ')
            sdvig = int(input('Сдвиг: '))
            deshifr = caser_shifer(text, sdvig, 'de')
            print(f'Расшифрованный текст: {deshifr}')
        elif choice =='3':
            print('Тогда в другой раз. Пока!')
            break
        else:
            print("Неверный выбор!")
menu()
