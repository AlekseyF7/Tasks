num = int(input('Введите ваше число: '))
chet = 'четное' if num % 2 == 0 else 'нечетное'
diapason = 'принадлежит диапазону (10, 50)' if 10 < num < 50 else 'не принадлежит диапазону (10, 50)'
print(f'Число {chet}, {diapason}.')