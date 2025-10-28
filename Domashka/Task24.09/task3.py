#задание про генератор паролей
from random import *
from string import *

print('Введите длину пароля: ')

spec_sym = '!@#$%^&*()'
s = int(input())
pas = ''.join(choice(ascii_uppercase + digits + spec_sym) for _ in range(s))
pas = list(pas)
shuffle(pas)
pas = ''.join(pas)
print(pas)