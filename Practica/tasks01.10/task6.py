import random
import time
import os

clear = lambda: os.system('cls')

words = ['калий', 'турок', 'скрип', 'салон']

stages = [
    # 0 ошибок
    """



     o    
    /|\\   
    / \\   

    """,
    # 1 ошибка
    """


     |     
     o    
    /|\\   
    / \\   

    """,
    # 2 ошибки
    """

    +----+  
     |     
     o    
    /|\\   
    / \\   

    """,
    # 3 ошибки
    """
     +     
    +----+  
     |     
     o    
    /|\\   
    / \\   

    """,
    # 4 ошибки
    """
     +     
    +----+  
     |     
     o    
    /|\\   
    / \\   
    =======
    """,
    # 5 ошибок
    """
     +     
    +----+  
     |    
     o    
    /|\\ | 
    / \\ |  
    =======
    """,
    # 6 ошибок — мёртвый
    """
     +     
    +----+  
     |   |   
     X  |
    /|\\ | 
    / \\ | 
    =======
    """
]


def check(word: str = 'актёр', letter: str = 'а'):
    if len(letter) == 1:
        return letter in word
    else:
        return 'error'


def render(errors: int = 0, word: str = 'актёр', letters: list = ['к', "т"], wrong_letters: list = ['щ', 'и']):
    clear()
    hidden_word = ['_', '_', '_', '_', '_']
    c = 0
    for i in word:
        if i in letters:
            hidden_word[c] = i
        c += 1

    if '_' not in hidden_word: win(errors, word, letters, wrong_letters)
    if errors == 6: lose(errors, word, letters, wrong_letters)

    print(f"--- Ошибок: {errors} ---")
    print(stages[errors])
    print("\n")
    print(f"Слово: {''.join(hidden_word)}")
    print(f"Неправильные буквы: {''.join(sorted(wrong_letters))}")
    print("\n")

    letter = input('Угадать букву: ').lower()
    if check(word, letter) == 'error':
        print("\n")
        print("Не понял! Попробуй еще раз!")
        print("\n")
    if check(word, letter):
        letters.append(letter)
    else:
        errors += 1
        wrong_letters.append(letter)
    render(errors, word, letters, wrong_letters)


def start():
    print('Игра Виселица!')
    print('Правила очень просты: я загадываю русское слово из 5-ти букв, твоя задача отгадать его!')
    print('У тебя есть право на 5 ошибок. Совершив 6-ую ошибку ты проиграешь')
    print("\n")
    print("\n")
    game()


def game():
    word = random.choice(words)

    print("Игра началась!")
    time.sleep(5)
    render(0, word, [], [])


def win(errors: int = 0, word: str = 'актёр', letters: list = ['к', "т"], wrong_letters: list = ['щ', 'и']):
    print("\n")
    print(f"Ты выиграл!")
    print(f"--- Ошибок: {errors} ---")
    print(stages[errors])
    print("\n")
    print(f"Слово: {word}")
    print(f"Неправильные буквы: {''.join(sorted(wrong_letters))}")
    print("\n")
    time.sleep(99999)


def lose(errors: int = 0, word: str = 'актёр', letters: list = ['к', "т"], wrong_letters: list = ['щ', 'и']):
    print("\n")
    print(f"Ты проиграл!")
    print(f"--- Ошибок: {errors} ---")
    print(stages[errors])
    print("\n")
    print(f"Слово: {word}")
    print(f"Неправильные буквы: {''.join(sorted(wrong_letters))}")
    print("\n")
    time.sleep(99999)
start()