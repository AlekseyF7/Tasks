#задача про анализатор текста
gl = 'уеыаоэяиюё'
alf = 'йцукенгшщзхъфывапролджэячсмитьбюё'

def analyzer(text: str):
    gl_letters = 0
    total_letters = 0
    for i in alf:
        total_letters += text.count(i)
        if i in gl: gl_letters += text.count(i)

    negl_letters = total_letters - gl_letters
    space_num = text.count(' ')
    words_num = len(text.split(' '))

    letters = {}
    for i in text:
        if i in letters:
            letters[i] += 1
        else:
            letters[i] = 1

    top1 = [0, '']
    top2 = [0, '']
    top3 = [0, '']
    for i in letters:
        if letters[i] > top1[0]:
            top3 = top2
            top2 = top1
            top1 = [letters[i], i]

        elif letters[i] > top2[0]:
            top3 = top2
            top2 = [letters[i], i]

        elif letters[i] > top3[0]:
            top3 = [letters[i], i]

    print(f'Количество гласных символов: {gl_letters}')
    print(f'Количество негласных символов: {negl_letters}')
    print(f'Количество пробелов: {space_num}')
    print(f'Топ 3 самых часто встречающихся символов: {top1[1], top2[1], top3[1]}')
    print(f'Количество слов: {words_num}')


text = input('Введите текст для анализа: ')
analyzer(text)
