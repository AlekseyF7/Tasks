import random

def check_winners(scores, student_score):
    result = ''
    top_three = sorted(scores, reverse=True)[:3]
    if student_score in top_three:
        result = 'Вы в тройке победителей!'
    else:
        result = 'Вы не попали в тройку победителей.'
    return result

scores = random.sample(range(1,101), 10)
student_score = int(input('Введите баллы Стаса: '))
if student_score not in scores:
    scores.append(student_score)
result = check_winners(scores, student_score)

print(result)