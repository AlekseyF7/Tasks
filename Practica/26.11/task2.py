import sqlite3


con = sqlite3.connect('db.sqlite3')
cur = con.cursor()

original_titles = [
    (1, 'Last Action Hero'),
    (2, 'Murder, She Wrote'),
    (3, 'Looney Tunes'),
    (4, 'Il Buono, il brutto, il cattivo'),
    (5, 'Who Framed Roger Rabbit'),
    (6, 'Merrie Melodies'),
    (7, 'Mrs. / Arris Goes to Paris')
]

video_products = [
    (1, 'Безумные мелодии Луни Тюнз', 3),
    (2, 'Весёлые мелодии', 6),
    (3, 'Кто подставил кролика Роджера', 5),
    (4, 'Хороший, плохой, злой', 4),
    (5, 'Последний киногерой', 1),
    (6, 'Она написала убийство', 2),
    (7, 'Мисси Харрис едет в Париж', 7)
]

cur.executemany('INSERT INTO original_titles VALUES(?, ?);', original_titles)
cur.executemany('INSERT INTO video_products VALUES(?, ?, ?);', video_products)

con.commit()
con.close()
