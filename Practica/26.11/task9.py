import sqlite3

con = sqlite3.connect('db.sqlite4')
cur = con.cursor()

directors__video_products_data = [
    (1, "For Three Men The Civil War Wasn't Hell. It Was Practice!"),
    (2, "This isn't the movies anymore"),
    (3, "Tonight on Murder She Wrote"),
    (4, "I'll be back")
]

cur.executemany('INSERT INTO directors__video_products VALUES(?, ?);', directors__video_products_data)

directors_data = [
    (1, 'Мультфильм'),
    (2, 'Мультсериал'),
    (3, 'Фильм'),
    (4, 'Сериал')
]

cur.executemany('INSERT INTO directors VALUES(?, ?);', directors_data)

video_products_data = [
    (1, 'Безумные мелодии Луни Тинз', 2, None),
    (2, 'Весёлые мелодии', 2, None),
    (3, 'Хороший, плохой, злой', 3, 1),
    (4, 'Последний киногерой', 3, 2),
    (5, 'Она написала убийство', 4, 3)
]

cur.executemany('INSERT INTO video_products VALUES(?, ?, ?, ?);', video_products_data)

con.commit()
con.close()