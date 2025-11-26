import sqlite3

con = sqlite3.connect('db.sqlite2')
cur = con.cursor()

cur.executescript('''
CREATE TABLE IF NOT EXISTS slogans (
    id INTEGER PRIMARY KEY,
    slogan_text TEXT
);

CREATE TABLE IF NOT EXISTS product_types (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_products (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    type_id INTEGER,
    slogan_id INTEGER,
    FOREIGN KEY(type_id) REFERENCES product_types(id),
    FOREIGN KEY(slogan_id) REFERENCES slogans(id)
)
''')

slogans_data = [
    (1, "For Three Men The Civil War Wasn't Hell. It Was Practice!"),
    (2, "This isn't the movies anymore"),
    (3, "Tonight on Murder She Wrote"),
    (4, "I'll be back")
]

cur.executemany('INSERT INTO slogans VALUES(?, ?);', slogans_data)

product_types_data = [
    (1, 'Мультфильм'),
    (2, 'Мультсериал'),
    (3, 'Фильм'),
    (4, 'Сериал')
]

cur.executemany('INSERT INTO product_types VALUES(?, ?);', product_types_data)

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
