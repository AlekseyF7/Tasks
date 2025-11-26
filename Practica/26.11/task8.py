import sqlite3

con = sqlite3.connect('db.sqlite4')
cur = con.cursor()

cur.executescript('''
CREATE TABLE IF NOT EXISTS video_products(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS directors(
    id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS directors__video_products(
    director_id INTEGER NOT NULL,
    video_product_id INTEGER NOT NULL,
    -- Пару полей назначаем композитным первичным ключом:
    PRIMARY KEY(director_id, video_product_id),
    FOREIGN KEY(director_id) REFERENCES directors(id),
    FOREIGN KEY(video_product_id) REFERENCES video_products(id)
);
''')

con.commit()
con.close()