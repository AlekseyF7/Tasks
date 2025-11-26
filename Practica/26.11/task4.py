import sqlite3


con = sqlite3.connect('db.sqlite')
cur = con.cursor()
cur .executescript('''
CREATE TABLE IF NOT EXISTS product_types(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS video_products(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type_id INTEGER NOT NULL,
    FOREIGN KEY(type_id) REFERENCES product_types(id)
);
''')

con.close()
