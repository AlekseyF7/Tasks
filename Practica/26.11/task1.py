import sqlite3


con = sqlite3.connect('db.sqlite3')
cur = con.cursor()

cur.executescript('''
CREATE TABLE IF NOT EXISTS original_titles(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_products(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    original_title_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY(original_title_id) REFERENCES original_titles(id)
);
''')

con.close()