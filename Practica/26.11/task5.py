import sqlite3

con = sqlite3.connect('db.sqlite')
cur = con.cursor()

# video_products = [
#     (1, 'Безумные мелодии Луни Тинз', 2),
#     (2, 'Весёлые мелодии', 2),
#     (3, 'Кто подставил кролика Роджера', 3),
#     (4, 'Хороший, плохой, злой', 3),
#     (5, 'Последний киногерой', 3),
#     (6, 'Она написала убийство', 4),
#     (7, 'Миссис Харрис едет в Париж', 3),
# ]

# product_types = [
#     (1, 'Мультфильм'),
#     (2, 'Мультсериал'),
#     (3, 'Фильм'),
#     (4, 'Сериал'),
# ]

# cur.executemany('INSERT INTO product_types VALUES(?, ?);', product_types)
# cur.executemany('INSERT INTO video_products VALUES(?, ?, ?);', video_products)

results = cur.execute('''
SELECT video_products.name, 
        product_types.name
FROM video_products, 
        product_types
WHERE video_products.type_id = product_types.id
AND
product_types.name = 'Фильм';
''')

for result in results:
    print(result)

con.commit()
con.close()
