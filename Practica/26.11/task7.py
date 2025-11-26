import sqlite3

con = sqlite3.connect('db.sqlite2')
cur = con.cursor()

results1 = cur.execute('''
SELECT *
FROM video_products
JOIN slogans  ON video_products.slogan_id = slogans.id;
''')

results2 = cur.execute('''
SELECT video_products.title,
        slogans.slogan_text
FROM video_products
CROSS JOIN slogans;                    
''')
for result in results1:
    print(result)

for resul in results2:
    print(resul)

con.close()