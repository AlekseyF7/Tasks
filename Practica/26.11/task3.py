import sqlite3


con = sqlite3.connect('db.sqlite3')
cur = con.cursor()

results = cur.execute('''
SELECT 
    video_products.title,
    original_titles.title
FROM 
    video_products,
    original_titles
WHERE 
    video_products.original_title_id = original_titles.id;
''')

for result in results:
    print(result)

con.close()