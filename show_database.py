from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

database_url = os.getenv("DATABASE_URL")

conn = psycopg2.connect(database_url)
cur = conn.cursor()

cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
""")

print("\nNEUROBAT DATABASE TABLES")
print("========================")

rows = cur.fetchall()

for row in rows:
    print(row[0])

cur.close()
conn.close()