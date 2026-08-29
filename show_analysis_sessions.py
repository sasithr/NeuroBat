from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

cur.execute("""
SELECT *
FROM analysis_sessions
ORDER BY id DESC
LIMIT 10;
""")

rows = cur.fetchall()

column_names = [desc[0] for desc in cur.description]

print("\nNEUROBAT - STORED ANALYSIS RECORDS")
print("=" * 80)

print(" | ".join(column_names))
print("-" * 80)

for row in rows:
    print(" | ".join(str(value) for value in row))

if not rows:
    print("No analysis records found.")

cur.close()
conn.close()