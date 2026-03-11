import psycopg2
import csv
import json

# Database connection
conn = psycopg2.connect(
    dbname="",
    user="",
    password="",
    host="",
    port=""
)
cur = conn.cursor()

# Run the query
query = "SELECT DATA FROM corpus.data_pt_beta limit 100;"
cur.execute(query)
rows = cur.fetchall()

# Export to CSV
#with open("output.csv", "w", newline="") as f:
    #writer = csv.writer(f)
    #writer.writerow([desc[0] for desc in cur.description])  # Column headers
    #writer.writerows(rows)

# Export to JSON
data = [dict(zip([desc[0] for desc in cur.description], row)) for row in rows]
with open("output.json", "w") as f:
    json.dump(data, f, indent=4)

cur.close()
conn.close()
print("Export complete!")