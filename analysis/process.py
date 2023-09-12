import pandas as pd
import sqlite3

# Read csv file.
df = pd.read_csv("Book1.csv")

# Connect to (create) database.
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "title": "CharField",
    "author": "CharField",
    "rating": "IntegerField", 
    "best seller": "BooleanField"
}
df.to_sql(name='home_book', con=conn, if_exists='replace', dtype=dtype, index=True, index_label="id")
conn.close()