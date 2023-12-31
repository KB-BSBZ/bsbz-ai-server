import pandas as pd
import sqlite3


# estate total
df = pd.read_csv(f"analysis\csv\estate_total.csv")

# Connect to (create) database.
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "product_id": "IntegerField",
    "price": "IntegerField",
    "year": "IntegerField", 
    "month": "IntegerField",
    "day": "IntegerField",
    "area": "FloatField",
}
df.to_sql(name='pricelog_estatelog', con=conn, if_exists='replace', dtype=dtype, index=True, index_label="id")
conn.close()




# music total
df = pd.read_csv(f"analysis\csv\music_total.csv")

database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "music_id": "IntegerField",
    "ymd": "CharField",
    "price_high": "IntegerField", 
    "price_low": "IntegerField",
    "price_close": "IntegerField",
    "pct_price_change": "FloatField",
    "cnt_units_traded": "IntegerField",
}
df.to_sql(name='pricelog_musiclog', con=conn, if_exists='replace', dtype=dtype, index=True, index_label="id")
conn.close()


# luxury total
df = pd.read_csv(f"analysis\csv\luxury_total.csv")

database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "luxury_id": "IntegerField",
    "ymd" : "CharField",
    "price": "IntegerField",
}
df.to_sql(name='pricelog_luxurylog', con=conn, if_exists='replace', dtype=dtype, index=True, index_label="id")
conn.close()


# Estate Text total
df = pd.read_csv(f"analysis\csv\estate_text.csv")

database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "id" : "IntegerField",
    "contents": "TextField",
    "pubdate" : "CharField",
}
df.to_sql(name='pricelog_estatetext', con=conn, if_exists='replace', dtype=dtype, index=False)
conn.close()

# Estate Text total
df = pd.read_csv(f"analysis\csv\luxury_text.csv")

database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "id" : "IntegerField",
    "contents": "TextField",
    "pubdate" : "CharField",
}
df.to_sql(name='pricelog_luxurytext', con=conn, if_exists='replace', dtype=dtype, index=False)
conn.close()


# comment total
df = pd.read_csv(f"./analysis/csv/total_comment.csv")

# Connect to (create) database.
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "music_id" : "Integer",
    "comment": "TextField",
}
df.to_sql(name='pricelog_musiccommentlog', con=conn, if_exists='replace', dtype=dtype, index=True, index_label="id")
conn.close()