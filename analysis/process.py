import pandas as pd
import sqlite3


# estate total
df = pd.read_csv(f"analysis\csv\estate_total.csv")

# Connect to (create) database.
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype={
    "product_id": "IntegerField",
    "price": "CharField",
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