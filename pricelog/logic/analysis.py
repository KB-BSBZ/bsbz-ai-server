import pandas as pd
import sqlite3
import numpy as np

database = "db.sqlite3"
conn = sqlite3.connect(database, check_same_thread=False)
np.random.seed(42)

product_id = 1



for i in range(1, 11):
    df1 = pd.read_sql(f"SELECT * FROM pricelog_estatelog WHERE product_id = {i}", conn)
    print(i)
    
    print(df1['area'].value_counts())
    
    print(df1['area'].value_counts().keys()[0])
    
    df1['area'].value_counts().keys()[0] > 70 and df1['area'].value_counts().keys()[0] < 90
    
# 1. 84.705, 84.751
# 2. 