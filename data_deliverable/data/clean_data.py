import pandas as pd
import sqlite3

data = pd.read_csv ('./data_deliverable/data/creditcard.csv')   
df = pd.DataFrame(data, columns= ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 
                                    'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
                                    'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'])

conn = sqlite3.connect('./data_deliverable/data/transactions.db')
cursor = conn.cursor()
conn.execute('''CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, Delta_T INTEGER NOT NULL, V1 FLOAT NOT NULL, V2 FLOAT NOT NULL, V3 FLOAT NOT NULL, V4 FLOAT NOT NULL, V5 FLOAT NOT NULL,
                                                        V6 FLOAT NOT NULL, V7 FLOAT NOT NULL, V8 FLOAT NOT NULL, V9 FLOAT NOT NULL, V10 FLOAT NOT NULL, V11 FLOAT NOT NULL, V12 FLOAT NOT NULL, V13 FLOAT NOT NULL,
                                                        V14 FLOAT NOT NULL, V15 FLOAT NOT NULL, V16 FLOAT NOT NULL, V17 FLOAT NOT NULL, V18 FLOAT NOT NULL, V19 FLOAT NOT NULL, V20 FLOAT NOT NULL,
                                                        V21 FLOAT NOT NULL, V22 FLOAT NOT NULL, V23 FLOAT NOT NULL, V24 FLOAT NOT NULL, V25 FLOAT NOT NULL, V26 FLOAT NOT NULL, V27 FLOAT NOT NULL, V28 FLOAT NOT NULL,
                                                        Amount FLOAT NOT NULL, Class INTEGER NOT NULL)''')

for row in df.itertuples():
    conn.execute('''
        INSERT INTO transactions (Delta_T, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17,
                                    V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount , Class)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''',
        (row.Time, row.V1, row.V2, row.V3, row.V4, row.V5, row.V6, row.V7, row.V8, row.V9, row.V10, row.V11, row.V12, row.V13,
        row.V14, row.V15, row.V16, row.V17,row.V18, row.V19, row.V20, row.V21, row.V22, row.V23, row.V24, row.V25,
        row.V26, row.V27, row.V28, row.Amount , row.Class)
        )
conn.commit()

df = pd.read_sql_query("select * from transactions limit 100;", conn)
# df.to_csv('example.csv')
print(df)
cursor.close()
conn.close()
    