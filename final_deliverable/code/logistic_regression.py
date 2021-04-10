import numpy as np
from sklearn.linear_model import LogisticRegression
import sqlite3 
import pandas as pd

def pre_process(db):
    """
    @param db - path to database
    @returns train_inputs, train_labels, test_inputs, test_labels
    """
    conn = sqlite3.connect(db);
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())
    data = pd.read_sql_query("""Select (Delta_T, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, 
                                        V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount , Class) from transactions;""", conn)
    train_split = int(0.8*len(data))
    train = data[0:train_split]
    test = data[train_split:len(data)]
    train_x = train.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    train_y = train.loc[:, ['Class']]
    test_x = test.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    test_y = test.loc[:, ['Class']]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    return train_x,train_y,test_x,test_y

def main():
    train_x,train_y,test_x,test_y = pre_process("./data_deliverable/transactions.db")
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)


if __name__ == '__main__':
    main()
