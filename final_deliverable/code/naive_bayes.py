import numpy as np
from sklearn.naive_bayes import GaussianNB,CategoricalNB
import sqlite3 
import pandas as pd
import random
import matplotlib.pyplot as plt

def pre_process(db):
    """
    @param db - path to database
    @returns train_inputs, train_labels, test_inputs, test_labels
    """
    conn = sqlite3.connect(db)
    data = pd.read_sql_query("Select Delta_T, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount , Class from transactions;", conn)
    train_split = int(0.8*len(data))
    train = data[0:train_split]
    test = data[train_split:len(data)]
    train_x = train.loc[:, ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    train_y = train.loc[:, ['Class']]
    test_x = test.loc[:, ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    test_y = test.loc[:, ['Class']]
    train_x = train_x.to_numpy()
    train_y = np.squeeze(train_y.to_numpy(),axis=1)
    test_x = test_x.to_numpy()
    test_y = np.squeeze(test_y.to_numpy(),axis=1)
    plot_data_dist(train_y,test_y)
    return train_x,train_y,test_x,test_y

def plot_data_dist(train_y,test_y):
    unique, counts = np.unique(train_y, return_counts=True)
    plt.bar(unique, np.log(counts))
    plt.title('Class Frequency for training data')
    plt.xlabel('Class')
    plt.ylabel('Log Value')
    plt.show()
    unique, counts = np.unique(test_y, return_counts=True)
    plt.bar(unique,  np.log(counts))
    plt.title('Class Frequency for testing data')
    plt.xlabel('Class')
    plt.ylabel('Log Value')
    plt.show()

def run_NB(train_x,train_y):
    clf = GaussianNB().fit(X=train_x,y=train_y)
    return clf

def get_accuracy(model, test_x,test_y):
    predictions = model.predict(test_x)
    overall_acc = np.mean(predictions==test_y)
    zero_big_counter = 0
    zero_counter = 0
    for x in range(len(test_y)):
        if test_y[x]==0:
            if predictions[x]==0:
                zero_counter += 1
            zero_big_counter += 1
    one_big_counter = 0
    one_counter = 0
    for x in range(len(test_y)):
        if test_y[x]==1:
            if predictions[x]==1:
                one_counter += 1
            one_big_counter += 1
    return  overall_acc, one_counter/one_big_counter, zero_counter/zero_big_counter

def create_val_plots(x_vals, vals_zeros,vals_ones):
    """
    method to create a plot for the training loss per epoch, and validation loss for the cross-validation scheme
    @param vals_zeros - array of training losses (1 entry per epoch)
    @param vals_ones - array of validation losses (1 entry per epoch)
    """
    plt.plot(x_vals, vals_zeros,label="non-fraud")
    plt.plot(x_vals, vals_ones,label="fraud")
    plt.title('Accuracy per number of iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(100, 210, 10))
    plt.legend() 
    plt.show()



def main():
    train_x,train_y,test_x,test_y = pre_process("./data_deliverable/data/transactions.db")
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    model = run_NB(train_x,train_y)
    accuracy = get_accuracy(model, test_x,test_y)
    print("Accuracy: ", accuracy)


if __name__ == '__main__':
    main()
