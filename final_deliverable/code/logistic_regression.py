import numpy as np
from sklearn.linear_model import LogisticRegression
import sqlite3 
import pandas as pd
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
    train_x = train.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    train_y = train.loc[:, ['Class']]
    test_x = test.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    test_y = test.loc[:, ['Class']]
    train_x = train_x.to_numpy()
    train_y = np.squeeze(train_y.to_numpy(),axis=1)
    test_x = test_x.to_numpy()
    test_y = np.squeeze(test_y.to_numpy(),axis=1)
    return train_x,train_y,test_x,test_y

def run_regression(train_x,train_y,num_iter):
    clf = LogisticRegression(max_iter=num_iter).fit(X=train_x,y=train_y)
    return clf

def get_accuracy(model, test_x,test_y):
    predictions = model.predict(test_x)
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
    return  one_counter/one_big_counter, zero_counter/zero_big_counter




def create_val_plots(x_vals, vals_zeros,vals_ones):
    """
    method to create a plot for the training loss per epoch, and validation loss for the cross-validation scheme
    @param vals_zeros - array of training losses (1 entry per epoch)
    @param vals_ones - array of validation losses (1 entry per epoch)
    """

    # z = [i for i in range(len(vals_ones[0]))]

    plt.plot(x_vals, vals_zeros[k],label="non-fraud")
    plt.plot(x_vals, vals_ones[k],label="fraud")
    plt.title('Accuracy per number of iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(100, 210, 10))
    plt.legend() 
    # plt.savefig('../results/val_plot.png')



def main():
    train_x,train_y,test_x,test_y = pre_process("./data_deliverable/data/transactions.db")
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    # arr = [1,10,50,100,150,200,250]
    new_arr = [100,110,120,130,140,150,160,170,180,190,200]
    # f_arr = [160,162,164,166,168,170]
    optimized_value = 165
    vals_zeros = []
    vals_ones = []
    for i in new_arr:
        model = run_regression(train_x,train_y,i)
        accuracy = get_accuracy(model, test_x,test_y)
        vals_ones.append(accuracy[0])
        vals_zeros.append(accuracy[1])
        print("Number of iterations: ",i,", Accuracy: ", accuracy)

    create_val_plots(new_arr, vals_zeros, vals_ones)


if __name__ == '__main__':
    main()
