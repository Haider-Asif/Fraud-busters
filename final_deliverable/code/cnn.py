import numpy as np
import sqlite3 
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def pre_process(db):
    """
    @param db - path to database
    @returns train_inputs, train_labels, test_inputs, test_labels
    """
    conn = sqlite3.connect(db)
    data = pd.read_sql_query("Select Delta_T, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount , Class from transactions;", conn)
    # random.shuffle(data)
    train_split = int(0.8*len(data))
    train = data[0:train_split]
    test = data[train_split:len(data)]
    train_x = train.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    train_y = train.loc[:, ['Class']]
    test_x = test.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
    test_y = test.loc[:, ['Class']]
    train_x = train_x.to_numpy()
    # train_x = np.expand_dims(train_x,axis=2)
    train_y = np.squeeze(train_y.to_numpy(),axis=1)
    test_x = test_x.to_numpy()
    # test_x = np.expand_dims(test_x,axis=2)
    test_y = np.squeeze(test_y.to_numpy(),axis=1)
    return train_x,train_y,test_x,test_y

def k_cross_validate_model(train_x, train_y, k):
    for i in range(k):
        validation_x = train_x[int(i*(1/k)*train_x.shape[0]):int((i+1)*(1/k)*train_x.shape[0])]
        validation_y = train_y[int(i*(1/k)*train_y.shape[0]):int((i+1)*(1/k)*train_y.shape[0])]
        training_x = np.concatenate((train_x[0:int(i*(1/k)*train_x.shape[0])],train_x[int((i+1)*(1/k)*train_x.shape[0]):train_x.shape[0]]), axis=0)
        training_y = np.concatenate((train_y[0:int(i*(1/k)*train_y.shape[0])],train_y[int((i+1)*(1/k)*train_y.shape[0]):train_y.shape[0]]), axis=0)
        model = tf.keras.Sequential()
        layer_1 = tf.keras.layers.Conv1D(32,3,strides=1,activation=tf.keras.layers.LeakyReLU(0.03))
        max_pool_1 = tf.keras.layers.MaxPool1D(3, strides=1)
        flatten_1 = tf.keras.layers.Flatten()
        dense_1 = tf.keras.layers.Dense(2, activation=tf.keras.layers.Softmax())
        model.add(layer_1)
        model.add(max_pool_1)
        model.add(flatten_1)
        model.add(dense_1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.fit(x=training_x, y=training_y, batch_size=64, epochs=8, validation_data=(validation_x,validation_y), shuffle=True)

def train_model(train_x, train_y):
    """
    Implements and trains the model using a cross-validation scheme with MSE loss
    param train_x: the training inputs
    param train_y: the training labels
    return: a trained model
    """
    model = tf.keras.Sequential()
    layer_1 = tf.keras.layers.Conv1D(16,3,strides=1,activation=tf.keras.layers.LeakyReLU(0.03))
    max_pool_1 = tf.keras.layers.MaxPool1D(3, strides=1)
    flatten_1 = tf.keras.layers.Flatten()
    dense_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(0.03))
    dense_2 = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(0.03))
    dense_3 = tf.keras.layers.Dense(2, activation=tf.keras.layers.Softmax())
    # model.add(layer_1)
    # model.add(max_pool_1)
    # model.add(flatten_1)
    model.add(dense_1)
    model.add(dense_2)
    model.add(dense_3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(x=train_x, y=train_y, batch_size=512, epochs=8)
    create_plots(history.history["loss"], history.history["sparse_categorical_accuracy"])
    return model

def create_plots(training_losses, training_accuracies):
    x = [i for i in range(len(training_losses))]
    plt.plot(x, training_losses)
    plt.title('Training Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    y = [i for i in range(len(training_accuracies))]
    plt.plot(y, training_accuracies)
    plt.title('Training Accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def get_accuracy(model, test_x,test_y):
    predictions = np.argmax(model.predict(test_x),axis=1)
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
    plt.plot(x_vals, vals_zeros,label="non-fraud")
    plt.plot(x_vals, vals_ones,label="fraud")
    plt.title('Accuracy per number of epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,len(x_vals)))
    plt.legend() 
    plt.show()
    # plt.savefig('./analysis_deliverable/visualizations/accuracy_plot.png')



def main():
    np.random.seed(0)
    random.seed(0)
    train_x,train_y,test_x,test_y = pre_process("./data_deliverable/data/transactions.db")
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    # num_epochs = 20
    # vals_zeros = []
    # vals_ones = []
    model = train_model(train_x,train_y)
    accuracy = get_accuracy(model, test_x,test_y)
    print(accuracy)
    # vals_ones.append(accuracy[0])
    # vals_zeros.append(accuracy[1])

    # create_val_plots(np.arange(1,num_epochs+1), vals_zeros, vals_ones)


if __name__ == '__main__':
    main()
