import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import random
import pandas as pd
import sqlite3

feature_columns = ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
MAX_CLUSTERS = 2
cmap = cm.get_cmap('tab10', MAX_CLUSTERS)


def get_processed_data(db):
    """
    @param db - path to database
    @returns unlabeled data 
    """
    conn = sqlite3.connect(db)
    data = pd.read_sql_query("Select Delta_T, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount , Class from transactions;", conn)
    labels = data.loc[:,['Class']]
    data = data.loc[:, ['Delta_T', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
  
    data = data.to_numpy()
    # data = np.squeeze(data,axis=1)
    return data, labels

def elbow_point_plot(cluster, errors):
    """
    This function helps create a plot representing the tradeoff between the
    number of clusters and the inertia values.

    :param cluster: 1D np array that represents K (the number of clusters)
    :param errors: 1D np array that represents the inertia values
    """
    plt.clf()
    plt.plot(cluster, errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('elbow_plot')
    plt.savefig(output_dir + "/elbow_plot.png")
    plt.show()


def min_max_scale(data):
    """
    Pre-processes the data by performing MinMax scaling.

    :param data: 2D numpy array of raw data
    :return: preprocessed data
    """
    num_rows, num_cols = np.shape(data)

    for cIdx in range(num_cols):
        feature = data[:,cIdx]
        x_min = np.min(feature)
        x_max = np.max(feature)
        for rIdx in range(num_rows):
            cur_pt = data[rIdx][cIdx]
            data[rIdx][cIdx] = (cur_pt - x_min) / (x_max - x_min)
    return data
    

def plot_clusters(data, cluster_centroids, centroid_indices):
    #Getting unique labels
    u_labels = np.unique(centroid_indices)

    #plotting the results:
    for i in u_labels:
        plt.scatter(data[centroid_indices == i , 0] , data[centroid_indices == i , 1] , label = i)
    plt.scatter(cluster_centroids[:,0] , cluster_centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()

def sk_learn_cluster(X, K):
    """
    Performs k-means clustering using library functions (scikit-learn). You can
    experiment with different initialization settings, but please initialize
    without any optional arguments (other than n_clusters) before submitting.

    :param X: 2D np array containing features of the songs
    :param K: number of clusters
    :return: a tuple of (cluster centroids, indices for each data point)
    """
    # TODO:
    kmeans = KMeans(n_clusters=K).fit(X)
    cluster_centroids = kmeans.cluster_centers_
    centroid_indices = kmeans.labels_

    return (cluster_centroids, centroid_indices)


def cluster_songs(music_data, labels, max_iters=300):
    """
    Performs k-means clustering on the provided music data. Here is where you
    will visualize the raw data, use elbow_point_plot to determine the
    optimal K, and run k-means with this value of K. Follow the TODO comments
    below for more details.

    You should return 4 things from this function (documented below)

    :param music_data: 2D numpy array of music data
    :param max_iters: max number of iterations to run k-means
    :return:
        centroids: calculated cluster centroids from YOUR k-means object
        idx: centroid indices for each data point from YOUR k-means object
        centroids_sklearn: calculated cluster centroids from the sklearn k-means object
        idx_sklearn: centroid indices for each data point from the sklearn k-means object
    """

    centroids_sklearn = np.zeros((MAX_CLUSTERS,len(feature_columns)))
    idx_sklearn = np.zeros((len(music_data),len(feature_columns)))

    # TODO: Perform MinMax scaling on the music_data
    scaled_data = min_max_scale(music_data)
    
    optimal_k = 2
    
    lib_centroids, lib_centroid_indices = sk_learn_cluster(scaled_data, optimal_k)
    inertia_array = calculate_WSS(scaled_data, 10)
    k_array = [1,2,3,4,5,6,7,8,9,10]
    plot_clusters(scaled_data, lib_centroids, lib_centroid_indices)
    print("k_array", k_array)
    print("inertia array", inertia_array)
    elbow_point_plot(k_array,inertia_array)
    print("lib centroids", lib_centroids)
    print("lib centroid_indices", lib_centroid_indices)

    class_0 = lib_centroid_indices[lib_centroid_indices==0]
    class_1 = lib_centroid_indices[lib_centroid_indices==1]
    print('class_0', class_0)
    print('class_1', class_1)
    print('class_1 len', len(class_1))
    print('class_0 len', len(class_0))
    
    return centroids_sklearn, idx_sklearn


def calculate_WSS(data, kmax):
  sse = []
  for k in range(1, kmax+1):
    centroids, indices =  sk_learn_cluster(data, k)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(data)):
      curr_center = centroids[indices[i]]
      curr_sse += (data[i, 0] - curr_center[0]) ** 2 + (data[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

def main():
    """
    Main function for running song clustering. You should not have to change
    anything in this function.
    """
    data, labels = get_processed_data("./data_deliverable/data/transactions.db")
    max_iters = 300  # Number of times the k-mean should run

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    centroids_sklearn, idx_sklearn = cluster_songs(data, labels, max_iters=max_iters)


if __name__ == '__main__':
    output_dir = "./final_deliverable/visualizations/kmeans"

    main()
