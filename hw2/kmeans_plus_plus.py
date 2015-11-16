from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

def mykmeans_plus_plus(X, max_clusters = 8, max_iterations=10000):
    print "Calling my-kmeans function"

    randState = 10312003
    rnd = np.random.RandomState(randState)
    #rnd = np.random.RandomState(len(X))[:max_clusters]

    centroids = np.empty(shape=(max_clusters, X.shape[1]))
    centroids[0] = X[rnd.permutation(len(X))[0]]

    for itr in range(1, max_clusters):

        d2weighting = pairwise_distances(X, centroids[:itr], metric='euclidean')
        d2weighting = np.power(np.amin(d2weighting, axis=1),2)
        d2weighting = d2weighting/np.sum(d2weighting)

        index = np.argmax(d2weighting)
        centroids[itr] = X[index + itr - 1]
    
    print centroids

    distortion_plot = plt.figure()
    d_ax = distortion_plot.add_subplot(111)

    centroid_plot = plt.figure()
    c_ax = centroid_plot.add_subplot(111)

    for nIter in range(0, max_iterations):
        for center in centroids:
            c_ax.plot(center[0], center[1],'r-')
        distances = pairwise_distances(X, centroids, metric='euclidean')
        clusters = np.argmin(distances,axis=1)
        min_distances = np.amin(distances, axis=1)
        sum_distortion = min_distances.sum()
        d_ax.plot(nIter, sum_distortion)
        data = np.concatenate([X, clusters[:,np.newaxis]], axis=1)
        for cRange in range(0,max_clusters):
            allpoints = data[np.where(data[:,(data.shape[1] - 1)] == cRange)][:,range(0, data.shape[1] -1)]
            centroids[cRange] = np.sum(allpoints, axis=0)/allpoints.shape[0]
        print "Iteration {0}: Distortion {1}".format(nIter, sum_distortion)
        d_ax.scatter(nIter, sum_distortion)

    distortion_plot.savefig('plus_distortion_plot.png')
    centroid_plot.savefig('plus_centroid_plot.png')
    return centroids

if __name__ == '__main__':

    scaler = StandardScaler()

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    scaler.fit(X)
    X_scaled = scaler.transform(X)
    mykmeans_plus_plus(X_scaled, max_clusters=3, max_iterations=100)
