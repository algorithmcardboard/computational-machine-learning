from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

def kmeans_raw(X, permutation, max_clusters = 8, max_iterations = 100, plot=False):
    #rnd = np.random.RandomState(len(X))[:max_clusters]
    print "Permutations are ", permutation

    centroids = X[permutation]

    if plot:
        distortion_plot = plt.figure()
        d_ax = distortion_plot.add_subplot(111)

        centroid_plot = plt.figure()
        c_ax = centroid_plot.add_subplot(111)

    sum_distortion = 0

    for nIter in range(0, max_iterations):
        if plot:
            for center in centroids:
                c_ax.plot(center[0], center[1],'r-')
        distances = pairwise_distances(X, centroids, metric='euclidean')
        clusters = np.argmin(distances,axis=1)
        min_distances = np.amin(distances, axis=1)
        sum_distortion = min_distances.sum()
        data = np.concatenate([X, clusters[:,np.newaxis]], axis=1)
        for cRange in range(0,max_clusters):
            allpoints = data[np.where(data[:,(data.shape[1] - 1)] == cRange)][:,range(0, data.shape[1] -1)]
            centroids[cRange] = np.sum(allpoints, axis=0)/allpoints.shape[0]
        print "Iteration {0}: Distortion {1}".format(nIter, sum_distortion)
        if plot:
            d_ax.scatter(nIter, sum_distortion)

    if plot:
        distortion_plot.savefig('distortion_plot_multi.png')
        centroid_plot.savefig('centroid_plot_multi.png')
    return centroids, sum_distortion

def mykmeans_multi(X, max_clusters = 8, max_iterations=100):
    min_distortion = float("inf")
    min_permutation = None

    for outIter in range(0, max_iterations):
        print "Iterating the outer loop for {0}".format(outIter)
        # Dont use a seed value as we have to generate multiple permutations
        randState = 10312003
        rnd = np.random.RandomState(randState)
        permutation = rnd.permutation(len(X))[:max_clusters]

        centroids, distortion = kmeans_raw(X, permutation, max_clusters=max_clusters, max_iterations=max_iterations, plot=False)
        if(distortion < min_distortion):
            min_distortion = distortion
            min_centroids  = centroids
            min_permutation = permutation

    centroids, distortion = kmeans_raw(X, min_permutation, max_clusters=max_clusters, max_iterations=max_iterations, plot=True)
    return centroids

if __name__ == '__main__':

    scaler = StandardScaler()

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    scaler.fit(X)
    X_scaled = scaler.transform(X)
    mykmeans_multi(X_scaled, max_clusters=3, max_iterations=100)
