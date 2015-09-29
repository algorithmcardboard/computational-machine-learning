import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def batch_perceptron(X_in, y_in, max_iterations):
    w = np.zeros(X_in.shape[1])
    for t in range(0,max_iterations):
        mis_classification = 0
        tmp = np.dot(X_in, w)
        for i in range(0, X_in.shape[0]):
            if((y_in[i] * tmp[i]) <= 0):
                w += (y_in[i] * X_in[i])
                mis_classification += 1
        print "iteration ", t, " error ",mis_classification 
    return w

def load_iris_data(class_to_reject, class_to_label_as_minus_one, class_to_label_as_one):
    scaler = StandardScaler()
    iris = datasets.load_iris()
    d = iris.data
    t = iris.target
    tmp = np.concatenate((d,t[:, np.newaxis]), axis=1)
    print tmp.shape
    final_data = tmp[np.where(tmp[:,4] != class_to_reject)]
    X = final_data[:, [0, 1, 2, 3]]
    y = final_data[:, 4]
    y[y == class_to_label_as_minus_one] = -1
    y[y == class_to_label_as_one] = 1
    print("mean : %s " % X.mean(axis=0))
    print("standard deviation : %s " % X.std(axis=0))
    scaler.fit(X)
    X = scaler.transform(X)
    print(X.shape)
    print("mean : %s " % X.mean(axis=0))
    print("standard deviation : %s " % X.std(axis=0))
    #seed for consistency
    rng = np.random.RandomState(7041989)
    permutation = rng.permutation(len(X))
    X, y = X[permutation], y[permutation]
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=1999)
    return train_X, test_X, train_y, test_y

train_X, test_X, train_y, test_y = load_iris_data(0, 1, 2)
batch_perceptron(train_X, train_y, 1000)
