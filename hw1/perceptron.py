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
        tmp = np.dot(X_in, w)
        for i in range(0, X_in.shape[0]):
            if((y_in[i] * tmp[i]) <= 0):
                w += (y_in[0] * X_in[0])
                break
        print w
    return w



scaler = StandardScaler()

iris = datasets.load_iris()
d = iris.data
t = iris.target

tmp = np.concatenate((d,t[:, np.newaxis]), axis=1)

print tmp.shape

final_data = tmp[np.where(tmp[:,4] != 2)]

X = final_data[:, [0,1, 2, 3]]
y = final_data[:, 4]

print("mean : %s " % X.mean(axis=0))
print("standard deviation : %s " % X.std(axis=0))

scaler.fit(X)

X = scaler.transform(X)
print(X.shape)

print("mean : %s " % X.mean(axis=0))
print("standard deviation : %s " % X.std(axis=0))

#seed for consistency
rng = np.random.RandomState(7041989)

#print(y)

permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=1999)

print train_X

y[y==0] = -1 #Change zeros to -1 for balance.


