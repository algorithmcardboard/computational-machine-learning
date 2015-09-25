import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

iris = datasets.load_iris()
d = iris.data
t = iris.target

tmp = np.concatenate((d,t[:, np.newaxis]), axis=1)

print tmp.shape

final_data = tmp[np.where(tmp[:,4] != 2)]

X = final_data[:, [0,1, 2, 3]]
Y = final_data[:, 4]

print("mean : %s " % X.mean(axis=0))
print("standard deviation : %s " % X.std(axis=0))

scaler.fit(X)

X_scaled = scaler.transform(X)
print(X_scaled.shape)

print("mean : %s " % X_scaled.mean(axis=0))
print("standard deviation : %s " % X_scaled.std(axis=0))
