import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def batch_perceptron(X_in, y_in, max_iterations, display_format, w):
    #w = np.zeros(X_in.shape[1])
    errors = []
    for t in range(0,max_iterations):
        mis_classification = 0
        tmp = np.dot(X_in, w)
        for i in range(0, X_in.shape[0]):
            if((y_in[i] * tmp[i]) <= 0):
                w += (y_in[i] * X_in[i])
                mis_classification += 1
        print "iteration ", t, " error ",mis_classification 
        errors.append(mis_classification)
        if(mis_classification == 0):
            break
    plt.plot(errors, 'r-');
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    return w, errors

def load_iris_data(class_to_reject, class_to_label_as_minus_one, class_to_label_as_one):
    randState = 7021947
    #randState = 21947 #doesnt converge even for 100000 iterations
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
    X_scaled = scaler.transform(X)
    X = X_scaled
    print(X.shape)
    print("mean : %s " % X.mean(axis=0))
    print("standard deviation : %s " % X.std(axis=0))
    #seed for consistency
    rng = np.random.RandomState(randState)
    permutation = rng.permutation(len(X))
    X, y = X[permutation], y[permutation]
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=randState)
    return train_X, test_X, train_y, test_y

train_X, test_X, train_y, test_y = load_iris_data(0, 1, 2) #takes time to converge.  Depends on random seed
#train_X, test_X, train_y, test_y = load_iris_data(2, 0, 1) # converges very fast. Really fast.
#train_X, test_X, train_y, test_y = load_iris_data(1, 2, 0) # converges very fast. This one too


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ["darkblue", "darkgreen"]
idx = np.where(train_y == -1)
ax.scatter(train_X[idx, 0], train_X[idx, 1], train_X[idx, 2], color=colors[0], label="Class %s -1")
idx = np.where(train_y == 1)
ax.scatter(train_X[idx, 0], train_X[idx, 1], train_X[idx, 2], color=colors[1], label="Class %s 1")

plt.show()

w, errors = batch_perceptron(train_X, train_y, 10000, 'ro', np.zeros(train_X.shape[1]))
w, n_errors = batch_perceptron(test_X, test_y, 10000, 'go', w)

plt.plot(errors, 'r-');
plt.plot(n_errors, 'b-');
plt.show()
