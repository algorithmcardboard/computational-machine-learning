##############################################################
# Author: Anirudhan J Rajagopalan
# email: ajr619@nyu.edu
# Collaborators: Michele Ceru
##############################################################
##############################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

MAX_ITER = 10000

def batch_perceptron(X_in, y_in, max_iterations, w, label):
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
        # It is safe to exit the iteration once mis_classification error is zero.  Because we have already went through all the 'm' data points available
        if(mis_classification == 0):
            break
    plt.plot(errors, 'r-');
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Perceptron with " + label)
    plt.show()
    return w, errors

def modified_batch_perceptron(X_in, y_in, eta, max_iterations, w, label):
    #w = np.zeros(X_in.shape[1])
    errors = []
    for t in range(0,max_iterations):
        mis_classification = 0
        tmp = np.dot(X_in, w)
        for i in range(0, X_in.shape[0]):
            if((y_in[i] * tmp[i]) <= 0):
                w += (eta * y_in[i] * X_in[i])
                mis_classification += 1
        print "iteration ", t, " error ",mis_classification 
        errors.append(mis_classification)
        if(mis_classification == 0):
            break
    plt.plot(errors, 'r-');
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Perceptron with " + label)
    plt.show()
    return w, errors

def load_iris_data(class_to_reject, class_to_label_as_minus_one, class_to_label_as_one):
    # This random number plays a huge role as it determines the split and shuffle of the data.
    randState = 7021947 # perceptron converges in some finite iterations.
    #randState = 21947 # doesnt converge even for 100000 iterations
    # A perceptron sometimes doesn't converge it seems. 
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


#plot the dataset.  3D for better visualization.  the 4th dimension is still missing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plot features 0, 1, 2
features = [0, 1, 2]
colors = ["darkblue", "darkgreen"]
idx = np.where(train_y == -1)
ax.scatter(train_X[idx, features[0]], train_X[idx, features[1]], train_X[idx, features[2]], color=colors[0], label="Class %s -1")
idx = np.where(train_y == 1)
ax.scatter(train_X[idx, features[0]], train_X[idx, features[1]], train_X[idx, features[2]], color=colors[1], label="Class %s 1")
plt.show()
# End of training set plot

w1, errors = batch_perceptron(train_X, train_y, MAX_ITER, np.zeros(train_X.shape[1]), 'Training data')
# Ideally you would want to test your learning parameter in your test data.  Here we are just running the algorithm against the test data and we might end up modifying w too.
#when you pass the w obtained before to the batch_perceptron algorithm, the algorithm converges really fast as we have already found a plane seperating the data points
#w2, n_errors = batch_perceptron(test_X, test_y, MAX_ITER, 'go', w1, 'Test data')
w2, n_errors = batch_perceptron(test_X, test_y, MAX_ITER, np.zeros(test_X.shape[1]), 'Test data')

plt.plot(errors, 'r-');
plt.plot(n_errors, 'b-');
plt.title("Combined plot of train and test")
plt.show()
print "W1 is ", w1, " w2 is ", w2

##############################################################
# Observations:
#   The perceptron algorithm may (or) may not converge depending on the split of the data.
#   The mis-classification error count goes down for most iterations when perceptron runs on a dataset that will converge
#   There are observable oscillation in mis-classification error when the perceptron does not converge
#   In Modified batch perceptron, changing eta changes the magnitude of the learning parameter w.  It doesn't change its direction
#   When eta = 1, modified batch perceptron and batch perceptron give the same results.
#   When eta is non zero, the magnitude of learning parameter is eta times the learning parameter got with batch_perceptron algorithm
##############################################################

#ETA = 1 # Same as batch_perceptron
#ETA = 0.5 # Iterations remain unchanged.  Same as batch perceptron.  changing eta changes the magnitude and not the direction
ETA = 1.5 # Same as batch_perceptron
w3, m_errors = modified_batch_perceptron(train_X, train_y, ETA, MAX_ITER, np.zeros(train_X.shape[1]), 'Training data')
w4, n_m_errors = modified_batch_perceptron(test_X, test_y, ETA, MAX_ITER, w3, 'Test data')

plt.plot(m_errors, 'r-');
plt.plot(n_m_errors, 'b-');
plt.title("Modified Perceptron: Combined plot of train and test")
plt.show()

print "W3 is ", w3, " w4 is ", w4
