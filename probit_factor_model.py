import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import time
from itertools import count

#data = make_classification(n_samples=30000, n_features=50**2, n_informative=15, n_redundant=10, n_classes=2, random_state=123)
data = make_classification(n_samples=5*30000, n_features=50, n_informative=15, n_redundant=10, n_classes=2, random_state=123)
X, y = data


# #for opt in "sag,saga,lbfgs,liblinear,newton-cg".split(","):
# for opt in "lbfgs,sag,saga,newton-cg".split(","):
#
#     cur_time = 0
#     for i in range(10):
log_reg = LogisticRegression(solver="lbfgs", random_state=123)
#
start = time.time()
log_reg.fit(X, y)
print(time.time()-start)
# cur_time = time.time() - start
# print("Time for {} method is {}".format(opt,cur_time))
#
# preds = log_reg.predict(X)
#
# def probit_solver(B0,X):
#
#     return np.linalg.solve(B0 + X.T @ X,X.T)


def sigmoid(x):

    return 1 / (1 + np.exp(-x))



## from https://github.com/cbernet/python-scripts/blob/master/SKLearn/linear_model/LogisticRegression.py
def newtonIteration(x, y, theta, weights, lambdapar=0.0001):
    hyptheta = sigmoid(x.dot(theta))
    zs = weights * (y - hyptheta)

    # gradient of the log likelihood
    gradll = x.T.dot(zs) - lambdapar * theta

    # D matrix
    D = -weights * hyptheta * (1 - hyptheta)


    # hessian H = XT D X - Lambda I
    H = x.T  @ (D[:, np.newaxis] * x)
    H[np.arange(H.shape[0]), np.arange(H.shape[1])] -= lambdapar

    # inverse of the hessian ***************
    # if things are done correctly, that's where
    # we should be spending our time.
    HinvGrad = np.linalg.solve(H, gradll)

    theta = theta - HinvGrad
    return theta


def train_logistic(X, y):

    theta = np.linalg.solve((X.T @ X), X.T @ y)

    weights = np.ones(y.shape)

    for i in count():

        new_theta = newtonIteration(X, y, theta, weights)
        w_change = np.linalg.norm(new_theta - theta)
        theta = new_theta

        if w_change < 1e-5:
            #print("Converged in {} iterations".format(i))
            return theta

        if i > 1e2:
            print("Failed to converge")
            return theta

thetas = train_logistic(X, y)

cur_time = 0
for i in range(100):
    #log_reg = LogisticRegression(solver=opt, random_state=123)
    start = time.time()
    thetas = train_logistic(X, y)

    #log_reg.fit(X, y)
    cur_time += time.time()-start
print("Time is {}".format(cur_time/100))
