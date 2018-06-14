import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import time
from itertools import count

import torch



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



def train_logistic_torch(X, y, b = 0, thetas= None,reg=1e-3):
    # print("Mean y in train logistic is {}".format(torch.mean(y)))
    if thetas is None:
        theta, _ = torch.gesv((X.t() @ y).view(-1, 1), X.t() @ X)
        if (theta != theta).any():
            raise ValueError("NaN in logistic init")
    else:
        theta=thetas.clone()
    theta = theta.view(-1)

    weights = torch.FloatTensor(torch.ones(y.shape)).cuda()
    #print((torch.round(torch.sigmoid(X @ theta)) == y).sum())

    for i in count():

        new_theta = newtonIteration_torch(X, y, b, theta, weights,lambdapar=reg)
        w_change = (new_theta - theta).norm()
        theta = new_theta
        #print((torch.round(torch.sigmoid(X @ theta)) == y).sum())
        if (theta!=theta).any():
            raise ValueError("NaN in logistic")
        if w_change < 1e-4:
            # print("Converged in {} iterations".format(i))
            return theta
        if i > 10:
            print("Failed to converge")
            return theta


def newtonIteration_torch(x, y, b, theta, weights, lambdapar=0):

    #assert (y >= 0).all() and (y <= 1).all(), "Wrong y values"
    #print("Mean y is {}".format(np.mean(y.cpu().numpy())))


    hyptheta = torch.sigmoid(x @ theta + b).view(-1)

    zs = weights * (y - hyptheta)
    # gradient of the log likelihood

    gradll = (x.t() @ zs).view(-1) - lambdapar * theta
    # D matrix
    D = -weights * hyptheta * (1 - hyptheta)

    # hessian H = XT D X - Lambda I
    H = x.t()  @ (D[:, np.newaxis] * x)
    H[np.arange(H.shape[0]), np.arange(H.shape[1])] -= lambdapar
    # inverse of the hessian ***************
    # if things are done correctly, that's where
    # we should be spending our time.
    #HinvGrad = np.linalg.solve(H, gradll)
    HinvGrad, _ = torch.gesv(gradll.view(-1, 1), H)
    theta = theta - HinvGrad.view(-1)
    #print("Mean theta is {}".format(np.mean(np.abs(theta.cpu().numpy()))))
    return theta


def train_linear_torch(X, y, b = 0, weights= None, thetas= None, reg=0.005):

    try:
        if weights is not None:
            theta, _ = torch.gesv((X.t() @ (weights * (y - b))).view(-1, 1), X.t() * weights @ X + reg * torch.eye(X.shape[1]).cuda())
        else:
            theta, _ = torch.gesv((X.t() @ (y - b)).view(-1, 1), X.t() @ X + reg * torch.eye(X.shape[1]).cuda())
    except:
        print(X.t() @ X, X)
        raise ValueError()

    if (theta != theta).any():
        raise ValueError("NaN in linear")
    return theta




if __name__ == "__main__":


    data = make_classification(n_samples=20*30000, n_features=50, n_informative=15, n_redundant=0, n_classes=2)
    X, y = data

    # #for opt in "sag,saga,lbfgs,liblinear,newton-cg".split(","):
    # for opt in "lbfgs,sag,saga,newton-cg".split(","):
    #
    #     cur_time = 0
    #     for i in range(10):
    #       log_reg = LogisticRegression(solver=opt, random_state=123)
    #
    #       start = time.time()
    #       log_reg.fit(X, y)
    #    print(time.time()-start)
    # cur_time = time.time() - start
    # print("Time for {} method is {}".format(opt,cur_time))
    #
    # preds = log_reg.predict(X)
    #
    # def probit_solver(B0,X):
    #
    #     return np.linalg.solve(B0 + X.T @ X,X.T)
    #data = make_classification(n_samples=30000, n_features=50**2, n_informative=15, n_redundant=10, n_classes=2, random_state=123)


    # cur_time = 0
    # for i in range(10):
    #     #log_reg = LogisticRegression(solver=opt, random_state=123)
    #     start = time.time()
    #     thetas = train_logistic(X, y)
    #
    #     #log_reg.fit(X, y)
    #     cur_time += time.time()-start
    # print("Time is {}".format(cur_time/100))


    Xt = torch.FloatTensor(X).cuda()
    yt = torch.FloatTensor(y).cuda()

    # cur_time = 0
    # for i in range(10):
    #     #log_reg = LogisticRegression(solver=opt, random_state=123)
    #     start = time.time()
    #     thetas_np = train_logistic(X, y)
    #     #thetas = train_logistic_torch(Xt, yt)
    #
    #     #log_reg.fit(X, y)
    #     cur_time += time.time()-start
    # print("Time is {}".format(cur_time/10))
    cur_time = 0
    pre = torch.gesv(Xt.t() , Xt.t() @ Xt)[0]
    cov = Xt.t() @ Xt
    for i in range(10):
        #log_reg = LogisticRegression(solver=opt, random_state=123)
        start = time.time()
        # thetas_np = train_logistic(X, y)
        theta = torch.gesv((Xt.t() @ yt).view(-1,1), cov)
        #log_reg.fit(X, y)
        cur_time += time.time()-start
    print("Time is {}".format(cur_time/10))
    cur_time = 0
    for i in range(10):
        #log_reg = LogisticRegression(solver=opt, random_state=123)
        start = time.time()
        # thetas_np = train_logistic(X, y)
        thetas = train_logistic_torch(Xt, yt)
        #log_reg.fit(X, y)
        cur_time += time.time()-start
   # print(thetas)
    print("Time is {}".format(cur_time/10))
    # cur_time = 0
    # for i in range(10):
    #     #log_reg = LogisticRegression(solver=opt, random_state=123)
    #     start = time.time()
    #     # thetas_np = train_logistic(X, y)
    #     thetas = train_logistic_torch(Xt, yt)
    #
    #     #log_reg.fit(X, y)
    #     cur_time += time.time()-start
    # print("Time is {}".format(cur_time/10))




    ## Test the optimization

    b = np.array([0.3, -0.7, 0.5])[:, None]

    N = 10000

    X = np.random.normal(size=(N, 2))
    X = np.hstack([np.ones(N)[:, None], X])

    y = X @ b #+ np.random.normal(size=X.shape[0])[:, None]

    Xt = torch.FloatTensor(X).cuda()
    yt = torch.sigmoid(torch.FloatTensor(y).cuda())

    res = train_logistic_torch(Xt, yt.view(-1))