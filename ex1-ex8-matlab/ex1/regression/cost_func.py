import numpy as np

def calc_cost(X, Y, theta):

    m = 2*len(Y)
    HYPOTHESIS = X @ theta
    COST = (1/m) * np.sum(np.square(HYPOTHESIS - Y), axis=0)
    return COST

def grad_descent(X, Y, theta, ALPHA = 0.001, ITERS = 10000, TOL = 1E-3):

    m = 2*len(Y)
    COST = [calc_cost(X, Y, theta)]

    for i in range(ITERS):
        HYPOTHESIS = X @ theta
        SLOPE = (ALPHA/m)*(HYPOTHESIS - Y).T @ X
        theta = theta - SLOPE.T
        COST.append(calc_cost(X, Y, theta))
        COST_DIFF = abs(COST[i+1] - COST[i])
        if max(COST_DIFF) < TOL:
            break

    COST = np.asmatrix(COST)
    return theta, COST

def normal_eqn(X, Y):
    mat_matT = X.T @ X
    mat_matT_inv = np.linalg.pinv(mat_matT)
    theta = (mat_matT_inv @ X.T) @ Y
    return theta

def scale_feature(X):
    X_copy = X.copy()
    X_copy[:,1:] = X_copy[:,1:] - np.mean(X_copy[:,1:], axis=0)
    X_copy[:,1:] = X_copy[:,1:]/np.std(X_copy[:,1:], axis=0)
    return X_copy
