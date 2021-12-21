import numpy as np

def calc_cost(X, Y, theta, TYPE='lin'):

    m = len(Y)
    if TYPE == 'lin':
        HYPOTHESIS = X @ theta
        COST = (0.5/m) * np.sum(np.square(HYPOTHESIS - Y), axis=0)    #alternatively 
                                            #(HYPOTHESIS - Y).T @ (HYPOTHESIS - Y)
    else:
        HYPOTHESIS = 1/(1 + np.exp(-X @ theta))
        COST = (-1/m) * (Y.T @ np.log(HYPOTHESIS) + (1 - Y).T @ np.log(1 - HYPOTHESIS))
    
    return COST, HYPOTHESIS

def grad_descent(X, Y, theta, ALPHA = 0.001, ITERS = 10000, TOL = 1E-3, TYPE='lin'):

    m = len(Y)
    COST, HYPOTHESIS = calc_cost(X, Y, theta, TYPE)
    COST = [np.asscalar(COST)]
    
    for i in range(ITERS):
        SLOPE = (ALPHA/m)*(HYPOTHESIS - Y).T @ X
        theta = theta - SLOPE.T
        COST_new, HYPOTHESIS = calc_cost(X, Y, theta, TYPE)
        COST.append(np.asscalar(COST_new))
        COST_DIFF = abs(COST[i+1] - COST[i])
        if COST_DIFF < TOL:
            break

    return theta, COST

def normal_eqn(X, Y):
    mat_matT = X.T @ X
    mat_matT_inv = np.linalg.pinv(mat_matT)
    theta = (mat_matT_inv @ X.T) @ Y
    return theta

def scale_feature(X):
    X_copy = X.copy()
    MEANMAT = np.mean(X_copy[:,1:], axis=0)
    X_copy[:,1:] = X_copy[:,1:] - MEANMAT
    STDMAT = np.std(X_copy[:,1:], axis=0)
    X_copy[:,1:] = X_copy[:,1:]/STDMAT
    return X_copy, MEANMAT, STDMAT

