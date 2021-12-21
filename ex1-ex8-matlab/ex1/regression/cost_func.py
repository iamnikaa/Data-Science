import numpy as np

def calc_cost(X, Y, theta, TYPE='lin', LAMBDA=0):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        theta ([type]): [description]
        TYPE (str, optional): [description]. Defaults to 'lin'.
        LAMBDA (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    NUM_OBS = len(Y)
    if TYPE == 'lin':
        HYPOTHESIS = X @ theta
        COST = (0.5/NUM_OBS) * (np.sum(np.square(HYPOTHESIS - Y), axis=0) + LAMBDA*(theta.T @ theta))    #alternatively 
                                            #(HYPOTHESIS - Y).T @ (HYPOTHESIS - Y)
    else:
        HYPOTHESIS = 1/(1 + np.exp(-X @ theta))
        COST = (-1/NUM_OBS) * (Y.T @ np.log(HYPOTHESIS) + (1 - Y).T @ np.log(1 - HYPOTHESIS)) \
               + LAMBDA * (0.5/NUM_OBS) * (theta.T @ theta)
    
    return COST, HYPOTHESIS

def grad_descent(X, Y, theta, ALPHA = 0.001, ITERS = 10000, TOL = 1E-3, TYPE='lin', LAMBDA=0):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        theta ([type]): [description]
        ALPHA (float, optional): [description]. Defaults to 0.001.
        ITERS (int, optional): [description]. Defaults to 10000.
        TOL ([type], optional): [description]. Defaults to 1E-3.
        TYPE (str, optional): [description]. Defaults to 'lin'.
        LAMBDA (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    NUM_OBS = len(Y)
    COST, HYPOTHESIS = calc_cost(X, Y, theta, TYPE, LAMBDA)
    COST = [np.asscalar(COST)]
    THETA_COEFF = (1 - ALPHA*LAMBDA/NUM_OBS)*np.ones((theta.shape[0],1))
    THETA_COEFF[0,0] = 1    # set first element of coefficient vector to 0 to avoid 
                            # penalizing the intercept term (bias)
    
    for i in range(ITERS):
        SLOPE = (ALPHA/NUM_OBS)*(HYPOTHESIS - Y).T @ X
        theta = THETA_COEFF * theta - SLOPE.T
        COST_new, HYPOTHESIS = calc_cost(X, Y, theta, TYPE, LAMBDA)
        COST.append(np.asscalar(COST_new))
        COST_DIFF = abs(COST[i+1] - COST[i])
        if COST_DIFF < TOL:
            break

    return theta, COST

def normal_eqn(X, Y):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    mat_matT = X.T @ X
    mat_matT_inv = np.linalg.pinv(mat_matT)
    theta = (mat_matT_inv @ X.T) @ Y
    return theta

def scale_feature(X):
    """[summary]

    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_copy = X.copy()
    MEANMAT = np.mean(X_copy[:,1:], axis=0)
    X_copy[:,1:] = X_copy[:,1:] - MEANMAT
    STDMAT = np.std(X_copy[:,1:], axis=0)
    X_copy[:,1:] = X_copy[:,1:]/STDMAT
    return X_copy, MEANMAT, STDMAT

