"""Set of functions useful for data visualization in regression analysis"""
    
import matplotlib.pyplot as plt
import numpy as np
from regression.cost_func import scale_feature

def map_feature(X, deg=2):
    """Maps two input features to specified degrees"""
    m, _ = X.shape
    X1 = X[:,[1]]
    X2 = X[:,[2]]
    X_final = np.ones((m, 1))
    for i in range(1, deg+1):
        for j in range(0, i+1):
            X_final = np.append(X_final, (X1**(i-j))*(X2**j), axis=1)
    return X_final

def plot_boundary(theta, minr=-1.5, maxr=1.5, deg=2, norm=1, meanmat=[], stdmat=[]):
    """Returns a plot of decision boundary"""
    
    n, _ = theta.shape
    LIN_VEC = np.linspace(minr, maxr, 100)
    ONE_VEC = np.ones((len(LIN_VEC),1))
    X_plot = np.array([LIN_VEC]).T
    X_plot = np.append(ONE_VEC, X_plot[:,[0]*(n-1)], axis=1)
    x, y = np.meshgrid(LIN_VEC, LIN_VEC)
    Y_plot = np.zeros(x.shape)
    
    try:
        if norm:
            LIN_VEC_MAP_MEAN = np.insert(meanmat, 0, [0], axis=0)
            LIN_VEC_MAP_STD = np.insert(stdmat, 0, [1], axis=0)
            for j in range(len(LIN_VEC)):
                for i in range(len(LIN_VEC)):
                    x_mult = (map_feature(np.array([[1, LIN_VEC[i], LIN_VEC[j]]]), deg=deg) - LIN_VEC_MAP_MEAN)/LIN_VEC_MAP_STD
                    Y_plot[i][j] = x_mult @ theta
        else:
            for j in range(len(LIN_VEC)):
                for i in range(len(LIN_VEC)):
                    x_mult = map_feature(np.array([[1, LIN_VEC[i], LIN_VEC[j]]]), deg=deg)
                    Y_plot[i][j] = x_mult @ theta
    except:
        print('Error in inputs')
            
    cl1 = plt.contour(x, y, Y_plot, levels=[0])
    plt.clabel(cl1, inline=1)
    return cl1, Y_plot