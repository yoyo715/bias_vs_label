# From https://github.com/vodp/py-kmm/blob/master/Kernel%20Meam%20Matching.ipynb

import os
import sys
import numpy as np 
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


# an implementation of Kernel Mean Matchin
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z,Z)
        kappa = np.sum(compute_rbf(Z,X),axis=1)*float(nz)/float(nx)
    else:
        raise ValueError('unknown kernel')
        
    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K


x = 11*np.random.random(200)- 6.0
y = x**2 + 10*np.random.random(200) - 5
Z = np.c_[x, y]

x = 2*np.random.random(10) - 6.0
y = x**2 + 10*np.random.random(10) - 5
X = np.c_[x, y]

coef = kernel_mean_matching(X, Z, kern='rbf', B=10)

plt.close()
plt.figure()
plt.scatter(Z[:,0], Z[:,1], color='black', marker='x')
plt.scatter(X[:,0], X[:,1], color='red')
plt.scatter(Z[:,0], Z[:,1], color='green', s=coef*10, alpha=0.5)

np.sum(coef > 1e-2)


