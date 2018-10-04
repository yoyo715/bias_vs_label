# optimal_beta3.py

import  os
import  sys
import  numpy as  np 
import  math
from cvxopt import matrix, solvers
from dictionary3 import Dictionary


# an implementation of Kernel Mean Matchin
# references:
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
        print(vx.shape, Z.T.shape)
        print(np.subtract(vx, Z).shape)
        print(np.sum((vx-Z.T)**2, axis=1).shape)
        
        K[i,:] = np.exp(-np.sum((vx-Z.T)**2, axis=1)/(2.0*sigma))
    return K


def main():
    WORDGRAMS=3
    MINCOUNT=2
    BUCKET=1000000

    print("starting dictionary creation") 
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET)
    
    n_train = dictionary.get_n_train_instances()
    n_test = dictionary.get_n_manual_instances()

    X_train = dictionary.get_trainset()
    X_test = dictionary.get_manual_testset()
    
    print("starting optimization")
    coef = kernel_mean_matching(X_train, X_test, kern='rbf', B=10)
    print(coef)
    
 
if __name__ == '__main__':
    main()
