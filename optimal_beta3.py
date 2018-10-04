# optimal_beta3.py

import  os
import  sys
import  numpy as  np 
import  math
import scipy as sp
from cvxopt import matrix, solvers, spmatrix, sparse, mul
from dictionary3 import Dictionary


# an implementation of Kernel Mean Matchin
# references:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.


def kernel_mean_matching(X_train, X_test, n_train, nz, kern='lin', B=1.0, eps=None):    
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        #K = sp.sparse.csr_matrix.dot(X_test, X_test.T)
        K = create_K(X_train, n_train)
        kappa = np.sum(sp.sparse.csr_matrix.dot(X_train, X_test.T)*float(n_train)/float(nz),axis=1)
    elif kern == 'rbf':
        K = compute_rbf(X_train,X_test)
        kappa = np.sum(compute_rbf(X_train,X_test),axis=1)*float(nz)/float(n_train)
    else:
        raise ValueError('unknown kernel')
        
    #K = K.toarray()
    print("K ", K.shape, type(K))
    K = K.astype(np.double)
    K = matrix(K)
    print("kappa ", kappa.shape, type(kappa))
    kappa = matrix(kappa)
    
    #G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    #h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    G = matrix(np.r_[np.ones((1,n_train)), -np.ones((1,n_train)), np.eye(n_train), -np.eye(n_train)])
    h = matrix(np.r_[n_train*(1+eps), n_train*(eps-1), B*np.ones((n_train,)), np.zeros((n_train,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef

def compute_rbf(X_train, X_test, sigma=1.0):
    K = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=float)
    print("*", K.shape)
    for i, vx in enumerate(X_train):
        print(vx.shape, X_test.shape)
        K[i,:] = np.exp(-np.sum((vx-X_test)**2, axis=1)/(2.0*sigma))
    return K

def linear_kernel(x_i, x_j):
    c = 0.1
    val = sp.sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val + c

def kernel(x_i, x_j):
    #return gaussian_kernel(x_i, x_j)
    return linear_kernel(x_i, x_j)

def create_K(X_train, n_train):
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K[i,j] = kernel(X_train[i], X_train[j])
            
    return K

def main():
    WORDGRAMS=3
    MINCOUNT=2
    BUCKET=1000000

    print("starting dictionary creation.............................") 
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET)
    X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    n_train = dictionary.get_n_train_instances()
    n_test = dictionary.get_n_manual_instances()

    X_train = dictionary.get_trainset()
    X_test = dictionary.get_manual_testset()
        
    print()
    print("starting optimization")
    coef = kernel_mean_matching(X_train, X_test, n_train, n_test, kern='lin', B=10)
    print(coef)
    
 
if __name__ == '__main__':
    main()
