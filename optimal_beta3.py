# optimal_beta3.py

import  os
import  sys
import  numpy as  np 
import  math
import scipy as sp
from cvxopt import matrix, solvers, spmatrix, sparse, mul
#from dictionary3 import Dictionary


# an implementation of Kernel Mean Matchin
# references:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.


# implementation based off of https://github.com/vodp/py-kmm/blob/master/Kernel%20Meam%20Matching.ipynb


def kernel_mean_matching(X_train, X_test, n_train, n_test, kern='lin', B=1.0, eps=None):    
    if eps == None:
        eps = B/math.sqrt(n_test)
    
    K = create_K(X_train, n_train, kern)*float(n_train)/float(n_test)
    kappa = create_k(X_train, n_train, X_test, n_test, kern)*float(n_train)/float(n_test)
        
    print("K ", K.shape, type(K))
    K = K.astype(np.double)
    K = matrix(K)
    print("kappa ", kappa.shape, type(kappa))
    kappa = matrix(kappa)
    
    G = matrix(np.r_[np.ones((1,n_train)), -np.ones((1,n_train)), np.eye(n_train), -np.eye(n_train)])
    h = matrix(np.r_[n_train*(1+eps), n_train*(eps-1), B*np.ones((n_train,)), np.zeros((n_train,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    print(coef[0:10])
    return coef  


#def compute_rbf(X_train, X_test, sigma=1.0):
    #K = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=float)
    #print("*", K.shape)
    #for i, vx in enumerate(X_train):
        #print(vx.shape, X_test.shape)
        #K[i,:] = np.exp(-np.sum((vx-X_test)**2, axis=1)/(2.0*sigma))
        #K[i,:] = np.exp(-np.sum((vx-X_train)**2, axis=1)/(2.0*sigma))   # NOTE: try this one
    #return K
    
    
def compute_rbf(x_i, x_j, sigma=5.0):
    return np.exp(-np.sum((x_i.toarray()-x_j.toarray())**2, axis=1)/(2.0*sigma))
    # ORRR return np.exp(-np.sum((vx-X_train)**2, axis=1)/(2.0*sigma))   # NOTE: try this one


# gaussian kernel not going to work with sparse matrices since norm is 0
def gaussian_kernel(x_i, x_j, sigma=1.0):
    return  np.exp(-norm(np.subtract(x_i - x_j))**2 / (2 * (sigma ** 2)))


def linear_kernel(x_i, x_j):
    c = 0.1
    val = sp.sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val + c


def poly_kernel(x_i, x_j, d=2):
    c = 0.1
    val = sp.sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val**d + c


def kernel(x_i, x_j, kern):
    if kern == 'lin':
        return linear_kernel(x_i, x_j)
    elif kern == 'rbf':
        #return gaussian_kernel(x_i, x_j)
        return compute_rbf(x_i, x_j)
    elif kern == 'poly':
        return poly_kernel(x_i, x_j)
    else:
        raise ValueError('unknown kernel')


def create_K(X_train, n_train, kern):
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K[i,j] = kernel(X_train[i], X_train[j], kern)
    return K


def create_k(X_train, n_train, X_test, n_test, kern):
    k = np.zeros((n_train))
    for i in range(n_train):
        xi_train = X_train[i]
        ki = compute_ki(n_train, n_test, xi_train, X_test, kern)
        k[i] = ki
    return k
    
    
def compute_ki(n_train, n_test, xi_train, X_test, kern):
    _sum =  np.zeros((n_test))
    for j in range(n_test):
        _sum[j] = kernel(xi_train, X_test[j], kern)
    return n_train/n_test * np.sum(_sum)


# only run as script for testing, otherwise dictionary calls kernel_means_matching()
#def main():
    #WORDGRAMS=3
    #MINCOUNT=2
    #BUCKET=1000000

    #print("starting dictionary creation.............................") 
    #dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET)
    #X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    #n_train = dictionary.get_n_train_instances()
    #n_test = dictionary.get_n_manual_instances()

    #X_train = dictionary.get_trainset()
    #X_test = dictionary.get_manual_testset()
        
    #print()
    #print("starting optimization")
    #coef = kernel_mean_matching(X_train, X_test, n_train, n_test, kern='lin', B=10)
    #print(coef)
    
 
#if __name__ == '__main__':
    #main()
