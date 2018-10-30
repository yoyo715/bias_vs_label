# optimal_beta4.py

import  os
import  sys
import  numpy as  np 
import  math
import scipy as sp
import time
from cvxopt import matrix, solvers, spmatrix, sparse, mul
#from dictionary3 import Dictionary


# an implementation of Kernel Mean Matchin
# references:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.


# implementation based off of https://github.com/vodp/py-kmm/blob/master/Kernel%20Meam%20Matching.ipynb


#def kernel_mean_matching(X_train, X_test, n_train, n_test, lin_c, kern='lin', B=1.0, eps=None):    
    #if eps == None:
        #eps = B/math.sqrt(n_test)
    
    #K = create_K(X_train, n_train, kern, lin_c)*float(n_train)/float(n_test)
    #kappa = create_k(X_train, n_train, X_test, n_test, kern, lin_c)*float(n_train)/float(n_test)
        
    #print("K ", K.shape, type(K))
    #K = K.astype(np.double)
    #K = matrix(K)
    #print("kappa ", kappa.shape, type(kappa))
    #kappa = matrix(kappa)
    
    #G = matrix(np.r_[np.ones((1,n_train)), -np.ones((1,n_train)), np.eye(n_train), -np.eye(n_train)])
    #h = matrix(np.r_[n_train*(1+eps), n_train*(eps-1), B*np.ones((n_train,)), np.zeros((n_train,))])

    #sol = solvers.qp(K, -kappa, G, h)
    #coef = np.array(sol['x'])
    #print(coef[0:10])
    #return coef  
    
    
# Z is training data, X is testing data
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    
    print("nx: ", nx, " nz: ", nz)
    
    if eps == None:
        eps = B/math.sqrt(nz)
        
    if kern == 'lin':
        K = np.dot(Z, Z.T) #+ 0.90
        K = K.todense()
        kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
        
    elif kern == 'rbf':
        K = compute_rbf(Z,Z)
        kappa = np.sum(compute_rbf(Z,X),axis=1)*float(nz)/float(nx)
        
    else:
        raise ValueError('unknown kernel')
    
    
    K = K.astype(np.double)
    K = matrix(K)
    
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


#def compute_rbf(X_train, X_test, sigma=1.0):
    #K = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=float)
    #print("*", K.shape)
    #for i, vx in enumerate(X_train):
        #print(vx.shape, X_test.shape)
        #K[i,:] = np.exp(-np.sum((vx-X_test)**2, axis=1)/(2.0*sigma))
        #K[i,:] = np.exp(-np.sum((vx-X_train)**2, axis=1)/(2.0*sigma))   # NOTE: try this one
    #return K
    
def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    Z = Z.todense()
    
    for i, vx in enumerate(X):
        vx = vx.todense()
        K[i,:] = np.exp(-np.sum(np.square(vx-Z), axis=1)/(2.0*sigma)).flatten()
    return K
    
    
#def compute_rbf(x_i, x_j, sigma=10.0):
    ##return np.exp(-np.sum((x_i.toarray()-x_j.toarray())**2, axis=1)/(2.0*sigma))
    #return np.exp(-np.sum((x_i.toarray()-x_j.toarray())**2, axis=1)/(2.0*sigma))   # NOTE: try this one


# gaussian kernel not going to work with sparse matrices since norm is 0
def gaussian_kernel(x_i, x_j, sigma=1.0):
    return  np.exp(-norm(np.subtract(x_i - x_j))**2 / (2 * (sigma ** 2)))


def linear_kernel(x_i, x_j, lin_c):
    val = sp.sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val + lin_c


def poly_kernel(x_i, x_j, d=2):
    c = 0.1
    val = sp.sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val**d + c


def kernel(x_i, x_j, kern, lin_c):
    if kern == 'lin':
        return linear_kernel(x_i, x_j, lin_c)
    elif kern == 'rbf':
        #return gaussian_kernel(x_i, x_j)
        return compute_rbf(x_i, x_j)
    elif kern == 'poly':
        return poly_kernel(x_i, x_j)
    else:
        raise ValueError('unknown kernel')


# only run as script for testing, otherwise dictionary calls kernel_means_matching()
#def main():
    ## args from Simple Queries paper
    #DIM=30
    #WORDGRAMS=2
    #MINCOUNT=8
    #MINN=3
    #MAXN=3
    #BUCKET=1000000

    ## adjust these
    #EPOCH=5
    #LR=0.15             # 0.15 good for ~5000
    #KERN = 'lin'        # lin or rbf or poly
    #NUM_RUNS = 1        # number of test runs
    #SUBSET_VAL = 300   # number of subset instances for self reported dataset
    #LIN_C = 0.90        # hyperparameter for linear kernel

    #print("starting dictionary creation.............................") 
    #dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, model='original')
    #X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    #n_train = dictionary.get_n_train_instances()
    #n_test = dictionary.get_n_manual_instances()

    #X_train = dictionary.get_trainset()
    #X_test = dictionary.get_manual_testset()
        
    #print()
    #print("starting optimization")
    ##coef = kernel_mean_matching(X_train, X_test, n_train, n_test, kern='lin', B=10)
    #coef = kernel_mean_matching(X_test, X_train, kern='lin', B=10)
    #print(coef)
    
    #print    
 
#if __name__ == '__main__':
    #main()
