# kmm.py

from dictionary3 import Dictionary
import numpy as np
from scipy import sparse
import math


def update_beta(X_train, X_test, beta, n_train, n_test):
    K = compute_gram(X_train, n_train)
    k = compute_k(n_train, n_test, X_train, X_test)
    return 0.5 * np.dot(np.dot(beta.T, K), beta) - np.dot(k.T, beta)


def update_alpha():
    return 1


def compute_gram(X, n):
    sigma = np.std(X)  # compute standard deviation ????
    
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = gaussian_kernel(X[i], X[j], sigma)
            
    return K


def compute_k(n_train, n_test, X_train, X_test):
    k = np.zeros((n_test, n_train))
    for i in range(n_train):
        xi_train = X_train[i]
        ki = compute_ki(n_train, n_test, xi_train, X_test)
        k[i] = ki
        
    return k
    
    
def compute_ki(n_train, n_test, xi_train, X_test):
    _sum =  np.zeros((n_test))
    for j in range(n_test):
        _sum[j] = gaussian_kernel(xi_train, X_test[j], sigma)
        
    return n_train/n_test * _sum


def compute_alpha(beta, _lambda, Y):
    return (1/_lambda)*(np.dot(beta, Y) * (1/math.log(10)) + np.dot(beta, Y))
    

def gaussian_kernel(x_i, x_j, sigma):
    return  np.exp(-linalg.norm(x_i - x_j)**2 / (2 * (sigma ** 2)))


def compute_loss(K, beta, Y, alpha, _lambda):
    return (np.dot(beta, Y) * np.log(np.exp(np.dot(alpha, K))
                                    - np.dot(np.dot(beta, Y), K)
                                    + (_lambda / 2) * np.dot(np.dot(alpha, alpha), K))


def main():
    # args from Simple Queries paper
    DIM=30
    LR=0.08
    WORDGRAMS=3
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1000000
    EPOCH=30

    print("starting dictionary creation") 
    
    # initialize training
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET)
    nwords = dictionary.get_nwords()
    nclasses = dictionary.get_nclasses()
    
    #initialize testing
    X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    N = dictionary.get_n_train_instances()
    N_test = dictionary.get_n_test_instances()
    
    print("Number of Train instances: ", N, " Number of Test instances: ", N_test)
    ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
    ntest_eachclass = dictionary.get_nlabels_eachclass_test()
    print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)
    
    ##### instantiations #######################################


    p = X_train.shape[1]
    
    # A
    A_n = p                 # cols
    A_m = DIM               # rows
    uniform_val = 1.0 / DIM
    np.random.seed(0)
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))

    # B
    B_n = DIM               # cols
    B_m = nclasses          # rows
    B = np.zeros((B_m, B_n))


    #### train ################################################




if __name__ == '__main__':
    main()


