import os
import sys
import numpy as np 
import math, time
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import sklearn.metrics.pairwise as sk
from sklearn.metrics import mean_squared_error

"""
    This script compares the two KMM methods I found online.
"""

def stochastic_kmm(X, Z, sum_betas, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    
    if eps == None:
        eps = B/math.sqrt(nz)
    
    if kern == 'lin':
        Ztemp = Z.reshape(1, -1)
        K = sk.linear_kernel(Ztemp, Ztemp)
        kappa = np.sum(sk.linear_kernel(Ztemp,X),axis=1)*float(nz)/float(nx)
        
    Kshape = K.shape
    kappashape = kappa.shape
    
    K = matrix(K)
    kappa = matrix(kappa)
    #G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    #G = matrix((np.r_[1.0, -1.0, 1.0, -1.0]).reshape(-1, 1))
    G = matrix((np.r_[sum_betas, -sum_betas, 1.0, -1.0]).reshape(-1, 1))
    
    #h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B, 0])
    
    #print("K.shape: ", Kshape)
    #print(K)
    
    #print("kappa.shape: ", kappashape)
    #print(kappa)
    
    #print("G.shape: ", (np.r_[1, -1, 1, -1]).reshape(-1, 1).shape)
    #print(G)
    
    #print("h.shape: ", np.r_[nz*(1+eps), nz*(eps-1), B, 0].shape)
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef
    

def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    print("X.shape: ", X.shape, "Z.shape: ", Z.shape)
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        K2 = np.dot(Z, Z.T)
        K = sk.linear_kernel(Z, Z) 
        
        print((K2==K).all())
        kappa = np.sum(sk.linear_kernel(Z,X),axis=1)*float(nz)/float(nx)
    elif kern == 'rbf':
        sigma = 1.0
        K = sk.rbf_kernel(Z, Z, sigma)
        kappa = np.sum(sk.rbf_kernel(Z,X),axis=1)*float(nz)/float(nx)
    else:
        raise ValueError('unknown kernel')
        
    Kshape = K.shape
    kappashape = kappa.shape
    
    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    print("**** ", np.ones((1,nz)).shape, np.eye(nz).shape)
    
    
    print("K.shape: ", Kshape)
    #print(K)
    
    print("kappa.shape: ", kappashape)
    #print(kappa)
    
    print("G.shape: ", np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)].shape)
    #print(G)
    
    print("h.shape: ", np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))].shape)
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    print(coef.shape)
    return coef



###################################################
def kmm(Xtrain, Xtest, sigma):
    n_tr = len(Xtrain)
    n_te = len(Xtest)

    # calculate Kernel
    print('Computing kernel for training data ...')
    #K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
    K = sk.linear_kernel(Xtrain, Xtrain)
    # make it symmetric
    #K = 0.9 * (K_ns + K_ns.transpose())

    # calculate kappa
    print('Computing kernel for kappa ...')
    #kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
    kappa_r = sk.linear_kernel(Xtrain, Xtest)
    print("************ ", kappa_r.shape)
    
    ones = np.ones(shape=(n_te, 1))
    kappa = np.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    B = 10.0
    eps = B/math.sqrt(n_tr)
    #eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)

    # constraints
    A0 = np.ones(shape=(1, n_tr))
    A1 = -np.ones(shape=(1, n_tr))
    A = np.vstack([A0, A1, -np.eye(n_tr), np.eye(n_tr)])
    b = np.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = np.vstack([b.T, -np.zeros(shape=(n_tr, 1)), np.ones(shape=(n_tr, 1)) * 1000])

    print('Solving quadratic program for beta ...')
    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    beta = solvers.qp(P, q, G, h)
    coef = np.array(beta['x'])
    #return [i for i in beta['x']]
    return coef

def getBeta(trainX, testX):
    beta = []
    gammab = 0.001
    #gammab = computeKernelWidth(trainX)
    #print("Gammab:", gammab)

    start = time.time()
    beta = kmm(trainX, testX, gammab)
    end = time.time()
    print("2nd version took: ", (end - start)/60.0, " time")
    #print("{0} Beta: {1}".format(len(beta), beta))

    return beta


def main():
    numtrain = 100
    x = 11*np.random.random(numtrain)- 6.0
    y = x**2 + 10*np.random.random(numtrain) - 5
    Z = np.c_[x, y]  # TRAINSET
    print("Z (trainset) shape: ", Z.shape)

    numtest = 200
    x = 2*np.random.random(numtest) - 6.0
    y = x**2 + 10*np.random.random(numtest) - 5
    X = np.c_[x, y]  # TESTSEST
    print("X (testset) shape: ", X.shape)

    start = time.time()
    coef = kernel_mean_matching(X.T,Z.T, kern='lin', B=10)
    end = time.time()
    print("1nd version took: ", (end - start)/60.0, " time")
    
    
    #########################################
    #sum_betas = 0
    #betas = [1.0] * numtrain
    #betas = np.array(betas)
    #for epoch in range(10):
        #i = 0
        ##sum_betas = 100
        #for sample in Z:
            #sum_betas = np.sum(betas)
            ##print(sum_betas)
            #temp = stochastic_kmm(X,sample, sum_betas, kern='lin', B=10)
            #betas[i] = temp[0,0]
            #i += 1
        #print("MSE, epoch(", epoch, "): ", mean_squared_error(coef, betas))
        
    
    #print("****")
    ##print(betas)
    ##print(coef)
    #print(np.sum(coef))
    #########################################
    
    
    #coef3 = np.array(coef3)
    #print(type(coef), type(coef2))

    plt.close()
    plt.figure()
    plt.scatter(Z[:,0], Z[:,1], s=20, color='black', marker='x')
    plt.scatter(X[:,0], X[:,1], s=20, color='red')
    plt.scatter(Z[:,0], Z[:,1], color='green', s=coef*50, alpha=0.5)
    #plt.scatter(Z[:,0], Z[:,1], color='blue', s=coef2*20, alpha=0.5)
    #plt.scatter(Z[:,0], Z[:,1], color='purple', s=betas*50, alpha=0.5)
    plt.show()
    
    #np.sum(coef > 1e-2)
    
    


if __name__ == '__main__':
    main()
