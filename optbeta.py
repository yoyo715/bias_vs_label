import numpy as np
from scipy.optimize import minimize
from dictionary3 import Dictionary
import math
from scipy import sparse
from scipy.sparse.linalg import norm

def objective(beta):
    return 0.5*np.subtract(np.dot(np.dot(beta.T, K), beta), np.dot(k.T, beta))

def constraint1(beta):
    return beta

def constraint2(beta):
    return B - beta

def constraint3(beta):
    epsilon = 1e-7
    return np.subtract(beta, n_train*(1-epsilon))

def constraint4(beta):
    epsilon = 1e-7
    return np.subtract(n_train*(epsilon - 1), beta)


####################################################################

# kernels: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear

def kernel(x_i, x_j):
    #return gaussian_kernel(x_i, x_j)
    return linear_kernel(x_i, x_j)

def linear_kernel(x_i, x_j):
    c = 5
    val = sparse.csr_matrix.dot(x_i, x_j.T).sum()
    return val + c

def polynomial_kernel(x_i, x_j):
    return 1

# gaussian kernel not going to work with sparse matrices since norm is 0
def gaussian_kernel(x_i, x_j):
    return  np.exp(-norm(np.subtract(x_i - x_j))**2 / (2 * (sigma ** 2)))
    

def create_K():
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            #K[i,j] = gaussian_kernel(X_train[i], X_train[j])
            #K[i,j] = linear_kernel(X_train[i], X_train[j])
            K[i,j] = kernel(X_train[i], X_train[j])
            
    return K


def create_k():
    k = np.zeros((n_train))
    for i in range(n_train):
        xi_train = X_train[i]
        ki = compute_ki(n_train, n_test, xi_train, X_test)
        k[i] = ki
        
    return k
    
    
def compute_ki(n_tr, n_te, xi_train, X_te):
    _sum =  np.zeros((n_te))
    for j in range(n_te):
        _sum[j] = kernel(xi_train, X_te[j])
        
    return n_tr/n_te * np.sum(_sum)


def get_optbeta():
    return beta


###################################################################


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

B = n_train
#sigma = np.std(X_train)  # compute standard deviation ????
sigma = 0.25

b = (0.0, B)
bounds = (b,b,b,b,b)
beta0 = np.zeros((n_train))

print("creating gram matrix")
K = create_K()
k = create_k()
print(K.shape, k.shape)
print("dont creating gram matrix")
print()

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}

cons = [con1, con2, con3, con4]

print("starting optimization")
sol = minimize(objective, beta0, method='SLSQP',
            bounds = bounds, constraints = cons)

beta = sol.x

print (sol.x)
    




