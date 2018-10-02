import numpy as np
from scipy.optimize import minimize
from dictionary3 import Dictionary

WORDGRAMS=3
MINCOUNT=2
BUCKET=1000000

dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET)

def objective(beta):
    K = np.random.rand(n_tr, n_tr)
    k = np.random.rand(n_tr)
    return 0.5*np.subtract(np.dot(np.dot(beta.T, K), beta), np.dot(k.T, beta))


def constraint1(beta):
    return beta

def constraint2(beta):
    B = 5
    return B - beta

def constraint3(beta):
    epsilon = 1e-7
    n_tr = 5
    return np.subtract(beta, n_tr*(1-epsilon))
    #return beta - n_tr*(1-epsilon)

def constraint4(beta):
    epsilon = 1e-7
    n_tr = 5
    return np.subtract(n_tr*(epsilon - 1), beta)
    #return n_tr*(epsilon - 1) - beta


B = 5
b = (0.0, B)
bounds = (b,b,b,b,b)


con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}

cons = [con1, con2, con3, con4]

sol = minimize(objective, beta0, method='SLSQP',
               bounds = bounds, constraints = cons)

print (sol.x)
