#main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse


# update rule for weight matrix B
# 
def update_b(B, A, x, labels, nlabels):
    # create S
    k =1 
    Sn = 0
    A_xn = sparse.csr_matrix.dot(A, x.T)  
    while k <= nlabels:
        product = np.dot(B[:, k-1].T, A_xn)
        Sn += np.dot(product, nlabels)
        k += 1

    K = 1
    BAxn = np.dot(B.T, A_xn)
    while K <= nlabels:
        # create I_k
        I_k = np.ones(nlabels)
        I_k[K-1] = 0    # kth element set to 0

        num = np.dot(I_k, BAxn) 
        denom = np.dot(B[:, K-1].T, A_xn)
        
        K += 1
        

# update rule for weight matrix A
def update_a():
    return 1


# calculates softmax value
def softmax(x, A, B, nlabels):
    temp = sparse.csr_matrix.dot(A, x.T)  
    product = np.dot(B.T, temp)
    exp = np.exp(product)
    return exp / nlabels * exp
    

# calculates log-likelihood
def log_likelihood(x, A, B, nlabels):
    return np.log(softmax(x, A, B, nlabels))


# finds the loss
def loss_function(label, loglike):
    return np.dot(label, loglike)


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.1
    WORDGRAMS=2
    MINCOUNT=2
    MINN=3
    MAXN=3
    #BUCKET=1 #000000
    BUCKET = 0
    EPOCH=20

    train = open('../cleaned_train_withstopwords.txt', 'r')
    dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
    input_ = dictionary.get_bagngram()
    labels = dictionary.get_labels()
    nwords = dictionary.get_nwords()
    nlabels = dictionary.get_nlabels()
    N = dictionary.get_ninstances()

    
    ##### instantiations #######################################

    # A
    A_n = DIM               # rows
    A_m = nwords + BUCKET   # cols
    uniform_val = 1.0 / DIM
    A = np.random.uniform(-uniform_val, uniform_val, (A_n, A_m))

    # B
    B_n = DIM               # rows
    B_m = nlabels           # cols
    B = np.zeros((B_n, B_m))

    #### train ################################################

    total_loss = 0
    #for i in range(EPOCH):
    # loop through each instance for SGD
    loss = 0
    l = 0
    for x in input_:
        loglike = log_likelihood(x, A, B, nlabels)
        loss = loss_function(labels[l], loglike)
        total_loss += loss        

        B_new = update_b(B, A, x, labels[l], nlabels)
        
        l += 1
        
    



if __name__ == '__main__':
    main()






