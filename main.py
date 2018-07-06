# main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse
from numpy.linalg import inv


def gradient_B(B, A, x, label, nlabels, alpha):
    j = 0
    xA = sparse.csr_matrix.dot(x, A)
    y_hat = stable_softmax(x, A, B)
    while j < nlabels:
        Bj = B[j, :]
        yj_hat = y_hat[j]
        yj = label[j]
        
        Bj_new = (yj_hat - yj) * xA
        #Bj_new = Bj_new / np.linalg.norm(Bj_new)
        
        B[j,:] = B[j,:] + alpha * Bj_new
        j += 1
    
    B = B / np.linalg.norm(B)
    #print(B)
    return B
        

# update rule for weight matrix A
def update_a():
    return 1


def stable_softmax(x, A, B):
    temp = sparse.csr_matrix.dot(x, A)  
    #print(temp)
    X = np.dot(temp, B.T)
    print(X)
    exps = np.exp(X - np.max(X))
    #exps = np.exp(X)
    #print(exps)
    return (exps / np.sum(exps))[0]


# finds the loss
def loss_function(x, A, B, label):
    loglike = np.log(stable_softmax(x, A, B))
    #print(loglike)
    return np.dot(label, loglike.T)


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.0001
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
    A_n = DIM               # cols
    A_m = nwords + BUCKET   # rows
    uniform_val = 1.0 / DIM
    #A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))
    A = np.ones((A_m, A_n))
    #print(A.shape)

    # B
    B_n = DIM               # cols
    B_m = nlabels           # rows
    B = np.zeros((B_m, B_n))
    #print(B.shape)


    #### train ################################################

    total_loss = 0
    #for i in range(EPOCH):
    # loop through each instance for SGD
    alpha = LR
    loss = 0
    l = 0
    for x in input_:
        #softmax = stable_softmax(x, A, B)
        label = labels[l]
        B_old = B
        
        loss = loss_function(x, A, B, label)
        #print(loss)
        B = gradient_B(B_old, A, x, label, nlabels, alpha)
        total_loss += loss        

        l += 1
        
    



if __name__ == '__main__':
    main()






