from dictionary_updated import Dictionary2
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt


# FORWARD PASS
def forward_pass(x, w1, w2):
    #h = np.dot(w1.T, x)
    h = sparse.csr_matrix.dot(w1.T, x.T)
    u = np.dot(w2.T, h)
    y = softmax(u)
    return y, h, u


# SOFTMAX ACTIVATION FUNCTION
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# TRAIN W2V model
def train(X_train, Y_train, p, n, nclasses, EPOCH, n_classes, LR):
    # INITIALIZE WEIGHT MATRICES
    w1 = np.random.uniform(-0.8, 0.8, (p, n))               # context matrix
    w2 = np.random.uniform(-0.8, 0.8, (n, n_classes))       # embedding matrix

    # CYCLE THROUGH EACH EPOCH
    for i in range(0, EPOCH):
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)

        loss = 0
        l = 0
        
        # CYCLE THROUGH EACH TRAINING SAMPLE
        #for w_t, w_c in X_train:
        for x in X_train:
            true_val = Y_train[l]    
                
            # FORWARD PASS
            y_pred, h, u = forward_pass(x, w1, w2)

            # CALCULATE ERROR
            #EI = np.sum([np.subtract(y_pred, word) for word in true_val], axis=0)
            EI = np.subtract(y_pred.T, true_val)
            
            # BACKPROPAGATION
            w1, w2 = backprop(EI, h, x, w1, w2, alpha)

            # CALCULATE LOSS
            loss += -np.log(y_pred)

            l += 1
        print('EPOCH:',i, 'LOSS:', loss)
    pass


# BACKPROPAGATION
def backprop(e, h, x, w1, w2, alpha):
    dl_dw2 = np.outer(h, e.T)  
    dl_dw1 = sparse.csr_matrix.dot(np.dot(w2, e.T), x)

    # UPDATE WEIGHTS
    w1 = w1 - (alpha * dl_dw1.T)
    w2 = w2 - (alpha * dl_dw2)
    return w1, w2


def main():
    
     # args from Simple Queries paper
    DIM=30
    LR=0.001
    WORDGRAMS=3
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1000000
    EPOCH=20

    print("starting dictionary creation") 
    
    # initialize training
    dictionary = Dictionary2(WORDGRAMS, MINCOUNT, BUCKET)
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
    
    p = X_train.shape[1]
    n = DIM
    
    ##### instantiations #######################################
    
    train(X_train, y_train, p, n, nclasses, EPOCH, nclasses, LR)
    


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    