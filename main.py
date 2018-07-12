# main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt

# computes the hidden layer
def compute_hidden(x, A):
    #hidden = sparse.csr_matrix.dot(x, A)
    hidden = sparse.csr_matrix.dot(A, x.T)
    return hidden / np.linalg.norm(hidden)


# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nlabels, alpha):
    j = 0
    hidden = compute_hidden(x, A)
    y_hat = stable_softmax(x, A, B)
    while j < nlabels:
        Bj = B[j, :]
        yj_hat = y_hat[j]
        yj = label[j]
        
        Bj_new = alpha*((yj_hat - yj) * hidden)
        Bj_new = np.reshape(Bj_new, (30))
        Bj_new = np.subtract(Bj, Bj_new)
        
        B[j, :] = Bj_new
            
        j += 1
    
    #B = B / np.linalg.norm(B)
    return B
        

# update rule for weight matrix A
def gradient_A():
    return 1


def stable_softmax(x, A, B):
    hidden = compute_hidden(x, A) 
    #X = np.dot(hidden, B.T)
    X = np.dot(B, hidden)
    exps = np.exp(X - np.max(X))
    return (exps / np.sum(exps))


# finds the loss
def loss_function(x, A, B, label):
    loglike = np.log(stable_softmax(x, A, B))
    return np.dot(label, loglike)


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.001
    WORDGRAMS=2
    MINCOUNT=2
    MINN=3
    MAXN=3
    #BUCKET=1 #000000
    BUCKET = 0
    EPOCH=5

    NUMTRAIN_INST = 10
    NUMTEST_INST = 5

    train = open('../cleaned_train_withstopwords_FULL2.txt', 'r')
    test = open('../cleaned_test_withstopwords_FULL.txt', 'r')
    #train = open('../cleaned_train_withstopwords.txt', 'r')
    print("starting TRAIN dictionary creation")
    dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
    input_ = dictionary.get_bagngram()
    labels = dictionary.get_labels()
    nwords = dictionary.get_nwords()
    nlabels = dictionary.get_nlabels()
    N = dictionary.get_ninstances()
    print("finished creating dictionary for TRAIN: ", N, " number of Train instances")
    
    print()
    print("starting TEST dictionary creation")
    dictionary_test = Dictionary(test, WORDGRAMS, MINCOUNT)
    input_test = dictionary.get_bagngram()
    labels_test = dictionary.get_labels()
    nwords_test = dictionary.get_nwords()
    nlabels_test = dictionary.get_nlabels()
    N_test = dictionary.get_ninstances()
    print("finished creating dictionary for TEST: ", N_test, " number of Test instances")
    
    ##### instantiations #######################################

    # A
    A_n = nwords + BUCKET   # cols
    A_m = DIM               # rows
    uniform_val = 1.0 / DIM
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))
    #print(A.shape)

    # B
    B_n = DIM               # cols
    B_m = nlabels           # rows
    B = np.zeros((B_m, B_n))
    #print(B.shape)


    #### train ################################################

    losses = []
    losses_test = []
    
    for i in range(EPOCH):
        print("EPOCH: ", i)
        # loop through each instance for SGD
        alpha = LR
        loss = 0
        l = 0
        total_loss = 0
        
        train_inst = 0
        
        # TRAINING
        for x in input_:
            label = labels[l]
            B_old = B
            A_old = A
            
            # forward prop
            loss = loss_function(x, A, B, label) 
            
            # back prop
            B = gradient_B(B_old, A, x, label, nlabels, alpha)  
            #A = gradient_A()
            
            total_loss += loss        
            l += 1

            train_inst += 1
            if train_inst == NUMTRAIN_INST:
                break
            
            
        #TESTING 
        q = 0
        total_loss_test = 0
        test_inst = 0
        for xtest in input_test:
            label_test = labels_test[q]
            loss_test = loss_function(xtest, A, B, label_test) 
            total_loss_test += loss_test
            q += 1
            
            test_inst += 1
            if test_inst == NUMTEST_INST:
                break
            
            
        losses_test.append(total_loss_test/N_test * -1)
        losses.append(total_loss/N * -1)
        i += 1

    epochs = [l for l in range(EPOCH)]
    plt.plot(epochs, losses)
    plt.plot(epochs, losses_test)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
        
    
if __name__ == '__main__':
    main()






