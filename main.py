# main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt


# computes the hidden layer
def compute_hidden(x, A):
    hidden = sparse.csr_matrix.dot(A, x.T)
    norm = np.linalg.norm(hidden)
    if norm == 0: 
       return hidden
    return hidden / norm 


# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nclasses, alpha, DIM):
    j = 0
    hidden = compute_hidden(x, A)
    y_hat = stable_softmax(x, A, B)
    while j < nclasses:
        Bj = B[j, :]
        yj_hat = y_hat[j]
        yj = label[j]
        
        Bj_new = alpha*( (yj_hat - yj) * hidden )
        Bj_new = np.reshape(Bj_new, (DIM))
        Bj_new = np.subtract(Bj, Bj_new)
        
        B[j, :] = Bj_new
            
        j += 1
    
    return B
        

# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, DIM):
    j = 0
    p = A.shape[1]
    y_hat = stable_softmax(x, A, B)
    A_new = np.zeros((DIM, p))
    
    while j < nclasses:
        Bj = np.reshape(B[j, :], (DIM, 1))
        yj_hat = y_hat[j]
        yj = label[j]
        
        a = (yj_hat - yj)* sparse.csr_matrix.dot(Bj, x)
        A_new = np.add(A_new, a)
        j += 1
        
    #norm = np.linalg.norm(A_new)
    #if norm != 0: 
        ##A_new = A_new / norm
        #A_new = np.divide(A_new, norm)
    
    #A = A - (alpha * A_new)
    A = np.subtract(A, (alpha * A_new))
    return A
            

def stable_softmax(x, A, B):
    hidden = compute_hidden(x, A) 
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
    LR=0.01
    WORDGRAMS=3
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1000000
    #BUCKET = 0
    EPOCH=10

    dataset = open('../cleaned_subset.txt', 'r').readlines()
    
    print("starting dictionary creation") 
    
    # initialize training
    dictionary = Dictionary(dataset, WORDGRAMS, MINCOUNT, BUCKET)
    nwords = dictionary.get_nwords()
    nclasses = dictionary.get_nclasses()
    
    #initialize testing
    X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    N = dictionary.get_n_train_instances()
    N_test = dictionary.get_n_test_instances()
    
    print("Number of Train instances: ", N, " Number of Test instances: ", N_test)
    
    
    ##### instantiations #######################################

    p = X_train.shape[1]
    
    # A
    #A_n = nwords + BUCKET   # cols
    A_n = p
    A_m = DIM               # rows
    uniform_val = 1.0 / DIM
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))
    #print(A.shape)

    # B
    B_n = DIM               # cols
    B_m = nclasses          # rows
    B = np.zeros((B_m, B_n))
    #print(B.shape)


    #### train ################################################

    losses = []
    losses_test = []
    print()
    
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        loss = 0
        l = 0
        total_loss = 0
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        # TRAINING
        for x in X_train:
            label = y_train[l]
            B_old = B
            A_old = A
            
            # forward prop
            loss = loss_function(x, A, B, label) 
            
            # back prop
            B = gradient_B(B_old, A_old, x, label, nclasses, alpha, DIM)  
            A = gradient_A(B_old, A_old, x, label, nclasses, alpha, DIM)
            
            total_loss += loss        
            l += 1
            
            
        # TESTING 
        q = 0
        total_loss_test = 0
        for xtest in X_test:
            label_test = y_test[q]
            loss_test = loss_function(xtest, A, B, label_test) 
            total_loss_test += loss_test
            q += 1
            
        print("train: ", total_loss/N * -1)
        print("test: ", total_loss_test/N_test * -1)
            
            
        losses_test.append(total_loss_test/N_test * -1)
        losses.append(total_loss/N * -1)
        i += 1
        
    
    
    # for plotting
    epochs = [l for l in range(EPOCH)]
    plt.plot(epochs, losses, 'r', label="training loss")
    plt.plot(epochs, losses_test, 'b', label="testing loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        
 
 
 
if __name__ == '__main__':
    main()






