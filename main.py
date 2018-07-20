# main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt

#np.seterr(divide='ignore', invalid='ignore')  # for RuntimeWarning: invalid value encountered in true_divide error

# computes the hidden layer
def compute_hidden(x, A):
    #hidden = sparse.csr_matrix.dot(x, A)
    hidden = sparse.csr_matrix.dot(A, x.T)
    
    norm = np.linalg.norm(hidden)
    if norm == 0: 
       return hidden
    return hidden / norm 


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
def gradient_A(B, A, x, label, nlabels, alpha):
    j = 0
    p = A.shape[1]
    y_hat = stable_softmax(x, A, B)
    A_new = np.zeros((30, p))
    while j < nlabels:
        Bj = np.reshape(B[j, :], (30, 1))
        yj_hat = y_hat[j]
        yj = label[j]
        
        a = (yj_hat - yj)* sparse.csr_matrix.dot(Bj, x)
        A_new = np.add(A_new, a)
        j += 1
        
    A = A - alpha * A_new
    return A
            

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
    LR=0.01
    WORDGRAMS=2
    MINCOUNT=2
    MINN=3
    MAXN=3
    #BUCKET=1 #000000
    BUCKET = 0
    EPOCH=20

    #train = open('/local_d/RESEARCH/simple-queries/data/query_gender.train', 'r')
    #test = open('/local_d/RESEARCH/simple-queries/data/query_gender.test', 'r')
    
    #train = open('../cleaned_train_subset.txt', 'r')
    #test = open('../cleaned_test_subset.txt', 'r')
    
    dataset = open('../cleaned_subset.txt', 'r')
    
    print("starting dictionary creation") 
    
    # initialize training
    #dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
    dictionary = Dictionary(dataset, WORDGRAMS, MINCOUNT)
    #input_ = dictionary.get_bagngram()
    #labels = dictionary.get_labels()
    nwords = dictionary.get_nwords()
    nlabels = dictionary.get_nlabels()
    #N = dictionary.get_ninstances()
    
    #initialize testing
    X_train, X_test, y_train, y_test = dictionary.train_and_testsplit()
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    #dictionary.create_test_instances(test)
    #input_test = dictionary.create_test_bagngrams()
    #N_test = dictionary.get_test_ninstances()
    #labels_test = dictionary.get_test_labels()
    
    print(N, " number of Train instances. ", N_test, " number of Test instances")
    #print("Train: F,M ", dictionary.get_nlabels_eachclass_train())
    #print("Test: F,M ", dictionary.get_nlabels_eachclass_test())
    
    
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
    print()
    
    for i in range(EPOCH):
        print("EPOCH: ", i)
        # loop through each instance for SGD
        loss = 0
        l = 0
        total_loss = 0
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        # TRAINING
        #for x in input_:
        for x in X_train:
            #print(len(x.data))
            
            #label = labels[l]
            label = y_train[l]
            B_old = B
            A_old = A
            
            # forward prop
            loss = loss_function(x, A, B, label) 
            
            # back prop
            B = gradient_B(B_old, A, x, label, nlabels, alpha)  
            A = gradient_A(B_old, A_old, x, label, nlabels, alpha)
            
            total_loss += loss        
            l += 1
            
            
        # TESTING 
        q = 0
        total_loss_test = 0
        #for xtest in input_test:
        for xtest in X_test:
            #label_test = labels_test[q]
            label_test = y_test[q]
            loss_test = loss_function(xtest, A, B, label_test) 
            total_loss_test += loss_test
            q += 1
            
            
        losses_test.append(total_loss_test/N_test * -1)
        losses.append(total_loss/N * -1)
        i += 1
        
        
    print("train: ", losses)
    print("test: ", losses_test)
    
    epochs = [l for l in range(EPOCH)]
    
    plt.plot(epochs, losses, 'r', label="training loss")
    plt.plot(epochs, losses_test, 'b', label="testing loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        
 
 
if __name__ == '__main__':
    main()





