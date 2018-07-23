# main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt



# computes the normalized hidden layer
# NOTE: only for computing gradient?
def compute_normalized_hidden(x, A):
    hidden = sparse.csr_matrix.dot(A, x.T)
    
    norm = np.linalg.norm(hidden)
    if norm == 0: 
       return hidden
    return hidden / norm 


# computes the hidden layer
def compute_hidden(x, A):
    hidden = sparse.csr_matrix.dot(A, x.T)
    return hidden
    

# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nclasses, alpha, DIM):
    #hidden = compute_normalized_hidden(x, A)
    hidden = compute_hidden(x, A)  # this one im pretty sure
    y_hat = stable_softmax(x, A, B)
    
    j = 0
    while j < nclasses:
        Bj = B[j, :]
        yj_hat = y_hat[j]
        yj = label[j]

        Bj_new = alpha * np.dot( hidden, (yj_hat - yj))
        Bj_new = np.subtract(Bj, Bj_new)
        
        B[j, :] = Bj_new
        j += 1
    
    #print(B)
    return B
        

# update rule for weight matrix A
def gradient_A1(B, A, x, label, nclasses, alpha, DIM):
    p = A.shape[1]
    Y_hat = stable_softmax(x, A, B)
    
    Y = np.subtract(Y_hat.T, label)
    YB = np.dot(Y, B)
    A_new = np.dot(alpha, sparse.csr_matrix.dot(YB.T, x))

    A = np.subtract(A, A_new)
    
    return A


# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, DIM):
    Y_hat = stable_softmax(x, A, B)
    
    i = 0
    while i < DIM:
        j = 0
        sum_ = 0
        while j < nclasses:
            yhat_nj = Y_hat[j]
            yn = label[j]
            b_ji = B[j, i]
        
            sum_ += ((yhat_nj - yn) * b_ji) 
            j += 1

        Ai_new = alpha * sparse.csr_matrix.dot(sum_, x)
        A[i, :] = np.subtract(A[i, :], Ai_new)

        i += 1    
    
    return A
            

def stable_softmax(x, A, B):
    hidden = compute_hidden(x, A) 
    #hidden = compute_normalized_hidden(x, A) 
    X = np.dot(B, hidden)
    #print(X)
    exps = np.exp(X - np.max(X))
    #print(exps / np.sum(exps))
    return (exps / np.sum(exps))


# finds the loss
def loss_function(x, A, B, label):
    #print(A)
    #print(stable_softmax(x, A, B))
    #print()
    #print()
    loglike = np.log(stable_softmax(x, A, B))
    return np.dot(label, loglike)


# computes the loss over entire dataset
def total_loss_function(X, Y, A, B, N):
    i = 0
    total_loss = 0
    for x in X:
        label = Y[i]
        loss = loss_function(x, A, B, label)
        total_loss += loss
        i += 1
        
    return -1.0/N * total_loss
    
    
    
# function to return prediction ACCURACY
def prediction_accuracy(X, Y, A, B, N):
    correct = 0
    i = 0
    for x in X:
        prediction = np.argmax(stable_softmax(x, A, B))
        label = np.argmax(Y[i])
        if prediction == label:
            correct += 1
        i += 1
        
    return correct / N


# function to return prediction ACCURACY
def prediction_error(X, Y, A, B, N):
    incorrect = 0
    i = 0
    for x in X:
        prediction = np.argmax(stable_softmax(x, A, B))
        label = np.argmax(Y[i])
        if prediction != label:
            incorrect += 1
        i += 1
        
    return incorrect / N
    


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.0001
    WORDGRAMS=3
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1000000
    #BUCKET = 0
    EPOCH=50

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
    np.random.seed(0)
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))
    #print(A.shape)

    # B
    B_n = DIM               # cols
    B_m = nclasses          # rows
    B = np.zeros((B_m, B_n))
    #print(B.shape)


    #### train ################################################

    losses_train = []
    losses_test = []
    print()
    
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        #alpha = LR
        
        l = 0
        # TRAINING
        for x in X_train:
            label = y_train[l]
            B_old = B
            A_old = A
            
            # back prop
            B = gradient_B(B_old, A_old, x, label, nclasses, alpha, DIM)  
            A = gradient_A(B_old, A_old, x, label, nclasses, alpha, DIM)
                  
            l += 1
            
            
        # TRAINING LOSS
        #print("***************** Finding loss *************************")
        train_loss = total_loss_function(X_train, y_train, A, B, N)
        print("Train: ", train_loss)
            
        # TESTING LOSS
        test_loss = total_loss_function(X_test, y_test, A, B, N_test)
        print("Test: ", test_loss)


        train_pred_acc = prediction_accuracy(X_train, y_train, A, B, N)
        test_pred_acc = prediction_accuracy(X_test, y_test, A, B, N_test)

        train_pred_err = prediction_error(X_train, y_train, A, B, N)
        test_pred_err = prediction_error(X_test, y_test, A, B, N_test)
        print("Train prediction accuracy: ", train_pred_acc, " Error: ", train_pred_err)
        print("Test prediction accuracy: ", test_pred_acc, " Error: ", test_pred_err)
        
        
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        
        i += 1
        
        
    
    
    #train_pred_err = prediction_error(X_train, y_train, A, B, N)
    #test_pred_err = prediction_error(X_test, y_test, A, B, N_test)
    #print("Train prediction accuracy: ", train_pred_acc, " Error: ", train_pred_err)
    #print("Test prediction accuracy: ", test_pred_acc, " Error: ", test_pred_err)
    
    # for plotting
    epochs = [l for l in range(EPOCH)]
    plt.plot(epochs, losses_train, 'r', label="training loss")
    plt.plot(epochs, losses_test, 'b', label="testing loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        
 
 
 
if __name__ == '__main__':
    main()






