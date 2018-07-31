# main.py

from dictionary import Dictionary
from dictionary_updated import Dictionary2
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
def gradient_B(B, A, x, label, nclasses, alpha, DIM, hidden, Y_hat):
    j = 0
    while j < nclasses:
        Bj = B[j, :]
        yj_hat = Y_hat[j]
        yj = label[j]

        Bj_new = np.multiply( (alpha *(yj_hat - yj)), hidden.T )
        Bj_new = np.subtract(Bj, Bj_new)
        
        B[j, :] = Bj_new
        j += 1
    
    return B


# this fuction checks the gradient of B
def check_B_gradient(B, A, label, x, nclasses):
    #print("**Checking B gradient")
    
    #find backprop derivative of B
    hidden = compute_normalized_hidden(x, A)
    #hidden = compute_hidden(x, A)  # this one im pretty sure?
    Y_hat = stable_softmax(x, A, B)

    j = 0
    
    B_grad = B.copy()
    while j < nclasses:
        Bj = B[j, :]
        yj_hat = Y_hat[j]
        yj = label[j]

        Bj_new = np.multiply( (yj_hat - yj), hidden.T )
        
        B_grad[j, :] = Bj_new
        j += 1
    
    eps = 0.0001
    
    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            
            # Copy the parameter matrix and change the current parameter slightly
            B_matrix_min = B.copy()
            B_matrix_min[row,col] -= eps
            B_matrix_plus = B.copy()
            B_matrix_plus[row,col] += eps
            
            # Compute the numerical gradient
            grad_num = (loss_function(x, A, B_matrix_plus, label) - loss_function(x, A, B_matrix_min, label))/(2*eps)
            
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, B_grad[row,col]):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(B_grad[row,col])))
            
    print('No B gradient errors found')


# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, DIM, Y_hat):
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

        #x_2 = x / np.sum(x)
        Ai_new =  sparse.csr_matrix.dot(sum_, x)
        A[i, :] = alpha * np.subtract(A[i, :], Ai_new)

        i += 1    

    return A


# this fuction checks the gradient of A
def check_A_gradient(B, A, label, x, nclasses, DIM):
    print("**Checking A gradient")
    
    # find backprop derivative of A
    Y_hat = stable_softmax(x, A, B)
    
    i = 0
    A_grad = A.copy()
    while i < DIM:
        j = 0
        sum_ = 0
        while j < nclasses:
            yhat_nj = Y_hat[j]
            yn = label[j]
            b_ji = B[j, i]
        
            sum_ += ((yhat_nj - yn) * b_ji) 
            j += 1

        #x_2 = x / np.sum(x)
        Ai_new = sparse.csr_matrix.dot(sum_, x)
        A_grad[i, :] =  Ai_new

        i += 1    

    
    eps = 0.0001
    
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            # Copy the parameter matrix and change the current parameter slightly
            A_matrix_min = A.copy()
            A_matrix_min[row,col] -= eps
            A_matrix_plus = A.copy()
            A_matrix_plus[row,col] += eps
            
            # Compute the numerical gradient
            grad_num = (loss_function(x, A_matrix_plus, B, label) - loss_function(x, A_matrix_min, B, label))/(2*eps)
            
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, A_grad[row,col]):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(A_grad[row,col])))
            
    print('No A gradient errors found')
            

def stable_softmax(x, A, B):
    hidden = compute_hidden(x, A)  
    X = np.dot(B, hidden)
    exps = np.exp(X - np.max(X))
    return (exps / np.sum(exps))


# finds the loss
def loss_function(x, A, B, label):
    loglike = np.log(stable_softmax(x, A, B))
    return - np.dot(label, loglike)


# computes the loss over entire dataset
def total_loss_function(X, Y, A, B, N):
    i = 0
    total_loss = 0
    for x in X:
        label = Y[i]
        loss = loss_function(x, A, B, label)
        total_loss += loss
        i += 1
        
    return (1.0/N) * total_loss


# function to return prediction error, precision, recall, F1 score
def metrics(X, Y, A, B, N):
    incorrect = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0    

    i = 0
    for x in X:
        prediction = np.argmax(stable_softmax(x, A, B))
        true_label = np.argmax(Y[i])

        if prediction != true_label:
            incorrect += 1

        if prediction == 1 and true_label == 1:
            true_pos += 1

        if prediction == 1 and true_label == 0:
            false_pos += 1

        if prediction == 0 and true_label == 0:
            true_neg += 1

        if prediction == 0 and true_label == 1:
            false_neg += 1
    
        i += 1
        
    print("confusion matrix: ")
    print("[ ", true_neg, false_pos, " ]")
    print("[ ", false_neg, true_pos, " ]")

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    F1 = 2 * ((precision * recall) / (precision + recall))
    classification_error = incorrect / N

    return classification_error, precision, recall, F1
    


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
    
    ##### instantiations #######################################


    p = X_train.shape[1]
    
    # A
    #A_n = nwords + BUCKET   # cols
    A_n = p
    A_m = DIM               # rows
    uniform_val = 1.0 / DIM
    np.random.seed(0)
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))

    # B
    B_n = DIM               # cols
    B_m = nclasses          # rows
    B = np.zeros((B_m, B_n))
    #B = np.random.uniform(-uniform_val, uniform_val, (B_m, B_n))


    #### train ################################################

    losses_train = []
    losses_test = []

    class_error_train = []  
    class_error_test = []

    prec_train = [] 
    prec_test = []

    recall_train = []
    recall_test = []

    F1_train = []
    F1_test = []

    print()
    print()
    
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        l = 0
        
        # TRAINING
        for x in X_train:       
            #print(x.toarray())
            
            label = y_train[l]
            B_old = B
            A_old = A
            
            # Forward Propogation
            hidden = sparse.csr_matrix.dot(A, x.T)
            #norm = np.linalg.norm(hidden)
            #if norm == 0: 
                #a1 = hidden
            #else:
                #sumtest = np.sum(x)
                ##print("** ", sumtest, norm)
                #a1 = hidden / sumtest 
                
            a1 = hidden  #/ np.sum(x)
                
            z2 = np.dot(B, a1)
            exps = np.exp(z2 - np.max(z2))
            Y_hat = exps / np.sum(exps)
            
            
            # Back prop with alt optimization
            B = gradient_B(B_old, A_old, x, label, nclasses, alpha, DIM, a1, Y_hat)  
            A = gradient_A(B_old, A_old, x, label, nclasses, alpha, DIM, Y_hat)
            
            
            # verify gradients
            #check_B_gradient(B_old, A_old, label, x, nclasses)
            #check_A_gradient(B_old, A_old, label, x, nclasses, DIM)
   
            l += 1
            
            
        # TRAINING LOSS
        train_loss = total_loss_function(X_train, y_train, A, B, N)
        print("Train:   ", train_loss)
            
        # TESTING LOSS
        test_loss = total_loss_function(X_test, y_test, A, B, N_test)
        print("Test:    ", test_loss)


        train_class_error, train_precision, train_recall, train_F1 = metrics(X_train, y_train, A, B, N)
        test_class_error, test_precision, test_recall, test_F1 = metrics(X_test, y_test, A, B, N_test)
        
        print()
        print("TRAIN:")
        print("         Classification Err: ", train_class_error)
        print("         Precision:          ", train_precision)
        print("         Recall:             ", train_recall)
        print("         F1:                 ", train_F1)

        print("TEST:")
        print("         Classification Err: ", test_class_error)
        print("         Precision:          ", test_precision)
        print("         Recall:             ", test_recall)
        print("         F1:                 ", test_F1)
        
        losses_train.append(train_loss)
        losses_test.append(test_loss)

        class_error_train.append(train_class_error)
        class_error_test.append(test_class_error)

        prec_train.append(train_precision)
        prec_test.append(test_precision)

        recall_train.append(train_recall)
        recall_test.append(test_recall)

        F1_train.append(train_F1)
        F1_test.append(test_F1)
        
        i += 1
        
        
    
    # for plotting
    epochs = [l for l in range(EPOCH)]

    plt.plot(epochs, losses_train, 'r', label="training loss")
    plt.plot(epochs, losses_test, 'b', label="testing loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(epochs, class_error_train, 'r', label="training classification err")
    plt.plot(epochs, class_error_test, 'b', label="testing classification err")
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        
    plt.plot(epochs,  F1_train, 'm', label="training F1 score")
    plt.plot(epochs, F1_test, 'c', label="testing F1 score")
    plt.ylabel('F1 Score')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(recall_train, prec_train, 'm', label="training")
    plt.plot(recall_test, prec_test, 'c', label="testing")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='upper left')
    plt.show()
 
 
 
if __name__ == '__main__':
    main()






