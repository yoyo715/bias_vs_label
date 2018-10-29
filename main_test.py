# main.py

from dictionary3 import Dictionary

import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize


'''
    This script is to test the implementation to compare to KMM2.py

'''


# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nclasses, alpha, DIM, hidden, Y_hat):    
    gradient = alpha * np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
    B_new = np.subtract(B, gradient)

    return B_new


# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, DIM, Y_hat):
    A_old = A
    first = np.dot(np.subtract(Y_hat.T, label), B)
    
    if np.sum(x) > 0:
        sec = x * (1.0/np.sum(x))
    else:
        sec = x

    gradient = alpha * sparse.csr_matrix.dot(first.T, sec)
    A = np.subtract(A_old, gradient) 
    
    return A



def stable_softmax(X): 
    #hidden = compute_normalized_hidden(x, A) 
    #X = np.dot(B, hidden)
    #exps = np.exp(X - np.max(X))
    #return (exps / np.sum(exps))

    axis = 0  # across rows

    # subtract the max for numerical stability
    X = X - np.expand_dims(np.max(X, axis = axis), axis)
    
    # exponentiate y
    X = np.exp(X)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)

    # finally: divide elementwise
    p = X / ax_sum

    return p


# calculates total loss using matrix operations (quicker than looping)
def get_total_loss(A, B, X, y, N):
    hidden = sparse.csr_matrix.dot(A, X.T)      
    
    sum_ = np.sum(X, axis = 1)
    sum_[sum_ == 0] = 1         # replace zeros with ones so divide will work
    sum_ = np.array(sum_)
    sum_ = sum_.flatten()
    
    hidden = hidden.T / sum_[:,None]
    hidden = hidden.T
    z2 = np.dot(B, hidden)
    
    Y_hat = stable_softmax(z2)
    loglike = np.log(Y_hat)
    
    loss = -np.multiply(y, loglike.T)  # need to multiply element wise here
    loss = np.sum(loss)/N
    
    return loss


# function to return prediction error, precision, recall, F1 score
def metrics(X, Y, A, B, N):
    incorrect = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0    
    
    # get predicted classes
    hidden = sparse.csr_matrix.dot(A, X.T)        
    sum_ = np.sum(X, axis = 1)
    sum_[sum_ == 0] = 1         # replace zeros with ones so divide will work
    sum_ = np.array(sum_).flatten()
    
    a1 = (hidden.T / sum_[:,None]).T
    z2 = np.dot(B, a1)
    Y_hat = stable_softmax(z2)

    
    # compare to actual classes
    prediction = np.argmax(Y_hat, axis=1)
    true_label = np.argmax(Y, axis=1)
    
    
    #if prediction != true_label:
        #incorrect += 1

    #if prediction == 1 and true_label == 1:
        #true_pos += 1

    #if prediction == 1 and true_label == 0:
        #false_pos += 1

    #if prediction == 0 and true_label == 0:
        #true_neg += 1

    #if prediction == 0 and true_label == 1:
        #false_neg += 1


    #print("confusion matrix: ")
    #print("[ ", true_neg, false_pos, " ]")
    #print("[ ", false_neg, true_pos, " ]")
    
    
    
    ## Compute fpr, tpr, thresholds and roc auc
    #fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    #roc_auc = auc(fpr, tpr)
    
    #print("AUC score: ", roc_auc)

    #if true_pos == 0 and false_pos == 0:
        #print("WARNING::True pos and False pos both zero")
        #precision = true_pos / 0.000001
        #recall = true_pos / 0.000001
        #F1 = 2 * ((precision * recall) / (precision + recall))
        #classification_error = incorrect / N
    #else:
        #precision = true_pos / (true_pos + false_pos)   # true pos rate (TRP)
        #recall = true_pos / (true_pos + false_neg)      # 
        #F1 = 2 * ((precision * recall) / (precision + recall))
        #classification_error = incorrect / N
        
    #print()

    return classification_error  #, precision, recall, F1, roc_auc, fpr, tpr
    
    


def main():

    # args from Simple Queries paper
    DIM=30
    WORDGRAMS=2
    MINCOUNT=8
    MINN=3
    MAXN=3
    BUCKET=1000000

    # adjust these
    EPOCH=20
    LR=0.15             # 0.15 good for ~5000
    KERN = 'lin'        # lin or rbf or poly
    NUM_RUNS = 1        # number of test runs
    SUBSET_VAL = 5000   # number of subset instances for self reported dataset
    LIN_C = 0.90        # hyperparameter for linear kernel
    
    BATCHSIZE = 1       # number of instances in each batch
    
    
    ##### instantiations #######################################
        
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    # initialize training
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, model='original')
    end = time.time()
    print("dictionary took ", (end - start)/60.0, " time to create.")
    nwords = dictionary.get_nwords()
    nclasses = dictionary.get_nclasses()
    
    #initialize testing
    X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    N_train = dictionary.get_n_train_instances()
    N_test = dictionary.get_n_test_instances()
    
    print("Number of Train instances: ", N_train, " Number of Test instances: ", N_test)
    ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
    ntest_eachclass = dictionary.get_nlabels_eachclass_test()
    print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)
    
    
    # manual labeled set (Kaggle dataset)
    X_manual = dictionary.get_manual_testset()
    y_manual = dictionary.get_manual_set_labels()
    N_manual = dictionary.get_n_manual_instances()
    print()
    print("Number of Manual testing instances: ", N_manual, " shape: ", X_manual.shape)
    nmanual_eachclass = dictionary.get_nlabels_eachclass_manual()
    print("N each class Manual testing instances: ", nmanual_eachclass)
    print("#####################################")
    
    
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



    #### train ################################################

    losses_train = []
    losses_test = []
    losses_manual = []

    print()
    print()
    
    traintime_start = time.time()
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        l = 0
        train_loss = 0
        
        start = 0
        batchnum = 0
        while start <= N_train:
            batch = X_train.tocsr()[start:start+BATCHSIZE, :]
            y_train_batch = y_train[start:start+BATCHSIZE, :] 

            B_old = B
            A_old = A
            
            # Forward Propogation
            hidden = sparse.csr_matrix.dot(A, batch.T)
            #if np.sum(x) > 0:
                #a1 = hidden * (1.0 / np.sum(x))  # axis = 1 across rows
            #else:
                #a1 = hidden
                
            sum_ = np.sum(batch, axis = 1)
            sum_[sum_ == 0] = 1         # replace zeros with ones so divide will work
            sum_ = np.array(sum_).flatten()
            
            a1 = (hidden.T / sum_[:,None]).T
                
            z2 = np.dot(B, a1)
            Y_hat = stable_softmax(z2)
    
            # Back prop with alt optimization
            B = gradient_B(B_old, A_old, batch, y_train_batch, nclasses, alpha, DIM, a1, Y_hat)  
            
            A = gradient_A(B_old, A_old, batch, y_train_batch, nclasses, alpha, DIM, Y_hat)

            
            #loglike = np.log(Y_hat)
            #train_loss += -np.dot(y_train_batch, loglike)
            
            batchnum += 1

            # NOTE figure this out, Might be missing last sample
            if start+BATCHSIZE >= N_train and start < N_train-1:   

                batch = X_train.tocsr()[start:-1, :]   # rest of train set
                y_train_batch = y_train[start:-1, :] 
                
                B_old = B
                A_old = A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(A, batch.T)
                
                sum_ = np.sum(batch, axis = 1)
                sum_[sum_ == 0] = 1         # replace zeros with ones so divide will work
                sum_ = np.array(sum_).flatten()
                
                a1 = (hidden.T  / sum_[:,None]).T
                
                #if np.sum(x) > 0:
                    #a1 = hidden * (1.0 / np.sum(x))  # axis = 1 across rows
                #else:
                    #a1 = hidden
                    
                z2 = np.dot(B, a1)
                Y_hat = stable_softmax(z2)
        
                # Back prop with alt optimization
                B = gradient_B(B_old, A_old, batch, y_train_batch, nclasses, alpha, DIM, a1, Y_hat)  
                
                A = gradient_A(B_old, A_old, batch, y_train_batch, nclasses, alpha, DIM, Y_hat)
            
                #loglike = np.log(Y_hat)
                #train_loss += -np.dot(y_train_batch, loglike)

                break
            else:
                start = start + BATCHSIZE

            
        # TRAINING LOSS
        #train_loss = train_loss * (1.0/N_train)
        #print("Train:   ", train_loss)

        train_loss = get_total_loss(A, B, X_train, y_train, N_train)
        print("Train:   ", train_loss)

        ## TESTING LOSS
        test_loss = get_total_loss(A, B, X_test, y_test, N_test)
        print("Test:    ", test_loss)
        
        #print("Difference = ", test_loss - train_loss)
        
        ## MANUAL SET TESTING LOSS
        manual_loss = get_total_loss(A, B, X_manual, y_manual, N_manual)
        print("Manual Set:    ", manual_loss)


        #train_class_error, train_precision, train_recall, train_F1, train_AUC, train_FPR, train_TPR = metrics(X_train, y_train, A, B, N_train)
        #test_class_error, test_precision, test_recall, test_F1, test_AUC, test_FPR, test_TPR = metrics(X_test, y_test, A, B, N_test)
        #manual_class_error, manual_precision, manual_recall, manual_F1, manual_AUC, manual_FPR, manual_TPR = metrics(X_manual, y_manual, A, B, N_manual)
        
        #print()
        #print("TRAIN:")
        #print("         Classification Err: ", train_class_error)
        #print("         Precision:          ", train_precision)
        #print("         Recall:             ", train_recall)
        #print("         F1:                 ", train_F1)

        #print("TEST:")
        #print("         Classification Err: ", test_class_error)
        #print("         Precision:          ", test_precision)
        #print("         Recall:             ", test_recall)
        #print("         F1:                 ", test_F1)
        
        #print()
        #print("MANUAL:")
        #print("         Classification Err: ", manual_class_error)
        #print("         Precision:          ", manual_precision)
        #print("         Recall:             ", manual_recall)
        #print("         F1:                 ", manual_F1)
        
        losses_train.append(train_loss)
        losses_test.append(test_loss)
        losses_manual.append(manual_loss)

        
        i += 1
    traintime_end = time.time()
    
    print("model took ", (traintime_end - traintime_start)/60.0, " time to train")

    epochs = [l for l in range(EPOCH)]
    
    plt.plot(epochs, losses_train, 'm', label="train")
    plt.plot(epochs, losses_test, 'c', label="test")
    plt.plot(epochs, losses_manual, 'g', label="manual")
    title = "Main_temp: n_train: ", N_train, " n_test: ", N_test, " n_manual ", N_manual
    #title = "Main_temp: n_train: ", N_train, " n_test: ", N_test
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        

    
 
 
 
if __name__ == '__main__':
    main()






