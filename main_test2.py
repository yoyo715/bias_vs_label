# main.py

#from dictionary import Dictionary
#from dictionary_updated import Dictionary2
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

    
# function to return prediction error, precision, recall, F1 score
def metrics(Y_hat, Y):
    incorrect = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0    
    
    y_true = []
    y_pred = []
    
    print(Y_hat)
    print(Y)

    i = 0
    while i < len(Y_hat):
        prediction = np.argmax(Y_hat[i])
        true_label = np.argmax(Y[i])
        
        y_true.append(true_label)
        y_pred.append(prediction)

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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    print("AUC score: ", roc_auc)

    if true_pos == 0 and false_pos == 0:
        print("WARNING::True pos and False pos both zero")
        precision = true_pos / 0.000001
        recall = true_pos / 0.000001
        F1 = 2 * ((precision * recall) / (precision + recall))
        classification_error = incorrect / N
    else:
        precision = true_pos / (true_pos + false_pos)   # true pos rate (TRP)
        recall = true_pos / (true_pos + false_neg)      # 
        F1 = 2 * ((precision * recall) / (precision + recall))
        classification_error = incorrect / N
        
    print()

    return classification_error, precision, recall, F1, roc_auc, fpr, tpr
    
    
def softmax(X, theta, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)


    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    return p



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
    LR=0.10             # 0.15 good for ~5000
    KERN = 'lin'        # lin or rbf or poly
    NUM_RUNS = 1        # number of test runs
    SUBSET_VAL = 5000   # number of subset instances for self reported dataset
    LIN_C = 0.90        # hyperparameter for linear kernel

    BATCHES = 50         # for batch gradient descent (number of splits of datatset)
    
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
    
    print("A: ", A.shape)
    print("B: ", B.shape)
    print("X_trian: ", X_train.shape)
    print("labels: ", y_train.shape)

    losses_train = []
    losses_test = []
    losses_manual = []

    print()
    print()
    
    X_train = normalize(X_train, axis=1, norm='l1')
    X_test = normalize(X_test, axis=1, norm='l1')
    X_manual = normalize(X_manual, axis=1, norm='l1')

    #X_train_batches = np.vsplit(X_train, BATCHES )
    #X_train = np.toarray(X_train)
    #X_train_batches = np.array_split(X_train.todense(), BATCHES)
    #print("****** ", X_train_batches[0].shape)

    #y_train_batches = np.array_split(y_train, BATCHES)
    
    traintime_start = time.time()
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        alpha = LR * ( 1 - i / EPOCH)  # linearly decaying lr alpha
        train_loss = 0

        batch_num = 0
        for batch in X_train_batches:
        
            # Forward Propogation
            hidden = sparse.csr_matrix.dot(A, batch.T)
            hidden = normalize(hidden, axis=1, norm='l1')
            z2 = np.dot(B, hidden)
            
            # softmax
            Y_hat = softmax(z2, theta = 1.0, axis = 0)
            #loglike = np.log(Y_hat)
            #train_loss = -np.multiply(y_train_batches[batch_num], loglike.T)  # need to multiply element wise here

            #### Back prop #########################################################
            # B update
            gradient = alpha * np.dot(np.subtract(Y_hat.T, y_train_batches[batch_num]).T, hidden.T)
            B = np.subtract(B, gradient)

            # A update
            first = np.dot(np.subtract(Y_hat.T, y_train_batches[batch_num]), B)
            gradient = alpha * sparse.csr_matrix.dot(first.T, batch)
            A = np.subtract(A, gradient) 

            batch_num += 1
 

        # TRAINING LOSS
        #train_loss = np.sum(train_loss)/N_train
        hidden_train = sparse.csr_matrix.dot(A, X_train.T)
        hidden_train = normalize(hidden_train, axis=1, norm='l1')
        z2_train = np.dot(B, hidden_train)
        
        Y_hat_train = softmax(z2_train, theta = 1.0, axis = 0)
        loglike_train = np.log(Y_hat_train)
        train_loss = -np.multiply(y_train, loglike_train.T)  # need to multiply element wise here
        train_loss = np.sum(train_loss)/N_train
        print("Train:   ", train_loss)
        
        ## TESTING LOSS
        hidden_test = sparse.csr_matrix.dot(A, X_test.T)
        hidden_test = normalize(hidden_test, axis=1, norm='l1')
        z2_test = np.dot(B, hidden_test)
        
        Y_hat_test = softmax(z2_test, theta = 1.0, axis = 0)
        loglike_test = np.log(Y_hat_test)
        test_loss = -np.multiply(y_test, loglike_test.T)  # need to multiply element wise here
        test_loss = np.sum(test_loss)/N_test
        print("Test:    ", test_loss)
        
        ## MANUAL SET TESTING LOSS
        hidden_man = sparse.csr_matrix.dot(A, X_manual.T)
        hidden_man = normalize(hidden_man, axis=1, norm='l1')
        z2_man = np.dot(B, hidden_man)
        
        Y_hat_man = softmax(z2_man, theta = 1.0, axis = 0)
        loglike_manual = np.log(Y_hat_man)
        manual_loss = -np.multiply(y_manual, loglike_manual.T)  # need to multiply element wise here
        manual_loss = np.sum(manual_loss)/N_manual
        
        print("Manual Set:    ", manual_loss)
        
        #### Back prop #########################################################
        # B update
        #gradient = alpha * np.dot(np.subtract(Y_hat.T, y_train).T, hidden.T)
        #B = np.subtract(B, gradient)

        # A update
        #first = np.dot(np.subtract(Y_hat.T, y_train), B)
        #gradient = alpha * sparse.csr_matrix.dot(first.T, X_train)
        #A = np.subtract(A, gradient) 
        
        

        #train_class_error, train_precision, train_recall, train_F1, train_AUC, train_FPR, train_TPR = metrics(Y_hat, y_train)
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
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        

    
 
 
 
if __name__ == '__main__':
    main()






