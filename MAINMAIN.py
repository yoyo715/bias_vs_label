# MAINMAIN.py

"""
    This script will run all experiments on fastText and fastKMMText.
    
"""

from dictionary3 import Dictionary

import numpy as np
import pandas as pd
import pickle, os
from scipy import sparse
import time
import sys
import warnings
import argparse

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support

# Method to get arguments
# User must input a source and destination and a log file
def get_args():
    parser = argparse.ArgumentParser(description='Enter trial number')
    
    parser.add_argument('-r', "--run", action='store', help="trial number", required=True)

    args = vars(parser.parse_args())

    return args



# model_version: 'original' or 'kmm;
def create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model_version):
    
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    # initialize training
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model=model_version)
    end = time.time()
    print("dictionary took ", (end - start)/60.0, " time to create.")
    
    return dictionary


# writing model specifications to an about file
def create_readme(DIM, WORDGRAMS, MINCOUNT, MINN, MAXN, BUCKET, EPOCH, LR, KMMLR, NUM_RUNS, SUBSET_VAL, KERN, LIN_C, BATCHSIZE):
    with open('/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/output2/README.md ', 'a+') as f:
        f.write('# Original Model specifications # \n\n')
        f.write('DIM: ' + str(DIM))
        f.write('\n\n')
        f.write('WORDGRAMS: ' + str(WORDGRAMS))
        f.write('\n\n')
        f.write('MINCOUNT: ' + str(MINCOUNT))
        f.write('\n\n')
        f.write('MINN: ' + str(MINN))
        f.write('\n\n')
        f.write('MAXN: ' + str(MAXN))
        f.write('\n\n')
        f.write('BUCKET: ' + str(BUCKET))
        f.write('\n\n')
        f.write('\n\n')
        f.write('\n\n')
        
        # Hyperparameters
        f.write('## Hyperparameters ##\n\n')
        f.write('EPOCH: ' + str(EPOCH))
        f.write('\n\n')
        f.write('LR: ' + str(LR))
        f.write('\n\n')
        f.write('KMMLR: ' + str(KMMLR))
        f.write('\n\n')
        f.write('NUM_RUNS: ' +  str(NUM_RUNS))
        f.write('\n\n')
        f.write('SUBSET_VAL: ' + str(SUBSET_VAL))
        
        f.write('\n\n')
        f.write('KERN: ' + str(KERN))
        f.write('\n\n')
        f.write('LINEAR KERNEL hyperparameter C: ' + str(LIN_C))
        f.write('\n\n')
        f.write('BATCH SIZE: ' + str(BATCHSIZE))
        

# writes the predicted labels to a file
def write_labels_tofile(fname, Y_true, Y_pred):
    #np.savetxt(fname, (Y_true.T, Y_pred), delimiter=',', newline='\n', fmt='%s')
    
    data = { 'Y_true': Y_true.T,
                'Y_predicted': Y_pred,
            }
    output = open(fname, 'wb')
    pickle.dump(data, output)
    output.close()
    
  
# writes the model to a file to be used again later
def save_model_tofile(A, B, fnameB, fnameA):
    #data = { 'A': A,
             #'B': B,
            #}
    #output = open(fname, 'wb')
    #pickle.dump(data, output)
    #output.close()
    #sparse.save_npz(fname, A)
    with open(fnameA, 'a+') as f:
        np.savetxt(f, A, fmt="%f", delimiter=",")
    
    with open(fnameB, 'a+') as f:
        np.savetxt(f, B, fmt="%f", delimiter=",")
         
        
# function to return prediction error, precision, recall, F1 score
def metrics(X, Y, A, B, N, test, trialnum, epoch):
    # get predicted classes
    hidden = sparse.csr_matrix.dot(A, X.T)        
    a1 = normalize(hidden, axis=0, norm='l1')
    z2 = np.dot(B, a1)
    Y_hat = stable_softmax(z2)

    # compare to actual classes
    prediction_max = np.argmax(Y_hat, axis=0)
    true_label_max = np.argmax(Y, axis=1)

    
    class_error = np.sum(true_label_max != prediction_max.T) * 1.0 / N
    class_acc = np.sum(true_label_max == prediction_max.T) * 1.0 / N
    
    if ( class_error + class_acc ) != 1:
        print("ERROR in computing class errror")
    
    #print(confusion_matrix(true_label_max, prediction_max))

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max, labels=[0,1]).ravel()
    
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(true_label_max, prediction_max)
    roc_auc = auc(fpr, tpr)
    
    #print("AUC score: ", roc_auc)
    #print()

    precision = true_pos / (true_pos + false_pos)           # true pos rate (TRP)
    recall = true_pos / (true_pos + false_neg)              # 
    F1 = 2 * ((precision * recall) / (precision + recall))
    
    
    # write labels to a file for future use
    #if test == 'train':
        #fname = 'label_output/train_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    #elif test == 'test':
        #fname = 'label_output/test_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    #elif test == 'manual':
        #fname = 'label_output/manual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
        
    #elif test == 'KMMtrain':
        #fname = 'KMMlabel_output/kmmtrain_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    #elif test == 'KMMtest':
        #fname = 'KMMlabel_output/kmmtest_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    #elif test == 'KMMmanual':
        #fname = 'KMMlabel_output/kmmmanual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    
    #write_labels_tofile(fname, Y, Y_hat)
    
    dir_ = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/'
    
    # TETON
    if test == 'train':
        fname = dir_+'label_output/train_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'test':
        fname = dir_+'label_output/test_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'manual':
        fname = dir_+'label_output/manual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
        
    elif test == 'KMMtrain':
        fname = dir_+'KMMlabel_output/kmmtrain_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'KMMtest':
        fname = dir_+'KMMlabel_output/kmmtest_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'KMMmanual':
        fname = dir_+'KMMlabel_output/kmmmanual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    
    write_labels_tofile(fname, Y, Y_hat)

    return class_error, precision, recall, F1, roc_auc, fpr, tpr


def stable_softmax(X): 
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
    
    a1 = normalize(hidden, axis=0, norm='l1')
    z2 = np.dot(B, a1)
    
    Y_hat = stable_softmax(z2)
    loglike = np.log(Y_hat)
    
    loss = -np.multiply(y, loglike.T)  # need to multiply element wise here
    loss = np.sum(loss)/N
    
    return loss



def write_stats(directory, name, loss, class_error, precision, recall, F1, AUC):
    
    #### WRITING LOSSES
    with open(directory+name+'_loss.txt', '+a') as f:
        f.write("%s," % loss)
            
    #### WRITING ERROR
    with open(directory+name+'_error.txt', '+a') as f:
        f.write("%s," % class_error)
            
    #### WRITING PRECISION
    with open(directory+name+'_precision.txt', '+a') as f:
        f.write("%s," % precision)
            
    #### WRITING RECALL
    with open(directory+name+'_recall.txt', '+a') as f:
        f.write("%s," % recall)
            
    #### WRITING F1
    with open(directory+name+'_F1.txt', '+a') as f:
        f.write("%s," % F1)
            
    #### WRITING AUC
    with open(directory+name+'_AUC.txt', '+a') as f:
        f.write("%s," % AUC)


###################################################################################################


###################################################################################################
"""   
        fasttext code below
"""
###################################################################################################
      
      
# finds gradient of B and returns an up
def gradient_B(B, A, label, alpha, hidden, Y_hat):    
    gradient = alpha * np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
    B_new = np.subtract(B, gradient)

    return B_new


# update rule for weight matrix A
def gradient_A(B, A, X, label, alpha, Y_hat):
    A_old = A
    first = np.dot(np.subtract(Y_hat.T, label), B)
    gradient = alpha * sparse.csr_matrix.dot(first.T, X)
    
    A = np.subtract(A_old, gradient) 
    
    return A        
        
        
def train_fasttext(EPOCH, LR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test,
                   y_manual, nclasses, A, B, N_train, N_test, N_manual, trialnum, dictionary):
    
    #### train ################################################

    losses_train = []
    losses_test = []
    losses_manual = []

    classerr_train = []
    classerr_test = []
    classerr_manual = []

    print()
    print()
    
    X_train = normalize(X_train, axis=1, norm='l1')
    X_test = normalize(X_test, axis=1, norm='l1')
    X_manual = normalize(X_manual, axis=1, norm='l1')
    
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
            a1 = normalize(hidden, axis=0, norm='l1')
            z2 = np.dot(B, a1)
            Y_hat = stable_softmax(z2)
    
            # Back prop with alt optimization
            B = gradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat)  
            A = gradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat)
            
            batchnum += 1

            # NOTE figure this out, Might be missing last sample
            if start+BATCHSIZE >= N_train and start < N_train-1:   
                batch = X_train.tocsr()[start:-1, :]   # rest of train set
                y_train_batch = y_train[start:-1, :] 
                
                B_old = B
                A_old = A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(A, batch.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(B, a1)
                Y_hat = stable_softmax(z2)
        
                # Back prop with alt optimization
                B = gradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat)  
                A = gradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat)

                break
            else:
                start = start + BATCHSIZE

            
        # TRAINING LOSS
        train_loss = get_total_loss(A, B, X_train, y_train, N_train)
        print("Train:   ", train_loss)

        ## TESTING LOSS
        test_loss = get_total_loss(A, B, X_test, y_test, N_test)
        print("Test:    ", test_loss)
        
        #print("Difference = ", test_loss - train_loss)
        
        ## MANUAL SET TESTING LOSS
        manual_loss = get_total_loss(A, B, X_manual, y_manual, N_manual)
        print("Manual Set:    ", manual_loss)
        print()

        losses_train.append(train_loss)
        losses_test.append(test_loss)
        losses_manual.append(manual_loss)
        
        train_class_error, train_precision, train_recall, train_F1, train_AUC, train_FPR, train_TPR = metrics(X_train, y_train, A, B, N_train, 'train', trialnum, i)
        
        test_class_error, test_precision, test_recall, test_F1, test_AUC, test_FPR, test_TPR = metrics(X_test, y_test, A, B, N_test, 'test', trialnum, i)
        
        manual_class_error, manual_precision, manual_recall, manual_F1, manual_AUC, manual_FPR, manual_TPR = metrics(X_manual, y_manual, A, B, N_manual, 'manual', trialnum, i)

        classerr_train.append(train_class_error)
        classerr_test.append(test_class_error)
        classerr_manual.append(manual_class_error)
        
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
        
        print()
        print("MANUAL:")
        print("         Classification Err: ", manual_class_error)
        print("         Precision:          ", manual_precision)
        print("         Recall:             ", manual_recall)
        print("         F1:                 ", manual_F1)
        
        
        write_fasttext_stats(trialnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                             train_AUC, test_loss, test_class_error, test_precision, test_recall,
                             test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                             manual_recall, manual_F1, manual_AUC)
        
        #fname = "./models/fasttext_trial"+str(trialnum)+"epoch"+str(i)+".pkl"
        #save_model_tofile(A, B, fname)
        
        fnameB = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/models/fasttext_trial_B_"+str(trialnum)+"epoch"+str(i)  #+".pkl"
        fnameA = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/models/fasttext_trial_A_"+str(trialnum)+"epoch"+str(i)
        save_model_tofile(A, B, fnameB, fnameA)
        
        
        i += 1
        
        
    traintime_end = time.time()
    
    print("model took ", (traintime_end - traintime_start)/60.0, " time to train")
    
    
    return losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual



def write_fasttext_stats(runnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                         train_AUC, test_loss, test_class_error, test_precision, test_recall,
                         test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                         manual_recall, manual_F1, manual_AUC):
    
    d = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/'
    #### WRITING LOSSES
    with open(d+'output2/loss_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_loss)
            
    with open(d+'output2/loss_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_loss)
            
    with open(d+'output2/loss_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_loss)
            
    #### WRITING ERROR
    with open(d+'output2/error_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_class_error)
    
    with open(d+'output2/error_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_class_error)
            
    with open(d+'output2/error_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_class_error)
            
    #### WRITING PRECISION
    with open(d+'output2/precision_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_precision)
            
    with open(d+'output2/precision_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_precision)
            
    with open(d+'output2/precision_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_precision)
            
    #### WRITING RECALL
    with open(d+'output2/recall_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_recall)
            
    with open(d+'output2/recall_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_recall)
            
    with open(d+'output2/recall_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_recall)
            
    #### WRITING F1
    with open(d+'output2/F1_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_F1)
            
    with open(d+'output2/F1_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_F1)
            
    with open(d+'output2/F1_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_F1)
            
    #### WRITING AUC
    with open(d+'output2/AUC_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_AUC)
            
    with open(d+'output2/AUC_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_AUC)
            
    with open(d+'output2/AUC_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_AUC)
        
        
        
###################################################################################################




###################################################################################################
"""   
        fastKMMtext code below
"""
###################################################################################################


def train_fastKMMtext(beta, EPOCH, LR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test,
                      y_manual, nclasses, A, B, N_train, N_test, N_manual, trialnum, dictionary):
    #### train ################################################
    
    losses_train = []
    losses_test = []
    losses_manual = []

    classerr_train = []
    classerr_test = []
    classerr_manual = []

    print()
    print()
    
    X_train = normalize(X_train, axis=1, norm='l1')
    X_test = normalize(X_test, axis=1, norm='l1')
    X_manual = normalize(X_manual, axis=1, norm='l1')
    
    traintime_start = time.time()
    for i in range(EPOCH):
        print()
        print("fastKMMtext EPOCH: ", i)
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        l = 0
        train_loss = 0
        
        start = 0
        batchnum = 0
        while start <= N_train:
            batch = X_train.tocsr()[start:start+BATCHSIZE, :]
            y_train_batch = y_train[start:start+BATCHSIZE, :] 
            beta_batch = beta[start:start+BATCHSIZE, :] 

            B_old = B
            A_old = A
            
            # Forward Propogation
            hidden = sparse.csr_matrix.dot(A, batch.T)
            a1 = normalize(hidden, axis=0, norm='l1')
            z2 = np.dot(B, a1)
            Y_hat = stable_softmax(z2)
    
            # Back prop with alt optimization
            B = KMMgradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat, beta_batch)  
            A = KMMgradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat, beta_batch)
            
            batchnum += 1

            # NOTE figure this out, Might be missing last sample
            if start+BATCHSIZE >= N_train and start < N_train-1:   
                batch = X_train.tocsr()[start:-1, :]   # rest of train set
                y_train_batch = y_train[start:-1, :] 
                beta_batch = beta[start:-1]
                
                B_old = B
                A_old = A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(A, batch.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(B, a1)
                Y_hat = stable_softmax(z2)
        
                # Back prop with alt optimization
                B = KMMgradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat, beta_batch)  
                A = KMMgradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat, beta_batch)
                break
            else:
                start = start + BATCHSIZE

            
        # TRAINING LOSS
        train_loss = get_total_loss(A, B, X_train, y_train, N_train)
        print("KMM Train:   ", train_loss)

        ## TESTING LOSS
        test_loss = get_total_loss(A, B, X_test, y_test, N_test)
        print("KMM Test:    ", test_loss)
        
        #print("Difference = ", test_loss - train_loss)
        
        ## MANUAL SET TESTING LOSS
        manual_loss = get_total_loss(A, B, X_manual, y_manual, N_manual)
        print("KMM Manual Set:    ", manual_loss)
        print()

        losses_train.append(train_loss)
        losses_test.append(test_loss)
        losses_manual.append(manual_loss)
        
        train_class_error, train_precision, train_recall, train_F1, train_AUC, train_FPR, train_TPR = metrics(X_train, y_train, A, B, N_train, 'KMMtrain', trialnum, i)
        
        test_class_error, test_precision, test_recall, test_F1, test_AUC, test_FPR, test_TPR = metrics(X_test, y_test, A, B, N_test, 'KMMtest', trialnum, i)
        
        manual_class_error, manual_precision, manual_recall, manual_F1, manual_AUC, manual_FPR, manual_TPR = metrics(X_manual, y_manual, A, B, N_manual, 'KMMmanual', trialnum, i)
        
        
        classerr_train.append(train_class_error)
        classerr_test.append(test_class_error)
        classerr_manual.append(manual_class_error)

        print()
        print("KMMTRAIN:")
        print("         Classification Err: ", train_class_error)
        print("         Precision:          ", train_precision)
        print("         Recall:             ", train_recall)
        print("         F1:                 ", train_F1)

        print("KMMTEST:")
        print("         Classification Err: ", test_class_error)
        print("         Precision:          ", test_precision)
        print("         Recall:             ", test_recall)
        print("         F1:                 ", test_F1)
        
        print()
        print("KMMMANUAL:")
        print("         Classification Err: ", manual_class_error)
        print("         Precision:          ", manual_precision)
        print("         Recall:             ", manual_recall)
        print("         F1:                 ", manual_F1)
        
        
        write_fastKMMtext_stats(trialnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                                train_AUC, test_loss, test_class_error, test_precision, test_recall,
                                test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                                manual_recall, manual_F1, manual_AUC)
        
        fnameB = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_B_"+str(trialnum)+"epoch"+str(i)  #+".pkl"
        fnameA = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_A_"+str(trialnum)+"epoch"+str(i)
        save_model_tofile(A, B, fnameB, fnameA)
        
        i += 1
        
    traintime_end = time.time()
    
    print("KMM model took ", (traintime_end - traintime_start)/60.0, " time to train")
    

    return losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual


# finds gradient of B and returns an up
def KMMgradient_B(B, A, label, alpha, hidden, Y_hat, beta):    
    first = np.multiply(beta.T, np.subtract(Y_hat.T, label).T)
    gradient = alpha *  np.dot(first, hidden.T)
    B_new = np.subtract(B, gradient)

    return B_new


# update rule for weight matrix A
def KMMgradient_A(B, A, X, label, alpha, Y_hat, beta):
    A_old = A
    a = np.multiply(beta.T, np.subtract(Y_hat.T, label).T)
    first = np.dot(a.T, B)
    gradient = alpha * sparse.csr_matrix.dot(first.T, X)
    
    A = np.subtract(A_old, gradient) 
    
    return A        
        

def write_fastKMMtext_stats(runnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                            train_AUC, test_loss, test_class_error, test_precision, test_recall, test_F1,
                            test_AUC, manual_loss, manual_class_error, manual_precision, manual_recall,
                            manual_F1, manual_AUC):
    d = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/'
    #### WRITING LOSSES
    with open(d+'KMMoutput2/loss_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_loss)
            
    with open(d+'KMMoutput2/loss_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_loss)
            
    with open(d+'KMMoutput2/loss_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_loss)
            
    #### WRITING ERROR
    with open(d+'KMMoutput2/error_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_class_error)
    
    with open(d+'KMMoutput2/error_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_class_error)
            
    with open(d+'KMMoutput2/error_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_class_error)
            
    #### WRITING PRECISION
    with open(d+'KMMoutput2/precision_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_precision)
            
    with open(d+'KMMoutput2/precision_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_precision)
            
    with open(d+'KMMoutput2/precision_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_precision)
            
    #### WRITING RECALL
    with open(d+'KMMoutput2/recall_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_recall)
            
    with open(d+'KMMoutput2/recall_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_recall)
            
    with open(d+'KMMoutput2/recall_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_recall)
            
    #### WRITING F1
    with open(d+'KMMoutput2/F1_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_F1)
            
    with open(d+'KMMoutput2/F1_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_F1)
            
    with open(d+'KMMoutput2/F1_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_F1)
            
    #### WRITING AUC
    with open(d+'KMMoutput2/AUC_train_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % train_AUC)
            
    with open(d+'KMMoutput2/AUC_test_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % test_AUC)
            
    with open(d+'KMMoutput2/AUC_manual_'+str(runnum)+'.txt', '+a') as f:
        f.write("%s," % manual_AUC)

    
    
###################################################################################################
"""   
        main code below
"""
###################################################################################################    
    
    
def main():
    
    args = get_args()
    print(args)
    
    run = args['run']

    #if not sys.warnoptions:
    #    warnings.simplefilter("ignore")
    
    # args from Simple Queries paper
    DIM=30
    WORDGRAMS=2
    MINCOUNT=2  #2 
    MINN=3
    MAXN=3
    BUCKET=1000000

    # adjust these
    EPOCH=20
    LR= 0.007   #0.008                 #0.007            # 0.008 good for fasttext
    KMMLR = 0.018  #0.014         #0.015 pretty good

    KERN = 'lin'        # lin or rbf or poly
    NUM_RUNS = 10        # number of test runs
    SUBSET_VAL = 10000   # number of subset instances for self reported dataset
    LIN_C = 1.0          # hyperparameter for linear kernel
    
    BATCHSIZE = 100       # number of instances in each batch
    
    #model = 'kmm'
    model = 'original'   # 'kmm' for kmm implementation
    
    d = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/'
    file_names_fasttext = [d+'output2/loss_train.txt', d+'output2/loss_test.txt', d+'output2/loss_manual.txt',
                  d+'output2/error_train.txt', d+'output2/error_test.txt', d+'output2/error_manual.txt',
                  d+'output2/precision_train.txt', d+'output2/precision_test.txt', d+'output2/precision_manual.txt',
                  d+'output2/recall_train.txt', d+'output2/recall_test.txt', d+'output2/recall_manual.txt',
                  d+'output2/F1_train.txt', d+'output2/F1_test.txt', d+'output2/F1_manual.txt',
                  d+'output2/AUC_train.txt', d+'output2/AUC_test.txt', d+'output2/AUC_manual.txt']
    
    file_names_fastKMMtext = [d+'KMMoutput2/loss_train.txt', d+'KMMoutput2/loss_test.txt', d+'KMMoutput2/loss_manual.txt',
                  d+'KMMoutput2/error_train.txt', d+'KMMoutput2/error_test.txt', d+'KMMoutput2/error_manual.txt',
                  d+'KMMoutput2/precision_train.txt', d+'KMMoutput2/precision_test.txt', d+'KMMoutput2/precision_manual.txt',
                  d+'KMMoutput2/recall_train.txt', d+'KMMoutput2/recall_test.txt', d+'KMMoutput2/recall_manual.txt',
                  d+'KMMoutput2/F1_train.txt', d+'KMMoutput2/F1_test.txt', d+'KMMoutput2/F1_manual.txt',
                  d+'KMMoutput2/AUC_train.txt', d+'KMMoutput2/AUC_test.txt', d+'KMMoutput2/AUC_manual.txt']
    
    
    create_readme(DIM, WORDGRAMS, MINCOUNT, MINN, MAXN, BUCKET, EPOCH, LR, KMMLR, NUM_RUNS, SUBSET_VAL, KERN, LIN_C, BATCHSIZE)
    
    types = ['loss', 'error', 'precision', 'recall', 'F1', 'AUC']
    
    
    #while run<NUM_RUNS:
    #print("*******************************************************RUN NUMBER: ", run)
    print("*******************************************************RUN NUMBER: ", run)
    print("KMMlr = ", KMMLR, " LR = ", LR)
    print()

    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model)
    
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
    #X_manual = dictionary.get_manual_testset_full()
    #y_manual = dictionary.get_manual_set_labels_full()
    #N_manual = dictionary.get_n_manual_instances_full()
    print()
    print("Number of Manual testing instances: ", N_manual, " shape: ", X_manual.shape)
    nmanual_eachclass = dictionary.get_nlabels_eachclass_manual()
    print("N each class Manual testing instances: ", nmanual_eachclass)
    
    
    p = X_train.shape[1]
    
    # A
    #A_n = nwords + BUCKET   # cols
    A_n = p
    A_m = DIM               # rows
    uniform_val = 1.0 / DIM
    np.random.seed(0)
    A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))
    Akmm = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))  # for kmm implementation

    # B
    B_n = DIM               # cols
    B_m = nclasses          # rows
    B = np.zeros((B_m, B_n))
    Bkmm = np.zeros((B_m, B_n))   # for kmm implementation
    
    
    #beta = dictionary.get_optbeta()       # NOTE: optimal KMM reweighting coefficient
    #print(beta)
    
    with open('/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/TESTTTTTT.txt', 'a+') as f:
        f.write("BETA FINISHED")

    # NOTE: run with ones to check implementation. Should get values close to original (w/out reweithting coef)
    #beta = np.ones((N_train, 1))  
    #print("Beta Dimensions: ", beta.shape)
    
    print("#####################################")
    #*******************************************************************
    
    losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual = train_fasttext(EPOCH, LR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test, y_manual, nclasses, A, B, N_train, N_test, N_manual, run, dictionary)
    
    
    #KMMlosses_train, KMMlosses_test, KMMlosses_manual, KMMclasserr_train, KMMclasserr_test, KMMclasserr_manual = train_fastKMMtext(beta, EPOCH, KMMLR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test, y_manual, nclasses, Akmm, Bkmm, N_train, N_test, N_manual, run, dictionary)
    
    
    # writing newline to file after each trial
    #for name in file_names_fasttext:
    #    with open(name, '+a') as f:
    #        f.write('\n')
            
    #writing newline to file after each trial
    #for name in file_names_fastKMMtext:
    #    with open(name, '+a') as f:
    #        f.write('\n')
            
    #dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]
    
    #for d in dirs:
        #for filename in os.listdir(d):
            #with open(filename, 'a+') as f:
                #f.write('\n')
    
    #run += 1
    
    #########################################################
    

 
 
if __name__ == '__main__':
    main()

    
    
    
    
    
