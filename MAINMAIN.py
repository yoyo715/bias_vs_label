# MAINMAIN.py

"""
    This script will run all experiments on fastText and fastKMMText.
    
"""

from dictionary3 import Dictionary

import numpy as np
import pandas as pd
import pickle, os
from scipy import sparse
from matplotlib import pyplot as plt
import time
import sys
import warnings

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support


def create_filenames_manstates(directory, file, types):
    csv_data = pd.read_csv(file,low_memory = False)
    df = pd.DataFrame(csv_data)
    states = df['state'].unique()
    
    filenames = []
    for state in states:
        state = state[2:-1]
        for type in types:
            filenames.append(directory+state+'_'+type+'.txt')
            
    return filenames


def create_filenames_mancouns(directory, file, types):
    csv_data = pd.read_csv(file,low_memory = False)
    df = pd.DataFrame(csv_data)
    countries = df['country'].unique()
    
    filenames = []
    for c in countries:
        for type in types:
            filenames.append(directory+c+'_'+type+'.txt')
            
    return filenames


def get_man_statesets():
    directory = '/local_d/RESEARCH/bias_vs_eff/locations_manual/states/'
    
    states = []
    for filename in os.listdir(directory):
        states.append(directory+filename)
    return states

      
def get_man_countrysets():
    directory = '/local_d/RESEARCH/bias_vs_eff/locations_manual/countries/'
    
    countries = []
    for filename in os.listdir(directory):
        countries.append(directory+filename)
    return countries


def get_self_statesets():
    directory = '/local_d/RESEARCH/simple-queries/data/state_datasets/'
    
    states = []
    for filename in os.listdir(directory):
        states.append(directory+filename)
    return states


def get_self_countrysets():
    directory = '/local_d/RESEARCH/simple-queries/data/country_datasets/'
    
    countries = []
    for filename in os.listdir(directory):
        countries.append(directory+filename)
    return countries
    
    

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
    with open('output/README.md ', '+a') as f:
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
def save_model_tofile(A, B, fname):
    #data = { 'A': A,
             #'B': B,
            #}
    #output = open(fname, 'wb')
    #pickle.dump(data, output)
    #output.close()
    scipy.sparse.save_npz(fname, A)
    
    with open(fname, 'a+') as f:
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

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
    
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
    
    
    # TETON
    if test == 'train':
        fname = 'label_output/train_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'test':
        fname = 'label_output/test_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'manual':
        fname = 'label_output/manual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
        
    elif test == 'KMMtrain':
        fname = 'KMMlabel_output/kmmtrain_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'KMMtest':
        fname = 'KMMlabel_output/kmmtest_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    elif test == 'KMMmanual':
        fname = 'KMMlabel_output/kmmmanual_trial'+str(trialnum)+'_epoch'+str(epoch)+'.pkl'
    
    write_labels_tofile(fname, Y, Y_hat)

    return class_error, precision, recall, F1, roc_auc, fpr, tpr


# function to return prediction error, precision, recall, F1 score
def metrics_subset(X, Y, A, B, N):
    # get predicted classes
    #print(A.shape, X.shape)

    hidden = sparse.csr_matrix.dot(A, X.T)    
    #hidden = np.dot(A, X.T)    
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
    
    #print(true_pos, false_neg, false_pos)
    if true_pos == 0 and false_pos == 0:
        precision = 0
    else:
        precision = 1.0 * true_pos / (1.0 * true_pos + 1.0 *false_pos)           # true pos rate (TRP)
        
    if true_pos == 0 and false_neg == 0:
        recall = 0
    else:
        recall = 1.0 * true_pos / (1.0 * true_pos + 1.0 * false_neg)   # 
        
    if precision == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * ((precision * recall) / (precision + recall))

    return class_error, precision, recall, F1, roc_auc, fpr, tpr



def show_plots(EPOCH, losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual,
                    KMMlosses_train, KMMlosses_test, KMMlosses_manual, KMMclasserr_train, KMMclasserr_test, KMMclasserr_manual,
                    N_train, N_test, N_manual, trialnum):

    epochs = [l for l in range(EPOCH)]
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, losses_train, 'm', linestyle='--', label="train")
    plt.plot(epochs, losses_test, 'c', linestyle='--', label="test")
    plt.plot(epochs, losses_manual, 'g', linestyle='--', label="manual")

    plt.plot(epochs, KMMlosses_train, 'm', label="KMMtrain")
    plt.plot(epochs, KMMlosses_test, 'c', label="KMMtest")
    plt.plot(epochs, KMMlosses_manual, 'g', label="KMMmanual")


    title = "Loss: n_train: ", N_train, " n_test: ", N_test, " n_manual ", N_manual
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')


    plt.subplot(2, 1, 2)
    plt.plot(epochs, classerr_train, 'm', linestyle='--', label="train")
    plt.plot(epochs, classerr_test, 'c', linestyle='--', label="test")
    plt.plot(epochs, classerr_manual, 'g', linestyle='--', label="manual")

    plt.plot(epochs, KMMclasserr_train, 'm', label="KMMtrain")
    plt.plot(epochs, KMMclasserr_test, 'c', label="KMMtest")
    plt.plot(epochs, KMMclasserr_manual, 'g', label="KMMmanual")

    title = "Class Error: n_train: ", N_train, " n_test: ", N_test, " n_manual ", N_manual
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    
    plt.tight_layout()

    #plt.savefig('/home/mcooley/Desktop/bias_vs_labelefficiency/PLOTS/3-'+str(trialnum)+'.png')
    plt.savefig('/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/PLOTS/4-'+str(trialnum)+'.png')
    #plt.show()



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
    files_man_states = get_man_statesets()
    files_man_coun = get_man_countrysets()
    
    files_self_states = get_self_statesets()
    files_self_coun = get_self_countrysets()


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
        
        
        write_fasttext_stats(train_loss, train_class_error, train_precision, train_recall, train_F1,
                             train_AUC, test_loss, test_class_error, test_precision, test_recall,
                             test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                             manual_recall, manual_F1, manual_AUC)
        
        #fname = "./models/fasttext_trial"+str(trialnum)+"epoch"+str(i)+".pkl"
        #save_model_tofile(A, B, fname)
        
        fname = "./models/fasttext_trial"+str(trialnum)+"epoch"+str(i)  #+".pkl"
        save_model_tofile(A, B, fname)
        
        #print("testing manual countries")
        #for f in files_man_coun:
            #s = open(f, encoding='utf8').readlines()
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-4]

            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(f, " class error (fasttext manual set): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_coun_out/ft/' # manualset country
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
        
        #print("testing manual states")
        #for f in files_man_states:
            #s = open(f, encoding='utf8').readlines()
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-4]
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext manual set): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_state_out/ft/' # manualset state
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
            
        #print("testing self states")
        #for f in files_self_states:
            #s = open(f, encoding='utf8').readlines()
            #del s[0]
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-6]
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_state_out/ft/' # selfset state
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)

        #print("tesetings self countires")
        #for f in files_self_coun:
            #state = f.split("_")[-1]
            #state = state[:-6]
            
            #s = open(f, encoding='utf8').readlines()
            #del s[0]
            
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_coun_out/ft/' # selfset country
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
            
            
        
        i += 1
        
        
    traintime_end = time.time()
    
    print("model took ", (traintime_end - traintime_start)/60.0, " time to train")
    
    
    return losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual



def write_fasttext_stats(train_loss, train_class_error, train_precision, train_recall, train_F1,
                         train_AUC, test_loss, test_class_error, test_precision, test_recall,
                         test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                         manual_recall, manual_F1, manual_AUC):
    
    #### WRITING LOSSES
    with open('output2/loss_train.txt', '+a') as f:
        f.write("%s," % train_loss)
            
    with open('output2/loss_test.txt', '+a') as f:
        f.write("%s," % test_loss)
            
    with open('output2/loss_manual.txt', '+a') as f:
        f.write("%s," % manual_loss)
            
    #### WRITING ERROR
    with open('output2/error_train.txt', '+a') as f:
        f.write("%s," % train_class_error)
    
    with open('output2/error_test.txt', '+a') as f:
        f.write("%s," % test_class_error)
            
    with open('output2/error_manual.txt', '+a') as f:
        f.write("%s," % manual_class_error)
            
    #### WRITING PRECISION
    with open('output2/precision_train.txt', '+a') as f:
        f.write("%s," % train_precision)
            
    with open('output2/precision_test.txt', '+a') as f:
        f.write("%s," % test_precision)
            
    with open('output2/precision_manual.txt', '+a') as f:
        f.write("%s," % manual_precision)
            
    #### WRITING RECALL
    with open('output2/recall_train.txt', '+a') as f:
        f.write("%s," % train_recall)
            
    with open('output2/recall_test.txt', '+a') as f:
        f.write("%s," % test_recall)
            
    with open('output2/recall_manual.txt', '+a') as f:
        f.write("%s," % manual_recall)
            
    #### WRITING F1
    with open('output2/F1_train.txt', '+a') as f:
        f.write("%s," % train_F1)
            
    with open('output2/F1_test.txt', '+a') as f:
        f.write("%s," % test_F1)
            
    with open('output2/F1_manual.txt', '+a') as f:
        f.write("%s," % manual_F1)
            
    #### WRITING AUC
    with open('output2/AUC_train.txt', '+a') as f:
        f.write("%s," % train_AUC)
            
    with open('output2/AUC_test.txt', '+a') as f:
        f.write("%s," % test_AUC)
            
    with open('output2/AUC_manual.txt', '+a') as f:
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

    #dir_man_coun = '/local_d/RESEARCH/bias_vs_eff/locations_manual/countries/'
    #dir_man_state = '/local_d/RESEARCH/bias_vs_eff/locations_manual/states/'
    
    #dir_self_count = '/local_d/RESEARCH/simple-queries/data/country_datasets/'
    #dir_self_state = '/local_d/RESEARCH/simple-queries/data/state_datasets/'
    
    files_man_states = get_man_statesets()
    files_man_coun = get_man_countrysets()
    
    files_self_states = get_self_statesets()
    files_self_coun = get_self_countrysets()
    
    #######
    
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
        
        
        write_fastKMMtext_stats(train_loss, train_class_error, train_precision, train_recall, train_F1,
                                train_AUC, test_loss, test_class_error, test_precision, test_recall,
                                test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                                manual_recall, manual_F1, manual_AUC)
        
        fname = "./kmmmodels/fastKMMtext_trial"+str(trialnum)+"epoch"+str(i)  #+".pkl"
        save_model_tofile(A, B, fname)
        
        #print("Testing manual countries")
        #for f in files_man_coun:
            #s = open(f, encoding='utf8').readlines()
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-4]
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(f, " class error (fasttext manual set): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_coun_out/fkmmt/' # manualset country
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)

        #print("Testing manual states")
        #for f in files_man_states:
            #s = open(f, encoding='utf8').readlines()
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-4]
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext manual set): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_state_out/fkmmt/' # manualset state
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
            
        #print("Testing self states")
        #for f in files_self_states:
            #s = open(f, encoding='utf8').readlines()
            #del s[0]
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()
            
            #state = f.split("_")[-1]
            #state = state[:-6]
            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_state_out/fkmmt/' # selfset state
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
            
        #print("testing self countries")
        #for f in files_self_coun:
            #state = f.split("_")[-1]
            #state = state[:-6]
            
            #s = open(f, encoding='utf8').readlines()
            #del s[0]
            
            #dictionary.create_instances_and_labels_SUBSETS(s)
            #X_state = dictionary.create_statecountry_bagngrams()
            #y_state = dictionary.create_statecountry_labels()
            #N = dictionary.get_n_subset_instances()

            
            #class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics_subset(X_state, y_state, A, B, N)
            ##print(state, " class error (fasttext): ", class_error)
            
            #dirft = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_coun_out/fkmmt/' # selfset country
            
            #loss = get_total_loss(A, B, X_state, y_state, N)
            #write_stats(dirft, state, loss, class_error, precision, recall, F1, roc_auc)
        
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
        

def write_fastKMMtext_stats(train_loss, train_class_error, train_precision, train_recall, train_F1,
                            train_AUC, test_loss, test_class_error, test_precision, test_recall, test_F1,
                            test_AUC, manual_loss, manual_class_error, manual_precision, manual_recall,
                            manual_F1, manual_AUC):
    
    #### WRITING LOSSES
    with open('KMMoutput2/loss_train.txt', '+a') as f:
        f.write("%s," % train_loss)
            
    with open('KMMoutput2/loss_test.txt', '+a') as f:
        f.write("%s," % test_loss)
            
    with open('KMMoutput2/loss_manual.txt', '+a') as f:
        f.write("%s," % manual_loss)
            
    #### WRITING ERROR
    with open('KMMoutput2/error_train.txt', '+a') as f:
        f.write("%s," % train_class_error)
    
    with open('KMMoutput2/error_test.txt', '+a') as f:
        f.write("%s," % test_class_error)
            
    with open('KMMoutput2/error_manual.txt', '+a') as f:
        f.write("%s," % manual_class_error)
            
    #### WRITING PRECISION
    with open('KMMoutput2/precision_train.txt', '+a') as f:
        f.write("%s," % train_precision)
            
    with open('KMMoutput2/precision_test.txt', '+a') as f:
        f.write("%s," % test_precision)
            
    with open('KMMoutput2/precision_manual.txt', '+a') as f:
        f.write("%s," % manual_precision)
            
    #### WRITING RECALL
    with open('KMMoutput2/recall_train.txt', '+a') as f:
        f.write("%s," % train_recall)
            
    with open('KMMoutput2/recall_test.txt', '+a') as f:
        f.write("%s," % test_recall)
            
    with open('KMMoutput2/recall_manual.txt', '+a') as f:
        f.write("%s," % manual_recall)
            
    #### WRITING F1
    with open('KMMoutput2/F1_train.txt', '+a') as f:
        f.write("%s," % train_F1)
            
    with open('KMMoutput2/F1_test.txt', '+a') as f:
        f.write("%s," % test_F1)
            
    with open('KMMoutput2/F1_manual.txt', '+a') as f:
        f.write("%s," % manual_F1)
            
    #### WRITING AUC
    with open('KMMoutput2/AUC_train.txt', '+a') as f:
        f.write("%s," % train_AUC)
            
    with open('KMMoutput2/AUC_test.txt', '+a') as f:
        f.write("%s," % test_AUC)
            
    with open('KMMoutput2/AUC_manual.txt', '+a') as f:
        f.write("%s," % manual_AUC)

    
    
###################################################################################################
"""   
        main code below
"""
###################################################################################################    
    
    
def main():

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    # args from Simple Queries paper
    DIM=30
    WORDGRAMS=2
    MINCOUNT=2  #2 
    MINN=3
    MAXN=3
    BUCKET=1000000

    # adjust these
    EPOCH=20
    LR= 0.008                 #0.007            # 0.008 good for fasttext
    KMMLR = 0.014         #0.015 pretty good

    KERN = 'lin'        # lin or rbf or poly
    NUM_RUNS = 10        # number of test runs
    SUBSET_VAL = 100   # number of subset instances for self reported dataset
    LIN_C = 0.9          # hyperparameter for linear kernel
    
    BATCHSIZE = 100       # number of instances in each batch
    
    model = 'kmm'
    #model = 'original'   # 'kmm' for kmm implementation
    
    file_names_fasttext = ['output2/loss_train.txt', 'output2/loss_test.txt', 'output2/loss_manual.txt',
                  'output2/error_train.txt', 'output2/error_test.txt', 'output2/error_manual.txt',
                  'output2/precision_train.txt', 'output2/precision_test.txt', 'output2/precision_manual.txt',
                  'output2/recall_train.txt', 'output2/recall_test.txt', 'output2/recall_manual.txt',
                  'output2/F1_train.txt', 'output2/F1_test.txt', 'output2/F1_manual.txt',
                  'output2/AUC_train.txt', 'output2/AUC_test.txt', 'output2/AUC_manual.txt']
    
    file_names_fastKMMtext = ['KMMoutput2/loss_train.txt', 'KMMoutput2/loss_test.txt', 'KMMoutput2/loss_manual.txt',
                  'KMMoutput2/error_train.txt', 'KMMoutput2/error_test.txt', 'KMMoutput2/error_manual.txt',
                  'KMMoutput2/precision_train.txt', 'KMMoutput2/precision_test.txt', 'KMMoutput2/precision_manual.txt',
                  'KMMoutput2/recall_train.txt', 'KMMoutput2/recall_test.txt', 'KMMoutput2/recall_manual.txt',
                  'KMMoutput2/F1_train.txt', 'KMMoutput2/F1_test.txt', 'KMMoutput2/F1_manual.txt',
                  'KMMoutput2/AUC_train.txt', 'KMMoutput2/AUC_test.txt', 'KMMoutput2/AUC_manual.txt']
    
    
    create_readme(DIM, WORDGRAMS, MINCOUNT, MINN, MAXN, BUCKET, EPOCH, LR, KMMLR, NUM_RUNS, SUBSET_VAL, KERN, LIN_C, BATCHSIZE)
    
    types = ['loss', 'error', 'precision', 'recall', 'F1', 'AUC']
    
    
    ##### manual set filenames
    #f1 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/location/Manual_Country_data.csv'
    #dir1 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_coun_out/ft/'      # manualset country
    #dir2 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_coun_out/fkmmt/'   # manualset country

    #man_coun_ft_filenames = create_filenames_mancouns(dir1, f1, types)
    #man_coun_fkmmt_filenames = create_filenames_mancouns(dir2, f1, types)
    
    #f2 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/location/Manual_State.csv'
    #dir3 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_state_out/ft/'     # manualset state
    #dir4 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/manual_state_out/fkmmt/'  # manualset state

    #man_state_ft_filenames = create_filenames_manstates(dir3, f2, types)
    #man_state_fkmmt_filenames = create_filenames_manstates(dir4, f2, types)
    
    
    ###### self labeled set filenames
    #f3 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/location/Self_Country_data.csv'
    #dir5 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_coun_out/ft/'        # selfset country
    #dir6 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_coun_out/fkmmt/'     # selfset country

    #self_coun_ft_filenames = create_filenames_mancouns(dir5, f3, types)
    #self_coun_fkmmt_filenames = create_filenames_mancouns(dir6, f3, types)
    
    #f4 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/location/Self_State_data.csv'
    #dir7 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_state_out/ft/'        # selfset state
    #dir8 = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/self_state_out/fkmmt/'     # selfset state

    #self_state_ft_filenames = create_filenames_manstates(dir7, f4, types)
    #self_state_fkmmt_filenames = create_filenames_manstates(dir8, f4, types)
    
    
    #########################################################
    
    run = 0
    
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
    
    
    beta = dictionary.get_optbeta()       # NOTE: optimal KMM reweighting coefficient
    #print(beta)
    

    # NOTE: run with ones to check implementation. Should get values close to original (w/out reweithting coef)
    #beta = np.ones((N_train, 1))  
    print("Beta Dimensions: ", beta.shape)
    
    print("#####################################")
    #*******************************************************************
    
    #man_states = get_man_statesets()
    #man_countries = get_man_countrysets()
    
    #self_states = get_self_statesets()
    #self_countries = get_self_countrysets()
    
    losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual = train_fasttext(EPOCH, LR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test, y_manual, nclasses, A, B, N_train, N_test, N_manual, run, dictionary)
    
    
    KMMlosses_train, KMMlosses_test, KMMlosses_manual, KMMclasserr_train, KMMclasserr_test, KMMclasserr_manual = train_fastKMMtext(beta, EPOCH, KMMLR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test, y_manual, nclasses, Akmm, Bkmm, N_train, N_test, N_manual, run, dictionary)
    
    
    # writing newline to file after each trial
    for name in file_names_fasttext:
        with open(name, '+a') as f:
            f.write('\n')
            
    #writing newline to file after each trial
    for name in file_names_fastKMMtext:
        with open(name, '+a') as f:
            f.write('\n')
            
    #dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]
    
    #for d in dirs:
        #for filename in os.listdir(d):
            #with open(filename, 'a+') as f:
                #f.write('\n')
    
    #show_plots(EPOCH, losses_train, losses_test, losses_manual, classerr_train, classerr_test, classerr_manual,
            #KMMlosses_train, KMMlosses_test, KMMlosses_manual, KMMclasserr_train, KMMclasserr_test, KMMclasserr_manual,
            #N_train, N_test, N_manual, run)
    
    #run += 1
    
    #########################################################
    

 
 
if __name__ == '__main__':
    main()

    
    
    
    
    
