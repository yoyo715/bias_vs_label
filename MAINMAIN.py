# MAINMAIN.py

"""
    This script will run all experiments on fastText and fastKMMText.
    
"""

from dictionary3 import Dictionary

import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize


# model_version: 'original' or 'kmm;
def create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, model_version):
    
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    # initialize training
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, model=model_version)
    end = time.time()
    print("dictionary took ", (end - start)/60.0, " time to create.")
    
    return dictionary


# writing model specifications to an about file
def create_readme(DIM, WORDGRAMS, MINCOUNT, MINN, MAXN, BUCKET, EPOCH, LR, NUM_RUNS, SUBSET_VAL):
    with open('output/README.md ', '+a') as f:
        f.write('# Original Model specifications # \n\n')
        f.write('DIM: ', DIM)
        f.write('\n\n')
        f.write('WORDGRAMS: ', WORDGRAMS)
        f.write('\n\n')
        f.write('MINCOUNT: ', MINCOUNT)
        f.write('\n\n')
        f.write('MINN: ', MINN)
        f.write('\n\n')
        f.write('MAXN: ', MAXN)
        f.write('\n\n')
        f.write('BUCKET: ', BUCKET)
        f.write('\n\n')
        f.write('\n\n')
        f.write('\n\n')
        
        # Hyperparameters
        f.write('## Hyperparameters ##\n\n')
        f.write('EPOCH: ', LR)
        f.write('\n\n')
        f.write('LR: ', LR)
        f.write('\n\n')
        f.write('NUM_RUNS: ', NUM_RUNS)
        f.write('\n\n')
        f.write('SUBSET_VAL: ', SUBSET_VAL)
        

# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nclasses, alpha, hidden, Y_hat):    
    gradient = alpha * np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
    B_new = np.subtract(B, gradient)

    return B_new


# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, Y_hat):
    A_old = A
    first = np.dot(np.subtract(Y_hat.T, label), B)
    
    # WARNING check this
    if np.sum(x) > 0:
        sec = x * (1.0/np.sum(x))
    else:
        sec = x

    gradient = alpha * sparse.csr_matrix.dot(first.T, sec)
    A = np.subtract(A_old, gradient) 
    
    return A


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

        
        
def train_fasttext(EPOCH, LR, BATCHSIZE, X_train, y_train, nclasses, A, B,  ):
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
                
            sum_ = np.sum(batch, axis = 1)
            sum_[sum_ == 0] = 1         # replace zeros with ones so divide will work
            sum_ = np.array(sum_).flatten()
            
            a1 = (hidden.T / sum_[:,None]).T
            z2 = np.dot(B, a1)
            Y_hat = stable_softmax(z2)
    
            # Back prop with alt optimization
            B = gradient_B(B_old, A_old, batch, y_train_batch, nclasses, alpha, a1, Y_hat)  
            A = gradient_A(B_old, A_old, batch, y_train_batch, nclasses, alpha, Y_hat)

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
        
        
def train_fastKMMtext():
    
    
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
    SUBSET_VAL = 500   # number of subset instances for self reported dataset
    LIN_C = 0.90        # hyperparameter for linear kernel
    
    BATCHSIZE = 1       # number of instances in each batch
    
    file_names_original = ['output/loss_train.txt', 'output/loss_test.txt', 'output/loss_manual.txt',
                  'output/error_train.txt', 'output/error_test.txt', 'output/error_manual.txt',
                  'output/precision_train.txt', 'output/precision_test.txt', 'output/precision_manual.txt',
                  'output/recall_train.txt', 'output/recall_test.txt', 'output/recall_manual.txt',
                  'output/F1_train.txt', 'output/F1_test.txt', 'output/F1_manual.txt',
                  'output/AUC_train.txt', 'output/AUC_test.txt', 'output/AUC_manual.txt']
    
    file_names_kmm = ['outputkmm/kmmloss_train.txt', 'outputkmm/kmmloss_test.txt', 'outputkmm/kmmloss_manual.txt',
                  'outputkmm/kmmerror_train.txt', 'outputkmm/kmmerror_test.txt', 'outputkmm/kmmerror_manual.txt',
                  'outputkmm/kmmprecision_train.txt', 'outputkmm/kmmprecision_test.txt', 'outputkmm/kmmprecision_manual.txt',
                  'outputkmm/kmmrecall_train.txt', 'outputkmm/kmmrecall_test.txt', 'outputkmm/kmmrecall_manual.txt',
                  'outputkmm/kmmF1_train.txt', 'outputkmm/kmmF1_test.txt', 'outputkmm/kmmF1_manual.txt',
                  'outputkmm/kmmAUC_train.txt', 'outputkmm/kmmAUC_test.txt', 'outputkmm/kmmAUC_manual.txt']
    
    
    create_readme(DIM, WORDGRAMS, MINCOUNT, MINN, MAXN, BUCKET, EPOCH, LR, NUM_RUNS, SUBSET_VAL)
    
    #########################################################
    
    for run in range(NUM_RUNS):
        print("*******************************************************RUN NUMBER: ", run)
        print()
    
        dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, 'original')
        
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
        
        #*******************************************************************
        
        train_fasttext(EPOCH, LR, BATCHSIZE, X_train, y_train, nclasses, A, B,  )
        train_fastKMMtext()
        
        
        
        run += 1
    
    #########################################################
    
    
    
    
    
    
    
 
 
if __name__ == '__main__':
    main()

    
    
    
    
    
