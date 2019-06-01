#CLASS_fasttext.py

"""
    This is the simple fasttext model without kmm applied.
"""

import numpy as np
from scipy import sparse, stats
from sklearn.preprocessing import normalize
from cvxopt import matrix, solvers
import time, math, sys, pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import sklearn.metrics.pairwise as sk


class FastText:
    def __init__(self, dictionary, learning_rate, DIM, EPOCH, batchsize):
        print()
        print("######################## FastText ########################")
        
        #self.save_dir = '/project/lsrtwitter/mcooley3/APRIL_2019_exps/fasttext/'
        self.save_dir = '/project/lsrtwitter/mcooley3/RACE_JUNE_2019_exps/fasttext/'
        
        self.LR = learning_rate
        self.EPOCH = EPOCH
        self.BATCHSIZE = batchsize
        self.run_number = dictionary.run_number  

        nwords = dictionary.nwords
        nclasses = dictionary.nclasses
        
        print("TRIAL: ", self.run_number)
        
        # Initialize Self-labeled Training Sets
        self.X_STRAIN = dictionary.X_STRAIN
        self.X_SVAL = dictionary.X_SVAL
        self.Y_STRAIN = dictionary.Y_STRAIN
        self.Y_SVAL = dictionary.Y_SVAL
        
        self.N_strain = dictionary.n_strain
        self.N_sval = dictionary.n_sval
        
        print("X_STRAIN.shape: ", self.X_STRAIN.shape, " X_SVAL.shape: ", self.X_SVAL.shape)
        print("Y_STRAIN.shape: ", self.Y_STRAIN.shape, " Y_SVAL.shape: ", self.Y_SVAL.shape)
        print("Number of STrain instances: ", self.N_strain, " Number of SVal instances: ", self.N_sval)
        print()
        
        # Initialize Random Sets (Kaggle dataset)
        self.X_RTEST = dictionary.X_RTEST
        self.X_RVAL = dictionary.X_RVAL
        self.Y_RTEST = dictionary.Y_RTEST
        self.Y_RVAL = dictionary.Y_RVAL
        
        self.N_rtest = dictionary.n_rtest
        self.N_rval = dictionary.n_rval
        
        print("X_RTEST.shape: ", self.X_RTEST.shape, " X_RVAL.shape: ", self.X_RVAL.shape)
        print("Y_RTEST.shape: ", self.Y_RTEST.shape, " Y_RVAL.shape: ", self.Y_RVAL.shape)
        print()
        
        # Initialize Self-Labeled Testing Set
        self.X_STEST = dictionary.X_STEST
        self.Y_STEST = dictionary.Y_STEST
        
        self.N_stest = dictionary.n_stest
        print("X_STEST.shape: ", self.X_STEST.shape)
        print("Y_STEST.shape: ", self.Y_STEST.shape)
        print()
        
        
        # A
        p = self.X_STRAIN.shape[1]    # cols
        A_n = p
        A_m = DIM                    # rows
        uniform_val = 1.0 / DIM
        np.random.seed(0)
        self.A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))

        # B
        B_n = DIM               # cols
        B_m = nclasses          # rows
        self.B = np.zeros((B_m, B_n))

        sys.stdout.flush()
        
    
    def stable_softmax(self, X): 
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
    def get_total_loss(self, A, B, X, y, N):
        hidden = sparse.csr_matrix.dot(A, X.T)      
        
        a1 = normalize(hidden, axis=0, norm='l1')
        z2 = np.dot(B, a1)
        
        Y_hat = self.stable_softmax(z2)
        loglike = np.log(Y_hat)
        
        loss = -np.multiply(y, loglike.T)  # need to multiply element wise here
        loss = np.sum(loss)/N
        
        return loss
        
        
    # finds gradient of B and returns an up
    def gradient_B(self, B, A, label, alpha, hidden, Y_hat):    
        gradient = alpha * np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
        B_new = np.subtract(B, gradient)

        return B_new


    # update rule for weight matrix A
    def gradient_A(self, B, A, X, label, alpha, Y_hat):
        A_old = A
        first = np.dot(np.subtract(Y_hat.T, label), B)
        gradient = alpha * sparse.csr_matrix.dot(first.T, X)
        
        A = np.subtract(A_old, gradient) 
        
        return A    
    
    def get_class_err(self, Y_hat, Y, N):        
        # compare to actual classes
        prediction_max = np.argmax(Y_hat, axis=0)
        true_label_max = np.argmax(Y, axis=1)
        
        class_error = np.sum(true_label_max != prediction_max.T) * 1.0 / N
        
        print(confusion_matrix(true_label_max, prediction_max))
        print()
        
        return class_error
    
    def compute_yhat(self, A, B, X):
        hidden = sparse.csr_matrix.dot(A, X.T)
        a1 = normalize(hidden, axis=0, norm='l1')
        z2 = np.dot(B, a1)
        Y_hat = self.stable_softmax(z2)
        return Y_hat
    
    
    def save_yhat_y(self, epoch, Y_STRAIN, Y_SVAL, Y_RTEST, Y_RVAL, Y_STEST,
                                   yhat_strain, yhat_sval, yhat_rtest, yhat_rval, yhat_stest):
        
        fname = 'fasttext_RUN'+self.run_number+'_EPOCH'+str(epoch)+'.pkl'
        
        data =  {  'Y_STRAIN': Y_STRAIN,
                    'Y_SVAL': Y_SVAL,
                    'Y_RTEST': Y_RTEST,
                    'Y_RVAL': Y_RVAL,
                    'Y_STEST': Y_STEST,
                    'yhat_strain': yhat_strain,
                    'yhat_sval': yhat_sval,
                    'yhat_rtest': yhat_rtest,
                    'yhat_rval': yhat_rval,
                    'yhat_stest': yhat_stest
                }
        
        output = open(self.save_dir+fname, 'wb')
        pickle.dump(data, output)
        output.close()
    
    
    def train_batch(self):
        print()
        print()
        print("Batch Training, BATCHSIZE:", self.BATCHSIZE)

        X_strain = normalize(self.X_STRAIN, axis=1, norm='l1')
        X_sval = normalize(self.X_SVAL, axis=1, norm='l1')
        X_rtest = normalize(self.X_RTEST, axis=1, norm='l1')
        X_rval = normalize(self.X_RVAL, axis=1, norm='l1')
        X_stest = normalize(self.X_STEST, axis=1, norm='l1')
        
        strain_loss = self.get_total_loss(self.A, self.B, X_strain, self.Y_STRAIN, self.N_strain)
        sval_loss = self.get_total_loss(self.A, self.B, X_sval, self.Y_SVAL, self.N_sval)
        rtest_loss = self.get_total_loss(self.A, self.B, X_rtest, self.Y_RTEST, self.N_rtest)
        rval_loss = self.get_total_loss(self.A, self.B, X_rval, self.Y_RVAL, self.N_rval)
        stest_loss = self.get_total_loss(self.A, self.B, X_stest, self.Y_STEST, self.N_stest)
        
        print("INITIAL STrain Loss:   ", strain_loss)
        print("INITIAL SVal Loss:   ", sval_loss)
        print("INITIAL RTest Loss:    ", rtest_loss)
        print("INITIAL RVal Loss:    ", rval_loss)
        print("INITIAL STest Loss:    ", stest_loss)
        print()
        
        yhat_strain = self.compute_yhat(self.A, self.B, X_strain)
        strain_class_error = self.get_class_err(yhat_strain, self.Y_STRAIN, self.N_strain)
        
        yhat_sval = self.compute_yhat(self.A, self.B, X_sval)
        sval_class_error = self.get_class_err(yhat_sval, self.Y_SVAL, self.N_sval)
        
        yhat_rtest = self.compute_yhat(self.A, self.B, X_rtest)
        rtest_class_error = self.get_class_err(yhat_rtest, self.Y_RTEST, self.N_rtest)
        
        yhat_rval = self.compute_yhat(self.A, self.B, X_rval)
        rval_class_error = self.get_class_err(yhat_rval, self.Y_RVAL, self.N_rval)
        
        yhat_stest = self.compute_yhat(self.A, self.B, X_stest)
        stest_class_error = self.get_class_err(yhat_stest, self.Y_STEST, self.N_stest)

        print("INITIAL STRAIN Classification Err: ", strain_class_error)
        print("INITIAL SVAL Classification Err: ", sval_class_error)
        print("INITIAL RTEST Classification Err: ", rtest_class_error)
        print("INITIAL RVAL Classification Err: ", rval_class_error)
        print("INITIAL STEST Classification Err: ", stest_class_error)
        print()
            
        print("_____________________________________________________")
        
        traintime_start = time.time()
        for i in range(self.EPOCH):
            print()
            print("FastText EPOCH: ", i)
            
            epoch_st = time.time()

            # linearly decaying lr alpha
            alpha = self.LR * ( 1 - i / self.EPOCH)
            
            # Shuffle data
            batch_indices = np.random.permutation(self.N_strain)
            X_strain_batch = X_strain.tocsr()[batch_indices]
            y_train_batch = self.Y_STRAIN[batch_indices]

            for j in range(0, self.N_strain, self.BATCHSIZE):                
                batch = X_strain_batch[j:j+self.BATCHSIZE]
                y_batch = y_train_batch[j:j+self.BATCHSIZE]

                B_old = self.B
                A_old = self.A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(self.A, batch.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(self.B, a1)
                Y_hat = self.stable_softmax(z2)
        
                # Back prop with alt optimization
                self.B = self.gradient_B(B_old, A_old, y_batch, alpha, a1, Y_hat)  
                self.A = self.gradient_A(B_old, A_old, batch, y_batch, alpha, Y_hat)
                
            epoch_et = time.time()
            
            strain_loss = self.get_total_loss(self.A, self.B, X_strain, self.Y_STRAIN, self.N_strain)
            sval_loss = self.get_total_loss(self.A, self.B, X_sval, self.Y_SVAL, self.N_sval)
            rtest_loss = self.get_total_loss(self.A, self.B, X_rtest, self.Y_RTEST, self.N_rtest)
            rval_loss = self.get_total_loss(self.A, self.B, X_rval, self.Y_RVAL, self.N_rval)
            stest_loss = self.get_total_loss(self.A, self.B, X_stest, self.Y_STEST, self.N_stest)
            
            print("STrain Loss:   ", strain_loss)
            print("SVal Loss:   ", sval_loss)
            print("RTest Loss:    ", rtest_loss)
            print("RVal Loss:    ", rval_loss)
            print("STest Loss:    ", stest_loss)
            print()
            
            yhat_strain = self.compute_yhat(self.A, self.B, X_strain)
            strain_class_error = self.get_class_err(yhat_strain, self.Y_STRAIN, self.N_strain)
            
            yhat_sval = self.compute_yhat(self.A, self.B, X_sval)
            sval_class_error = self.get_class_err(yhat_sval, self.Y_SVAL, self.N_sval)
            
            yhat_rtest = self.compute_yhat(self.A, self.B, X_rtest)
            rtest_class_error = self.get_class_err(yhat_rtest, self.Y_RTEST, self.N_rtest)
            
            yhat_rval = self.compute_yhat(self.A, self.B, X_rval)
            rval_class_error = self.get_class_err(yhat_rval, self.Y_RVAL, self.N_rval)
            
            yhat_stest = self.compute_yhat(self.A, self.B, X_stest)
            stest_class_error = self.get_class_err(yhat_stest, self.Y_STEST, self.N_stest)

            print("STRAIN Classification Err: ", strain_class_error)
            print("SVAL Classification Err: ", sval_class_error)
            print("RTEST Classification Err: ", rtest_class_error)
            print("RVAL Classification Err: ", rval_class_error)
            print("STEST Classification Err: ", stest_class_error)
            print()
            
            #self.save_yhat_y(i, self.Y_STRAIN, self.Y_SVAL, self.Y_RTEST, self.Y_RVAL, self.Y_STEST,
                                   #yhat_strain, yhat_sval, yhat_rtest, yhat_rval, yhat_stest)
            
    
            print("~~~~Epoch took ", (epoch_et - epoch_st)/60.0, " minutes")            
            print()
            print("_____________________________________________________")
            sys.stdout.flush()
            
            i += 1
            
        traintime_end = time.time()
        print("FastText model took ", (traintime_end - traintime_start)/60.0, " minutes to train")
        

    def train(self):
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
            for x in X_train:       
                label = y_train[l]
                B_old = B
                A_old = A
                    
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(A, x.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(B, a1)
                Y_hat = stable_softmax(z2)
        
                # Back prop with alt optimization
                B = gradient_B(B_old, A_old, label, alpha, a1, Y_hat)  
                A = gradient_A(B_old, A_old, x, label, alpha, Y_hat)

                    
                # TRAINING LOSS
                train_loss = get_total_loss(A, B, X_train, y_train, N_train)
                print("Train:   ", train_loss)

                # TESTING LOSS
                test_loss = get_total_loss(A, B, X_test, y_test, N_test)
                print("Test:    ", test_loss)
                
                # MANUAL SET TESTING LOSS
                manual_loss = get_total_loss(A, B, X_manual, y_manual, N_manual)
                print("Manual Set:    ", manual_loss)
                print()

                losses_train.append(train_loss)
                losses_test.append(test_loss)
                losses_manual.append(manual_loss)
                
                l += 1
                
            
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
            
            
            i += 1
            
        traintime_end = time.time()
        print("FastText model took ", (traintime_end - traintime_start)/60.0, " time to train")
        
    
