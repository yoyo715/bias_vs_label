# CLASS_wfasttext.py

"""
    wFastText model class

"""

import numpy as np
from scipy import sparse, stats
from sklearn.preprocessing import normalize
from cvxopt import matrix, solvers
import time, math, sys

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import sklearn.metrics.pairwise as sk



class wFastText:
    def __init__(self, dictionary, learning_rate, DIM, EPOCH, kmmB, batchsize, kernel):
        self.LR = learning_rate
        self.EPOCH = EPOCH
        self.kmmB = kmmB
        self.BATCHSIZE = batchsize
        self.kernel = kernel
        
        nwords = dictionary.get_nwords()
        nclasses = dictionary.get_nclasses()
        
        print("TRIAL: ", dictionary.run_number)
        
        #initialize testing
        self.X_train, self.X_test, self.y_train, self.y_test = dictionary.get_train_and_test()
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        self.N_train = dictionary.get_n_train_instances()
        self.N_test = dictionary.get_n_test_instances()
        
        print("Number of Train instances: ", self.N_train, " Number of Test instances: ", self.N_test)
        ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
        ntest_eachclass = dictionary.get_nlabels_eachclass_test()
        print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)
        
        
        # manual labeled set (Kaggle dataset)
        self.X_manual = dictionary.get_manual_testset()
        self.y_manual = dictionary.get_manual_set_labels()
        self.N_manual = dictionary.get_n_manual_instances()
        
        print()
        print("Number of Manual testing instances: ", self.N_manual, " shape: ", self.X_manual.shape)
        nmanual_eachclass = dictionary.get_nlabels_eachclass_manual()
        print("N each class Manual testing instances: ", nmanual_eachclass)
        
        
        # A
        p = self.X_train.shape[1]    # cols
        A_n = p
        A_m = DIM               # rows
        uniform_val = 1.0 / DIM
        np.random.seed(0)
        self.A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))

        # B
        B_n = DIM               # cols
        B_m = nclasses          # rows
        self.B = np.zeros((B_m, B_n))

        sys.stdout.flush()
        self.betas = self.create_optbeta()       # NOTE: optimal KMM reweighting coefficient
        sys.stdout.flush()
        
    
    def create_optbeta(self):
        print("starting beta optimization..............................")
        
        start = time.time()
        
        opt_beta = self.kernel_mean_matching(self.X_manual, self.X_train, kern=self.kernel, B=self.kmmB, eps=None)
        
        end = time.time()
        print("Beta took ", (end - start)/60.0, " minutes to optimize.")
        print("About Beta: ")
        print(stats.describe(opt_beta))
        
        return opt_beta

    
    # Z is training data, X is testing data
    def kernel_mean_matching(self, X, Z, kern='lin', B=1.0, eps=None):
        nx = X.shape[0]
        nz = Z.shape[0]
        
        print("nx: ", nx, " nz: ", nz)
        
        if eps == None:
            eps = B/math.sqrt(nz)
            
        if kern == 'lin':
            K = np.dot(Z, Z.T) 
            K = K.todense() #+ self.lin_c  
            kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
            
            #K2 = sk.linear_kernel(Z.T, Z.T)             ##WARNING double check this
            #kappa2 = np.sum(sk.linear_kernel(Z, X), axis=1)*float(nz)/float(nx)
        elif kern == 'rbf':
            K=sk.rbf_kernel(Z, Z)
            kappa = np.sum(sk.rbf_kernel(Z, X), axis=1)*float(nz)/float(nx)
        elif kern == 'poly':
            K=sk.polynomial_kernel(Z, Z)
            kappa = np.sum(sk.polynomial_kernel(Z, X), axis=1)*float(nz)/float(nx)
        elif kern == 'laplacian':
            K=sk.laplacian_kernel(Z, Z)
            kappa = np.sum(sk.laplacian_kernel(Z, X), axis=1)*float(nz)/float(nx)
        elif kern == 'sigmoid':
            K=sk.sigmoid_kernel(Z, Z)
            kappa = np.sum(sk.sigmoid_kernel(Z, X), axis=1)*float(nz)/float(nx)
            
        else:
            raise ValueError('unknown kernel')
        
        
        K = K.astype(np.double)
        K = matrix(K)        
        kappa = matrix(kappa)

        G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
        h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
        
        print("starting solver")
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, -kappa, G, h)
        print(sol)
        coef = np.array(sol['x'])
        return coef

        
        
###########################################################################################################

      
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
    def KMMgradient_B(self, B, A, label, alpha, hidden, Y_hat, beta):    
        first = np.multiply(beta.T, np.subtract(Y_hat.T, label).T)
        gradient = alpha *  np.dot(first, hidden.T)
        B_new = np.subtract(B, gradient)

        return B_new


    # update rule for weight matrix A
    def KMMgradient_A(self, B, A, X, label, alpha, Y_hat, beta):
        A_old = A
        a = np.multiply(beta.T, np.subtract(Y_hat.T, label).T)
        first = np.dot(a.T, B)
        gradient = alpha * sparse.csr_matrix.dot(first.T, X)
        
        A = np.subtract(A_old, gradient) 
        
        return A       
    
    
    # function to return prediction error, precision, recall, F1 score
    def metrics(self, X, Y, A, B, N, test, epoch):
        # get predicted classes
        hidden = sparse.csr_matrix.dot(A, X.T)        
        a1 = normalize(hidden, axis=0, norm='l1')
        z2 = np.dot(B, a1)
        Y_hat = self.stable_softmax(z2)

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


        return class_error #, precision, recall, F1, roc_auc, fpr, tpr
        
        
    def train_batch(self):
        losses_train = []
        losses_test = []
        losses_manual = []

        classerr_train = []
        classerr_test = []
        classerr_manual = []

        print()
        print()
        
        X_train = normalize(self.X_train, axis=1, norm='l1')
        X_test = normalize(self.X_test, axis=1, norm='l1')
        X_manual = normalize(self.X_manual, axis=1, norm='l1')
       
        print("Shuffling dataset")
        sys.stdout.flush()
        np.random.shuffle(X_train)
 
        traintime_start = time.time()
        for i in range(self.EPOCH):
            print()
            print("wFastText EPOCH: ", i)
            
            # linearly decaying lr alpha
            alpha = self.LR * ( 1 - i / self.EPOCH)
            
            l = 0
            train_loss = 0
            
            start = 0
            batchnum = 0
            while start <= self.N_train:
                print(start)
                
                batch = X_train.tocsr()[start:start+self.BATCHSIZE, :]
                y_train_batch = self.y_train[start:start+self.BATCHSIZE, :] 
                beta_batch = self.betas[start:start+self.BATCHSIZE, :] 

                B_old = self.B
                A_old = self.A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(self.A, batch.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(self.B, a1)
                Y_hat = self.stable_softmax(z2)
        
                # Back prop with alt optimization
                self.B = self.KMMgradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat, beta_batch)  
                self.A = self.KMMgradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat, beta_batch)
                
                batchnum += 1

                # NOTE figure this out, Might be missing last sample
                if start+self.BATCHSIZE >= self.N_train and start < self.N_train-1:   
                    batch = X_train.tocsr()[start:-1, :]   # rest of train set
                    y_train_batch = self.y_train[start:-1, :] 
                    beta_batch = self.betas[start:-1]
                    
                    B_old = self.B
                    A_old = self.A
                    
                    # Forward Propogation
                    hidden = sparse.csr_matrix.dot(self.A, batch.T)
                    a1 = normalize(hidden, axis=0, norm='l1')
                    z2 = np.dot(self.B, a1)
                    Y_hat = self.stable_softmax(z2)
            
                    # Back prop with alt optimization
                    self.B = self.KMMgradient_B(B_old, A_old, y_train_batch, alpha, a1, Y_hat, beta_batch)  
                    self.A = self.KMMgradient_A(B_old, A_old, batch, y_train_batch, alpha, Y_hat, beta_batch)
                    break
                else:
                    start = start + self.BATCHSIZE

                
            # TRAINING LOSS
            train_loss = self.get_total_loss(self.A, self.B, X_train, self.y_train, self.N_train)
            print("KMM Train Loss:   ", train_loss)

            ## TESTING LOSS
            test_loss = self.get_total_loss(self.A, self.B, X_test, self.y_test, self.N_test)
            print("KMM Test Loss:    ", test_loss)
            
            ## MANUAL SET TESTING LOSS
            manual_loss = self.get_total_loss(self.A, self.B, X_manual, self.y_manual, self.N_manual)
            print("KMM Manual Set Loss:    ", manual_loss)
            print()

            losses_train.append(train_loss)
            losses_test.append(test_loss)
            losses_manual.append(manual_loss)
            
            train_class_error = self.metrics(X_train, self.y_train, self.A, self.B, self.N_train, 'KMMtrain', i)
            
            test_class_error = self.metrics(X_test, self.y_test, self.A, self.B, self.N_test, 'KMMtest', i)
            
            manual_class_error = self.metrics(X_manual, self.y_manual, self.A, self.B, self.N_manual, 'KMMmanual', i)
            
            
            classerr_train.append(train_class_error)
            classerr_test.append(test_class_error)
            classerr_manual.append(manual_class_error)

            print()
            print("KMMTRAIN Classification Err: ", train_class_error)
            #print("         Precision:          ", train_precision)
            #print("         Recall:             ", train_recall)
            #print("         F1:                 ", train_F1)

            print("KMMTEST Classification Err:", test_class_error)
            #print("         Precision:          ", test_precision)
            #print("         Recall:             ", test_recall)
            #print("         F1:                 ", test_F1)
            
            print()
            print("KMMMANUAL Classification Err: ", manual_class_error)
            #print("         Precision:          ", manual_precision)
            #print("         Recall:             ", manual_recall)
            #print("         F1:                 ", manual_F1)
            
            print("_____________________________________________________")
            sys.stdout.flush()
            
            i += 1
            
        traintime_end = time.time()
        print("KMM model took ", (traintime_end - traintime_start)/60.0, " minutes to train")
        
    
    def train(self):
        losses_train = []
        losses_test = []
        losses_manual = []

        classerr_train = []
        classerr_test = []
        classerr_manual = []

        print()
        print()
        
        X_train = normalize(self.X_train, axis=1, norm='l1')
        X_test = normalize(self.X_test, axis=1, norm='l1')
        X_manual = normalize(self.X_manual, axis=1, norm='l1')
        
        traintime_start = time.time()
        for i in range(self.EPOCH):
            epoch_st = time.time()
            print()
            print("wFastText EPOCH: ", i)
            
            # linearly decaying lr alpha
            alpha = self.LR * ( 1 - i / self.EPOCH)
            
            l = 0
            
            for x in X_train:
                label = self.y_train[l]
                beta = self.betas[l]
                
                B_old = self.B
                A_old = self.A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(self.A, x.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(self.B, a1)
                Y_hat = self.stable_softmax(z2)
        
                # Back prop with alt optimization
                self.B = self.KMMgradient_B(B_old, A_old, label, alpha, a1, Y_hat, beta)  
                self.A = self.KMMgradient_A(B_old, A_old, x, label, alpha, Y_hat, beta)

            epoch_et = time.time()
            print("~~~~Epoch took ", (epoch_et - epoch_st)/60.0, " minutes")

            # TRAINING LOSS
            train_loss = self.get_total_loss(self.A, self.B, X_train, self.y_train, self.N_train)
            print("KMM Train:   ", train_loss)

            ## TESTING LOSS
            test_loss = self.get_total_loss(self.A, self.B, X_test, self.y_test, self.N_test)
            print("KMM Test:    ", test_loss)
            
            ## MANUAL SET TESTING LOSS
            manual_loss = self.get_total_loss(self.A, self.B, X_manual, self.y_manual, self.N_manual)
            print("KMM Manual:    ", manual_loss)
            print()

            losses_train.append(train_loss)
            losses_test.append(test_loss)
            losses_manual.append(manual_loss)
            
            train_class_error = self.metrics(X_train, self.y_train, self.A, self.B, self.N_train, 'KMMtrain', i)
            
            test_class_error = self.metrics(X_test, self.y_test, self.A, self.B, self.N_test, 'KMMtest', i)
            
            manual_class_error = self.metrics(X_manual, self.y_manual, self.A, self.B, self.N_manual, 'KMMmanual', i)
            
            
            classerr_train.append(train_class_error)
            classerr_test.append(test_class_error)
            classerr_manual.append(manual_class_error)

            print()
            print("KMMTRAIN Classification Err: ", train_class_error)
            #print("         Precision:          ", train_precision)
            #print("         Recall:             ", train_recall)
            #print("         F1:                 ", train_F1)

            print("KMMTEST Classification Err:", test_class_error)
            #print("         Precision:          ", test_precision)
            #print("         Recall:             ", test_recall)
            #print("         F1:                 ", test_F1)
            
            print("KMMMANUAL Classification Err: ", manual_class_error)
            #print("         Precision:          ", manual_precision)
            #print("         Recall:             ", manual_recall)
            #print("         F1:                 ", manual_F1)
            
            print("_____________________________________________________")
            sys.stdout.flush()
            
            
            i += 1
            
        traintime_end = time.time()
        print("~~~~KMM model took ", (traintime_end - traintime_start)/60.0, " minutes to train")
        
