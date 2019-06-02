#CLASS_wfasttext-ck_new.py

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


"""
    wFastText-ck:
        beta_i = r_0*beta
        beta)i = r_1*beta


    Reweight beta coefficients AFTER optimization.
    
"""


class wFastText_ck_new:
    def __init__(self, dictionary, learning_rate, DIM, EPOCH, kmmB, batchsize, kernel, r_female, r_male):
        print()
        print("######################## wFastText-ck_new ########################")
        
        #self.save_dir = '/project/lsrtwitter/mcooley3/APRIL_2019_exps/new_wfasttext-ck/'
        self.save_dir = '/project/lsrtwitter/mcooley3/RACE_JUNE_2019_exps/new_wfasttext-ck/'
        
        self.LR = learning_rate
        self.EPOCH = EPOCH
        self.kmmB = kmmB
        self.BATCHSIZE = batchsize
        self.kernel = kernel
        self.run_number = dictionary.run_number  
        
        self.r_female = r_female   #0.5
        self.r_male = r_male       #2.0
    
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

        
        
###########################################################################################################

    def create_optbeta(self):
        print("starting beta optimization..............................")
        
        start = time.time()
        
        X = sparse.csr_matrix.dot(self.A, self.X_RTEST.T)
        Z = sparse.csr_matrix.dot(self.A, self.X_STRAIN.T)
        
        opt_beta = self.kernel_mean_matching(X.T, Z.T, kern=self.kernel, B=self.kmmB, eps=None)
        
        ####### wFastText-ck METHOD: ######
        true_label_max = np.argmax(self.Y_STRAIN, axis=1)
        
        opt_beta[true_label_max==1] *= self.r_female
        opt_beta[true_label_max==0] *= self.r_male
        
        end = time.time()
        print("Beta took ", (end - start)/60.0, " minutes to optimize.")
        print("About Beta: ")
        print(stats.describe(opt_beta))
        print()
        sys.stdout.flush()
        
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
            K = K.todense()
            kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
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
    
    
    def save_betas_yhat_y(self, epoch, betas, Y_STRAIN, Y_SVAL, Y_RTEST, Y_RVAL, Y_STEST,
                                   yhat_strain, yhat_sval, yhat_rtest, yhat_rval, yhat_stest):
        fname = 'NEW_wfasttext_cf_RUN'+self.run_number+'_EPOCH'+str(epoch)+'.pkl'
        
        data =  {   'betas': betas,
                    'Y_STRAIN': Y_STRAIN,
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

        print("INITIAL KMM_STRAIN Classification Err: ", strain_class_error)
        print("INITIAL KMM_SVAL Classification Err: ", sval_class_error)
        print("INITIAL KMM_RTEST Classification Err: ", rtest_class_error)
        print("INITIAL KMM_RVAL Classification Err: ", rval_class_error)
        print("INITIAL KMM_STEST Classification Err: ", stest_class_error)
        print()
            
        print("_____________________________________________________")
        
        traintime_start = time.time()
        
        for i in range(self.EPOCH):
            print()
            print("wFastText-ck_new EPOCH: ", i)
            epoch_st = time.time()

            # linearly decaying lr alpha
            alpha = self.LR * ( 1 - i / self.EPOCH)
 
            # NOTE: optimal KMM reweighting coefficient
            self.betas = self.create_optbeta()  
            
            print("starting training with new betas")
            
            # Shuffle data
            batch_indices = np.random.permutation(self.N_strain)
            X_strain_batch = X_strain.tocsr()[batch_indices]
            y_train_batch = self.Y_STRAIN[batch_indices]
            betas = self.betas[batch_indices]

            for j in range(0, self.N_strain, self.BATCHSIZE):                
                batch = X_strain_batch[j:j+self.BATCHSIZE]
                y_batch = y_train_batch[j:j+self.BATCHSIZE]
                beta_batch = betas[j:j+self.BATCHSIZE]

                B_old = self.B
                A_old = self.A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(self.A, batch.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(self.B, a1)
                Y_hat = self.stable_softmax(z2)
                        
                # Back prop with alt optimization
                self.B = self.KMMgradient_B(B_old, A_old, y_batch, alpha, a1, Y_hat, beta_batch)  
                self.A = self.KMMgradient_A(B_old, A_old, batch, y_batch, alpha, Y_hat, beta_batch)
                
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


            print("KMM_STRAIN Classification Err: ", strain_class_error)
            print("KMM_SVAL Classification Err: ", sval_class_error)
            print("KMM_RTEST Classification Err: ", rtest_class_error)
            print("KMM_RVAL Classification Err: ", rval_class_error)
            print("KMM_STEST Classification Err: ", stest_class_error)
            print()
            
            
            #self.save_betas_yhat_y(i, self.betas, self.Y_STRAIN, self.Y_SVAL, self.Y_RTEST, self.Y_RVAL, self.Y_STEST,
                                   #yhat_strain, yhat_sval, yhat_rtest, yhat_rval, yhat_stest)
    
            print("~~~~Epoch took ", (epoch_et - epoch_st)/60.0, " minutes")            
            print()
            print("_____________________________________________________")
            sys.stdout.flush()
            
            i += 1
            
        traintime_end = time.time()
        print("wFastText-ck_new model took ", (traintime_end - traintime_start)/60.0, " minutes to train")
        
        
