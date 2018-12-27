#CLASS_wfasttext-ck.py

import  os, sys, math, time
import  numpy as  np 
import scipy as sp
from cvxopt import matrix, solvers, spmatrix, sparse, mul


"""
    wFastText-ck:
        beta_i = r_0*beta
        beta)i = r_1*beta


    Reweight beta coefficients AFTER optimization.
    
"""


class wFastText_ck:
    def __init__(self, dictionary, learning_rate):
        self.LR = learning_rate
        
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
        
        
        # A
        p = X_train.shape[1]    # cols
        A_n = p
        A_m = DIM               # rows
        uniform_val = 1.0 / DIM
        np.random.seed(0)
        self.A = np.random.uniform(-uniform_val, uniform_val, (A_m, A_n))

        # B
        B_n = DIM               # cols
        B_m = nclasses          # rows
        self.B = np.zeros((B_m, B_n))
        
        
        ####### wFastText-ck METHOD: ######
        self.lin_c = 0.9                    # hyperparameter for linear kernel
        self.kernel = 'lin'   
        self.betas = create_optbeta()       # NOTE: optimal KMM reweighting coefficient
        
        true_label_max = np.argmax(y_train, axis=1)
        
        r_female = 0.5
        r_male = 2.0
        
        self.betas[true_label_max==1] *= r_female
        self.betas[true_label_max==0] *= r_male
        
        
###########################################################################################################

    def create_optbeta(self):
        print("starting beta optimization..............................")
        
        start = time.time()
        
        opt_beta = kernel_mean_matching(self.manual_test_bag_ngrams, self.train_bag_ngrams, self.lin_c, kern=self.kernel, B=6.0, eps=None)
        
        end = time.time()
        print("Beta took ", (end - start)/60.0, " minutes to optimize.")
        
        return opt_beta
      
      
    # Z is training data, X is testing data
    def kernel_mean_matching(X, Z, lin_c, kern='lin', B=1.0, eps=None):
        nx = X.shape[0]
        nz = Z.shape[0]
        
        print("nx: ", nx, " nz: ", nz)
        
        if eps == None:
            eps = B/math.sqrt(nz)
            
        if kern == 'lin':
            K = np.dot(Z, Z.T) 
            K = K.todense() + lin_c   #+ 0.9
            kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
            
        #elif kern == 'rbf':
            #K = compute_rbf(Z,Z)
            #kappa = np.sum(compute_rbf(Z,X),axis=1)*float(nz)/float(nx)
            
        else:
            raise ValueError('unknown kernel')
        
        
        K = K.astype(np.double)
        K = matrix(K)
        
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
        h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
        
        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])
        return coef


    # doesnt work
    def compute_rbf(X, Z, sigma=1.0):
        K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
        Z = Z.todense()
        
        for i, vx in enumerate(X):
            vx = vx.todense()
            K[i,:] = np.exp(-np.sum(np.square(vx-Z), axis=1)/(2.0*sigma)).flatten()
        return K
    

###########################################################################################################

      
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
        
        
    def train_batch():
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
            print("wFastText EPOCH: ", i)
            
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
            
            
            #write_fastKMMtext_stats(trialnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                                    #train_AUC, test_loss, test_class_error, test_precision, test_recall,
                                    #test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                                    #manual_recall, manual_F1, manual_AUC)
            
            #fnameB = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_B_"+str(trialnum)+"epoch"+str(i)  #+".pkl"
            #fnameA = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_A_"+str(trialnum)+"epoch"+str(i)
            #save_model_tofile(A, B, fnameB, fnameA)
            
            i += 1
            
        traintime_end = time.time()
        print("KMM model took ", (traintime_end - traintime_start)/60.0, " time to train")
        
    
    def train():
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
            print("wFastText EPOCH: ", i)
            
            # linearly decaying lr alpha
            alpha = LR * ( 1 - i / EPOCH)
            
            l = 0
            
            for x in X_train:
                label = y_train[l]
                beta = betas[l]
                B_old = B
                A_old = A
                
                # Forward Propogation
                hidden = sparse.csr_matrix.dot(A, x.T)
                a1 = normalize(hidden, axis=0, norm='l1')
                z2 = np.dot(B, a1)
                Y_hat = stable_softmax(z2)
        
                # Back prop with alt optimization
                B = KMMgradient_B(B_old, A_old, label, alpha, a1, Y_hat, beta)  
                A = KMMgradient_A(B_old, A_old, x, label, alpha, Y_hat, beta)

                # TRAINING LOSS
                train_loss = get_total_loss(A, B, X_train, y_train, N_train)
                print("KMM Train:   ", train_loss)

                ## TESTING LOSS
                test_loss = get_total_loss(A, B, X_test, y_test, N_test)
                print("KMM Test:    ", test_loss)
                
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
            
            
            #write_fastKMMtext_stats(trialnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                                    #train_AUC, test_loss, test_class_error, test_precision, test_recall,
                                    #test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                                    #manual_recall, manual_F1, manual_AUC)
            
            #fnameB = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_B_"+str(trialnum)+"epoch"+str(i)  #+".pkl"
            #fnameA = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial_A_"+str(trialnum)+"epoch"+str(i)
            #save_model_tofile(A, B, fnameB, fnameA)
            
            i += 1
            
        traintime_end = time.time()
        print("KMM model took ", (traintime_end - traintime_start)/60.0, " time to train")
        
