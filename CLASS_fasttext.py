#CLASS_fasttext.py

"""


"""


class FastText:
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
    
    
    def train_batch():
        #def train_fasttext(EPOCH, LR, BATCHSIZE, X_train, X_test, X_manual, y_train, y_test,
                   #y_manual, nclasses, A, B, N_train, N_test, N_manual, trialnum, dictionary):

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
            
            
            #write_fasttext_stats(trialnum, train_loss, train_class_error, train_precision, train_recall, train_F1,
                                #train_AUC, test_loss, test_class_error, test_precision, test_recall,
                                #test_F1, test_AUC, manual_loss, manual_class_error, manual_precision,
                                #manual_recall, manual_F1, manual_AUC)
            
            ##fname = "./models/fasttext_trial"+str(trialnum)+"epoch"+str(i)+".pkl"
            ##save_model_tofile(A, B, fname)
            
            #fnameB = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/models/fasttext_trial_B_"+str(trialnum)+"epoch"+str(i)  #+".pkl"
            #fnameA = "/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/models/fasttext_trial_A_"+str(trialnum)+"epoch"+str(i)
            #save_model_tofile(A, B, fnameB, fnameA)
            
            
            i += 1
            
        traintime_end = time.time()
        print("model took ", (traintime_end - traintime_start)/60.0, " time to train")
        
    
