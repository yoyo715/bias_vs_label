# main.py

from dictionary import Dictionary
from dictionary_updated import Dictionary2
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

    

# finds gradient of B and returns an up
def gradient_B(B, A, x, label, nclasses, alpha, DIM, hidden, Y_hat):    
    gradient = alpha * np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
    B_new = np.subtract(B, gradient)

    return B_new


# update rule for weight matrix A
def gradient_A(B, A, x, label, nclasses, alpha, DIM, Y_hat, drop1):
    A_old = A
    first = np.dot(np.subtract(Y_hat.T, label), B)
    gradient = alpha * sparse.csr_matrix.dot(first.T, sec) * drop1
    A = np.subtract(A_old, gradient) 
    
    return A


def stable_softmax(x, A, B): 
    hidden = sparse.csr_matrix.dot(A, x.T)
    X = np.dot(B, hidden)
    exps = np.exp(X - np.max(X))
    return (exps / np.sum(exps))


# finds the loss
def loss_function(x, A, B, label):
    loglike = np.log(stable_softmax(x, A, B))
    return -np.dot(label, loglike)


# computes the loss over entire dataset
def total_loss_function(X, Y, A, B, N):
    i = 0
    total_loss = 0
    for x in X:
        label = Y[i]
        loss = loss_function(x, A, B, label)
        total_loss += loss
        i += 1
        
    return (1.0/N) * total_loss



# function to return prediction error, precision, recall, F1 score
def metrics(X, Y, A, B, N, b1, b2):
    incorrect = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0    
    
    y_true = []
    y_pred = []

    i = 0
    for x in X:
        prediction = np.argmax(stable_softmax(x, A, B, b1, b2))
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


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.1
    WORDGRAMS=3
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1000000
    EPOCH=20
    
    dropout_percent = 0.7

    print("starting dictionary creation") 
    
    # initialize training
    dictionary = Dictionary2(WORDGRAMS, MINCOUNT, BUCKET)
    nwords = dictionary.get_nwords()
    nclasses = dictionary.get_nclasses()
    
    #initialize testing
    X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    N = dictionary.get_n_train_instances()
    N_test = dictionary.get_n_test_instances()
    
    print("Number of Train instances: ", N, " Number of Test instances: ", N_test)
    ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
    ntest_eachclass = dictionary.get_nlabels_eachclass_test()
    print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)
    
    ##### instantiations #######################################


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

    class_error_train = []  
    class_error_test = []

    prec_train = [] 
    prec_test = []

    recall_train = []
    recall_test = []

    F1_train = []
    F1_test = []
    
    AUC_train = []
    AUC_test = []

    print()
    print()
    
    for i in range(EPOCH):
        print()
        print("EPOCH: ", i)
        
        # linearly decaying lr alpha
        alpha = LR * ( 1 - i / EPOCH)
        
        l = 0
        train_loss = 0
        
        # TRAINING
        for x in X_train:       
            
            label = y_train[l]
            B_old = B
            A_old = A
            
            # Forward Propogation
            hidden = sparse.csr_matrix.dot(A_old, x.T)
            z2 = np.dot(B, hidden)
            exps = np.exp(z2 - np.max(z2))
            Y_hat = exps / exps.sum(axis=0)
            
            # Error
            error = np.subtract(Y_hat.T, label)
            
            d1_dw2 = np.outer(hidden, error)
            d1_dw1 = sparse.csr_matrix.dot(np.dot(error, B).T, x)
            #d1_dw1 = np.outer(x, np.dot(B.T, error.T))
            
            A = A - (alpha * d1_dw1)
            B = B - (alpha * d1_dw2).T
            

            #loglike = np.log(exps)
            print(label)
            print(Y_hat)
            loss = -np.dot(label, Y_hat)
            print(loss)
            
            train_loss += loss
            
   
            l += 1
            
            
        ## TRAINING LOSS
        #train_loss = total_loss_function(X_train, y_train, A, B, N)
        print("Train:   ", train_loss)
            
        # TESTING LOSS
        test_loss = total_loss_function(X_test, y_test, A, B, N_test)
        print("Test:    ", test_loss)
        
        print("Difference = ", test_loss - train_loss)


        train_class_error, train_precision, train_recall, train_F1, train_AUC, train_FPR, train_TPR = metrics(X_train, y_train, A, B, N)
        test_class_error, test_precision, test_recall, test_F1, test_AUC, test_FPR, test_TPR = metrics(X_test, y_test, A, B, N_test)
        
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
        
        losses_train.append(train_loss)
        losses_test.append(test_loss)

        class_error_train.append(train_class_error)
        class_error_test.append(test_class_error)

        prec_train.append(train_precision)
        prec_test.append(test_precision)

        recall_train.append(train_recall)
        recall_test.append(test_recall)

        F1_train.append(train_F1)
        F1_test.append(test_F1)
        
        AUC_train.append(train_AUC)
        AUC_test.append(test_AUC)
        
        i += 1
        
        
    
    # for plotting
    epochs = [l for l in range(EPOCH)]

    plt.plot(epochs, losses_train, 'r', label="training loss")
    plt.plot(epochs, losses_test, 'b', label="testing loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(epochs, class_error_train, 'r', label="training classification err")
    plt.plot(epochs, class_error_test, 'b', label="testing classification err")
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(epochs, AUC_train, 'm', label="training AUC scores")
    plt.plot(epochs, AUC_test, 'c', label="testing AUC scores")
    plt.ylabel('AUC Scores')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
        
    plt.plot(epochs,  F1_train, 'm', label="training F1 score")
    plt.plot(epochs, F1_test, 'c', label="testing F1 score")
    plt.ylabel('F1 Score')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.title('FINAL Receiver Operating Characteristic (ROC curve)')
    plt.plot([0,1],[0,1],'r--')
    plt.plot(train_FPR, train_TPR, 'm', label="training, AUC score = %f" % train_AUC)
    plt.plot(test_FPR, test_TPR, 'c', label="testing, AUC score = %f" % test_AUC)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.legend(loc='upper left')
    plt.show()

    #plt.plot(recall_train, prec_train, 'm', label="training")
    #plt.plot(recall_test, prec_test, 'c', label="testing")
    #plt.ylabel('Precision')
    #plt.xlabel('Recall')
    #plt.legend(loc='upper left')
    #plt.show()
 
 
 
if __name__ == '__main__':
    main()






