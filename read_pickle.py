

import pickle
import numpy as np
from dictionary3 import Dictionary
import time
from scipy import sparse

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


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



# function to return prediction error, precision, recall, F1 score
def metrics(X, Y, A, B, N):
    # get predicted classes
    print(A.shape, X.shape)

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
    
    print(confusion_matrix(true_label_max, prediction_max))

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
    
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(true_label_max, prediction_max)
    roc_auc = auc(fpr, tpr)
    
    print("AUC score: ", roc_auc)
    print()

    precision = true_pos / (true_pos + false_pos)           # true pos rate (TRP)
    recall = true_pos / (true_pos + false_neg)              # 
    F1 = 2 * ((precision * recall) / (precision + recall))

    return class_error, precision, recall, F1, roc_auc, fpr, tpr


def main():
    # args from Simple Queries paper
    DIM=30
    WORDGRAMS=2
    MINCOUNT=3
    MINN=3
    MAXN=3
    #BUCKET=1000000

    # adjust these
    EPOCH=20
    LR= 0.008                 #0.007            # 0.008 good for fasttext
    KMMLR = 0.014         #0.015 pretty good

    KERN = 'lin'        # lin or rbf or poly
    NUM_RUNS = 5        # number of test runs
    SUBSET_VAL = 10000   # number of subset instances for self reported dataset
    LIN_C = 0.9          # hyperparameter for linear kernel
    
    BATCHSIZE = 100       # number of instances in each batch
    
    #model = 'kmm'
    model = 'original'   # 'kmm' for kmm implementation

    #########################################################
    
    pkl_file = open('/home/mcooley/Desktop/bias_vs_labelefficiency/kmmmodels/fastKMMtext_trial10epoch0.pkl', 'rb')

    data1 = pickle.load(pkl_file)

    A = data1['A']
    B = data1['B']

    BUCKET = A.shape[1]
    print("Bucket size ", BUCKET)

    c = np.dot(B, A)

    print(c.shape)

    pkl_file.close()
        
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, model)

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


    ##############################

    
    class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics(X_test, y_test, A, B, N_train)
    print("class error: ", class_error)



    
 
if __name__ == '__main__':
    main()

    
    

