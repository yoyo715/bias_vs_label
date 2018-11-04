

import pickle
import numpy as np
from dictionary3 import Dictionary
import time
from scipy import sparse
import os


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

# create a list of list of sort_filenames
# ie list = [trial1[epoch1, ... ], .... trialn[epoch1, ....] ]
def sort_filenames(directory):
    for filename in os.listdir(directory):
        
        
def get_man_statesets(directory):
    for filename in os.listdir(directory):
        
        
def get_man_countrysets(directory):
    for filename in os.listdir(directory):
        

def get_self_statesets(directory):
    for filename in os.listdir(directory):
        

def get_self_countrysets(directory):
    for filename in os.listdir(directory):


# this function creates the instances of the manually labeled (Kaggle) dataset
def testing_create_instances_and_labels(file_):
    words =  []
    labels = []
    documents = []
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')
    num = 0
    
    #print(self.manual_set[358])

    # loop through each instance in training data, gets labels
    for x in file_:
        if num != 361 and num != 360 and num != 359:
            i = 0
            inst = ''
            label = x[0:10]
            if label[0:9] != '__label__':
                print("ERROR in label creation. Label: ", label)
                break
            else:
                labels.append(float(label[-1]))
                
            sent = ''
            word = ''
            for w in x[10:]:
                if w in whitelist:
                    if w == '\t':
                        inst = inst + '\t' + sent
                        sent = ''
                        word = ''
                        i += 1
                    elif w != ' ':
                        word = word + w
                    else:
                        if "http" not in word and word != "RT" and word != "rt":
                            sent = sent + ' ' + word
                            word = ''
                        else:
                            word = ''
            
            documents.append(inst)
        num += 1
    self.manual_instances = documents
    self.y_manual = labels
    
    self.n_manual_instances = len(self.manual_instances)
    
    
# index 0: label 0
# index 1: label 1
def create_statecountry_labels(n, nclasses, y):
    labels = np.zeros((n, nclasses))
    
    nummales = 0
    numfemales = 0
    
    i = 0
    for label in labels:
        if y[i] == 0:
            label[0] = 1.0
            nummales += 1         
        elif y[i] == 1:
            label[1] = 1.0
            numfemales += 1       
        
        i += 1
        
    return labels



def write_stats(directory, loss, class_error, precision, recall, F1, AUC):
    
    #### WRITING LOSSES
    with open(directory+'/loss_train.txt', '+a') as f:
        f.write("%s," % loss)
            
    #### WRITING ERROR
    with open(directory+'output/error_train.txt', '+a') as f:
        f.write("%s," % class_error)
            
    #### WRITING PRECISION
    with open(directory+'output/precision_train.txt', '+a') as f:
        f.write("%s," % precision)
            
    #### WRITING RECALL
    with open(directory+'output/recall_train.txt', '+a') as f:
        f.write("%s," % recall)
            
    #### WRITING F1
    with open(directory+'output/F1_train.txt', '+a') as f:
        f.write("%s," % F1)
            
    #### WRITING AUC
    with open(directory+'output/AUC_train.txt', '+a') as f:
        f.write("%s," % AUC)
            


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
    
    directory_fasttext = ''
    directory_fastKMMtext = ''
    sortedfilenames_fasttext = sort_filenames(directory_fasttext)
    sortedfilenames_fastKMMtext = sort_filenames(directory_fastKMMtext)
    
    
    man_states = get_man_statesets()
    man_countries = get_man_countrysets()
    
    self_states = get_self_statesets()
    self_countries = get_self_countrysets()


    for trial in sortedfilenames:
        ###### Create new dictionary at for each trial
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
        
        for epochname in trial:
        
            filenameft = directory_fasttext+epochname+'.pkl'
            pkl_fileft = open(filenameft, 'rb')
            
            filenamefkmmt = directory_fastKMMtext+epochname+'.pkl'
            pkl_filefkmmt = open(filenamefkmmt, 'rb')

            dataft = pickle.load(pkl_fileft)
            datafkmmt = pickle.load(pkl_filefkmmt)

            A_ft = dataft['A']
            B_ft = dataft['B']
            
            A_fkmmt = datafkmmt['A']
            B_fkmmt = datafkmmt['B']

            print("Bucket size ", A_ft.shape[1], A_fkmmt.shape[1])
            
            pkl_fileft.close()
            pkl_filefkmmt.close() 
            
            
            ############## now testing on state and country datasets
            
            # test on all manual state datasets
            for state in man_states:
                s = open(state, encoding='utf8').readlines()
                instances = testing_create_instances_and_labels(s)
                X_state = dictionary.create_statecountry_bagngrams(instances)
                y_state = create_statecountry_labels()
                
                class_error, precision, recall, F1, roc_auc, fpr, tpr = metrics(X_state, y_test, A_ft, B_ft, N)
                print(state, " class error (fasttext): ", class_error)
                
                kmmclass_error, kmmprecision, kmmrecall, kmmF1, kmmroc_auc, kmmfpr, kmmtpr = metrics(X_state, y_test, A_fkmmt, B_fkmmt, N)
                print(state, " class error (fastKMMText): ", class_error)
                
                # write to file
                dirft = 'stateoutput_FT/'
                dirfkmmt = 'stateoutput_FKMMT/'
                
                write_stats(dirft, class_error, precision, recall, F1, roc_auc)
                write_stats(dirfkmmt, kmmclass_error, kmmprecision, kmmrecall, kmmF1, kmmroc_auc)
                
            for country in man_countries:
                
                
            for state in self_states:
                
                
            for country in self_countries:
                
                
            
            



    
 
if __name__ == '__main__':
    main()

    
    

