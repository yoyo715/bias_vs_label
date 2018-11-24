import pickle
import numpy as np
import os
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

from dictionary3 import Dictionary

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





 
def main():
    
    #DIM=30
    #WORDGRAMS=2
    #MINCOUNT=2  #2 
    #MINN=3
    #MAXN=3
    #BUCKET=1000000

    ## adjust these
    #EPOCH=20
    #LR= 0.008                 #0.007            # 0.008 good for fasttext
    #KMMLR = 0.015         #0.015 pretty good

    #KERN = 'lin'        # lin or rbf or poly
    #NUM_RUNS = 10        # number of test runs
    #SUBSET_VAL = 500   # number of subset instances for self reported dataset
    #LIN_C = 0.9          # hyperparameter for linear kernel
    
    #BATCHSIZE = 100       # number of instances in each batch
    
    #model = 'original'
    
    #run = 0
    
    #dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model)
    
    #nwords = dictionary.get_nwords()
    #nclasses = dictionary.get_nclasses()
    
    ##initialize testing
    #X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #N_train = dictionary.get_n_train_instances()
    #N_test = dictionary.get_n_test_instances()
    
    #print("Number of Train instances: ", N_train, " Number of Test instances: ", N_test)
    #ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
    #ntest_eachclass = dictionary.get_nlabels_eachclass_test()
    #print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)
    
    
    
    ## manual labeled set (Kaggle dataset)
    #X_manual = dictionary.get_manual_testset()
    #y_manual = dictionary.get_manual_set_labels()
    #N_manual = dictionary.get_n_manual_instances()
    #print()
    #print("Number of Manual testing instances: ", N_manual, " shape: ", X_manual.shape)
    #nmanual_eachclass = dictionary.get_nlabels_eachclass_manual()
    #print("N each class Manual testing instances: ", nmanual_eachclass)
    
    
    
    epochs = ['epoch0.', 'epoch1.', 'epoch2.',
            'epoch3.',
            'epoch4.',
            'epoch5.',
            'epoch6.',
            'epoch7.',
            'epoch8.',
            'epoch9.',
            'epoch10.',
            'epoch11.',
            'epoch12.',
            'epoch13.',
            'epoch14.',
            'epoch15.',
            'epoch16.',
            'epoch17.',
            'epoch18.',
            'epoch19.']
    
    trials = ['trial0_',
            'trial1_',
            'trial2_',
            'trial3_',
            'trial4_',
            'trial5_',
            'trial6_',
            'trial7_',
            'trial8_',
            'trial9_',
            'trial10_',
            'trial11_',
            'trial12_',
            'trial13_',
            'trial14_',
            'trial15_',
            'trial16_',
            'trial17_',
            'trial18_',
            'trial19_',
            'trial20_',
            'trial21_',
            'trial22_',
            'trial23_',
            'trial24_',
            'trial25_',
            'trial26_',
            'trial27_',
            'trial28_',
            'trial29_'
            ]
    
    
    
    #directory_fasttext = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/KMMlabel_output/'
    directory_fasttext = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/KMMlabel_output/'

    outdir = '/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/label_analysis/KMManalysis/'
    #outdir = '/local_d/RESEARCH/NEW_OUTPUT2/label_analysis/'
    
    #s = np.empty([10000, 1])
    #print(s.shape)
    
    #true_label_max = np.argmax(y_train, axis=1)
    
    #dir = '/local_d/RESEARCH/NEW_OUTPUT1/beta3/'
    #for filenameft in os.listdir(dir):
        #print(filenameft)
        
        #pkl_fileft = open(dir+filenameft, 'rb')
        #dataft = pickle.load(pkl_fileft)
                    
        #beta = dataft['Beta']
        
        #print(beta.shape)
        
        #index = np.arange(beta[true_label_max==0].shape[0])
        #index2 = np.arange(beta[true_label_max==1].shape[0])
        #plt.bar(index2, beta[true_label_max==1].flatten(), color='g', alpha = 0.5)
        #plt.bar(index, beta[true_label_max==0].flatten(), color='b', alpha = 0.5)
        ##plt.xticks(index)
        #plt.xlabel("Male and Females beta weights ")
        #plt.show()
    
    missed_true_pos = 0 
    missed_true_neg = 0
    num = 0
    #training statistics
    for trial in trials:
        for epoch in epochs:
            for filenameft in os.listdir(directory_fasttext):
                if "train_" in filenameft and trial in filenameft and epoch in filenameft:
                    #print(filenameft)
                    pkl_fileft = open(directory_fasttext+filenameft, 'rb')
                    dataft = pickle.load(pkl_fileft)
                    
                    Y_true = dataft['Y_true']
                    Y_predicted = dataft['Y_predicted']
                    N = Y_true.shape[0]
                    
                    pkl_fileft.close()
                    
                    # compare to actual classes
                    prediction_max = np.argmax(Y_predicted, axis=0)
                    true_label_max = np.argmax(Y_true, axis=0)
                    
                    unique, counts = np.unique(true_label_max, return_counts=True)
                    # males -> 0 , true neg
                    N_trueneg = counts[0]
                    # females -> 1, true pos
                    N_truepos = counts[1]
                    
                    #print(N_trueneg, N_truepos)
                    #print(confusion_matrix(true_label_max, prediction_max))
                    
                    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
                    #print(true_neg, false_pos, false_neg, true_pos)
                    
                    #print(1.0 - true_neg/N_trueneg )
                    missed_true_neg += 1.0 - true_neg/N_trueneg 
                    #print(missed_true_neg)
                    
                    missed_true_pos += 1.0 - true_pos/N_truepos 
                    #print(missed_true_pos)
                    
                    #print()
                    
                    #fname = outdir+'KMMTRAIN_percent_truepos_missed.txt'
                    #with open(fname, 'a+') as f:
                    #    f.write("%s," % missed_true_pos)
                        
                    #fname2 = outdir+'KMMTRAIN_percent_trueneg_missed.txt'
                    #with open(fname2, 'a+') as f:
                    #    f.write("%s," % missed_true_neg)
                        
                        
        #fname = outdir+'KMMTRAIN_percent_truepos_missed.txt'
        #with open(fname, 'a+') as f:
        #    f.write('\n')
            
        #fname2 = outdir+'KMMTRAIN_percent_trueneg_missed.txt'
        #with open(fname2, 'a+') as f:
        #    f.write('\n')
            
    print(" missed_true_neg ", missed_true_neg, N_trueneg)
    print("missed_true_pos ", missed_true_pos / N_truepos )
        
    '''    
    missed_true_pos = 0 
    missed_true_neg = 0
    # testing statistics
    for trial in trials:
        for epoch in epochs:
            for filenameft in os.listdir(directory_fasttext):
                if "test" in filenameft and trial in filenameft and epoch in filenameft:
                    #print(filenameft)
                    pkl_fileft = open(directory_fasttext+filenameft, 'rb')
                    dataft = pickle.load(pkl_fileft)
                    
                    Y_true = dataft['Y_true']
                    Y_predicted = dataft['Y_predicted']
                    N = Y_true.shape[0]
                    
                    pkl_fileft.close()
                    
                    # compare to actual classes
                    prediction_max = np.argmax(Y_predicted, axis=0)
                    true_label_max = np.argmax(Y_true, axis=0)
                    
                    unique, counts = np.unique(true_label_max, return_counts=True)
                    # males -> 0 , true neg
                    N_trueneg = counts[0]
                    # females -> 1, true pos
                    N_truepos = counts[1]
                    
                    #print(N_trueneg, N_truepos)
                    #print(confusion_matrix(true_label_max, prediction_max))
                    
                    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
                    #print(true_neg, false_pos, false_neg, true_pos)
                    
                    missed_true_neg += 1.0 - true_neg/N_trueneg
                    #print(missed_true_neg)
                    
                    missed_true_pos += 1.0 - true_pos/N_truepos
                    #print(missed_true_pos)
                    
                    #print()
                    
                    #fname = outdir+'KMMTEST_percent_truepos_missed.txt'
                    #with open(fname, 'a+') as f:
                    #    f.write("%s," % missed_true_pos)
                        
                    #fname2 = outdir+'KMMTEST_percent_trueneg_missed.txt'
                    #with open(fname2, 'a+') as f:
                    #    f.write("%s," % missed_true_neg)
                        
                        
        #fname = outdir+'KMMTEST_percent_truepos_missed.txt'
        #with open(fname, 'a+') as f:
            #f.write('\n')
            
        #fname2 = outdir+'KMMTEST_percent_trueneg_missed.txt'
        #with open(fname2, 'a+') as f:
            #f.write('\n')
            
    print("TEST missed_true_neg ", missed_true_neg / N_trueneg)
    print("TEST missed_true_pos ", missed_true_pos / N_truepos )
    
    
    missed_true_pos = 0 
    missed_true_neg = 0
    # manual statistics
    for trial in trials:
        for epoch in epochs:
            for filenameft in os.listdir(directory_fasttext):
                if "manual" in filenameft and trial in filenameft and epoch in filenameft:
                    #print(filenameft)
                    pkl_fileft = open(directory_fasttext+filenameft, 'rb')
                    dataft = pickle.load(pkl_fileft)
                    
                    Y_true = dataft['Y_true']
                    Y_predicted = dataft['Y_predicted']
                    N = Y_true.shape[0]
                    
                    pkl_fileft.close()
                    
                    # compare to actual classes
                    prediction_max = np.argmax(Y_predicted, axis=0)
                    true_label_max = np.argmax(Y_true, axis=0)
                    
                    unique, counts = np.unique(true_label_max, return_counts=True)
                    # males -> 0 , true neg
                    N_trueneg = counts[0]
                    # females -> 1, true pos
                    N_truepos = counts[1]
                    
                    #print(N_trueneg, N_truepos)
                    #print(confusion_matrix(true_label_max, prediction_max))
                    
                    true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
                    #print(true_neg, false_pos, false_neg, true_pos)
                    
                    missed_true_neg += 1.0 - true_neg/N_trueneg
                    #print(missed_true_neg)
                    
                    missed_true_pos == 1.0 - true_pos/N_truepos
                    #print(missed_true_pos)
                    
                    #print()
                    
                    #fname = outdir+'KMMMANUAL_percent_truepos_missed.txt'
                    #with open(fname, 'a+') as f:
                    #    f.write("%s," % missed_true_pos)
                        
                    #fname2 = outdir+'KMMMANUAL_percent_trueneg_missed.txt'
                    #with open(fname2, 'a+') as f:
                    #    f.write("%s," % missed_true_neg)
                        
                      
    print("man missed_true_neg ", missed_true_neg / N_trueneg)
    print("man missed_true_pos ", missed_true_pos / N_truepos )
        #fname = outdir+'KMMMANUAL_percent_truepos_missed.txt'
        #with open(fname, 'a+') as f:
            #f.write('\n')
            
        #fname2 = outdir+'KMMMANUAL_percent_trueneg_missed.txt'
        #with open(fname2, 'a+') as f:
            #f.write('\n')
                    
                    
'''
 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
