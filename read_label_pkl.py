import pickle
import numpy as np
import os
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


 
def main():
    
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
    
    
    ftdir = '/project/lsrtwitter/mcooley3/output_data/fasttext/model_outputs/labels'
    wftdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext/model_outputs/labels'
    wftcfdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext_cf/model_outputs/labels'
    wftckdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext_ck/model_outputs/labels'
    
    
    #list_ = ["train_", "test_", "manual"]
    list_ = ["train_", "test_", "manual"]
    
    dirs = [ftdir, wftdir, wftcfdir, wftckdir]
    
    for d in dirs:
        for type in list_:
            
            missed_truepos = []
            missed_trueneg = []
        
            missed_true_pos = 0 
            missed_true_neg = 0
            num = 0
            
            #training statistics
            for trial in trials:
                for epoch in epochs:
                    for filenameft in os.listdir(d):
                        if type in filenameft and trial in filenameft and epoch in filenameft:
                            pkl_fileft = open(d+filenameft, 'rb')
                            dataft = pickle.load(pkl_fileft)
                            
                            Y_true = dataft['Y_true']
                            Y_predicted = dataft['Y_predicted']
                            N = Y_true.shape[0]
                            pkl_fileft.close()
                            
                            # compare to actual classes
                            prediction_max = np.argmax(Y_predicted, axis=0)
                            true_label_max = np.argmax(Y_true, axis=0)
                            unique, counts = np.unique(true_label_max, return_counts=True)
                            
                            N_trueneg = counts[0]       # males -> 0 , true neg
                            N_truepos = counts[1]       # females -> 1, true pos
                            
                            true_neg, false_pos, false_neg, true_pos = confusion_matrix(true_label_max, prediction_max).ravel()
                            
                            # number of missed true neg and pos
                            missed_true_neg = 1.0 - true_neg/N_trueneg
                            missed_true_pos = 1.0 - true_pos/N_truepos
                            
                            missed_trueneg.append(missed_true_neg)
                            missed_truepos.append(missed_true_pos)
                            
            
            print(d)
            print("Avg. ", type, " missed true neg = ", sum(missed_trueneg)/len(missed_trueneg))
            print("Avg. ", type, " missed true pos = ", sum(missed_truepos)/len(missed_truepos))
            print()
        
 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
