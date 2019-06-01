import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix


def main():
    ftdir = '../APIRL_TRIALS/fasttext/'
    wftdir = '../APIRL_TRIALS/new_wfasttext/'
    wftckdir = '../APIRL_TRIALS/new_wfasttext-ck/'
    
    dirs = [ftdir, wftdir, wftckdir]
    
    epochs = ['EPOCH0.', 'EPOCH1.', 'EPOCH2.',
            'EPOCH3.',
            'EPOCH4.',
            'EPOCH5.',
            'EPOCH6.',
            'EPOCH7.',
            'EPOCH8.',
            'EPOCH9.',
            'EPOCH10.',
            'EPOCH11.',
            'EPOCH12.',
            'EPOCH13.',
            'EPOCH14.',
            'EPOCH15.',
            'EPOCH16.',
            'EPOCH17.',
            'EPOCH18.',
            'EPOCH19.']
    
    trials = ['RUN0_',
            'RUN1_',
            'RUN2_',
            'RUN3_',
            'RUN4_',
            'RUN5_',
            'RUN6_',
            'RUN7_',
            'RUN8_',
            'RUN9_',
            'RUN10_',
            'RUN11_',
            'RUN12_',
            'RUN13_',
            'RUN14_',
            'RUN15_',
            'RUN16_',
            'RUN17_',
            'RUN18_',
            'RUN19_',
            ]
    
    for d in dirs:
            
        missed_truepos = []
        missed_trueneg = []
    
        missed_true_pos = 0 
        missed_true_neg = 0
        num = 0
        
        #training statistics
        for trial in trials:
            for epoch in epochs:
                for filenameft in os.listdir(d):
                    if trial in filenameft and epoch in filenameft:
                        pkl_fileft = open(d+filenameft, 'rb')
                        dataft = pickle.load(pkl_fileft)
                        
                        #Y_true = dataft['Y_RVAL']           # Random set
                        #Y_predicted = dataft['yhat_rval']   # Random set
                        
                        Y_true = dataft['Y_STEST']           # sl val set
                        Y_predicted = dataft['yhat_stest']   # sl val set
                        
                        
                        N = Y_true.T.shape[0]
                        pkl_fileft.close()
                        
                        # compare to actual classes
                        prediction_max = np.argmax(Y_predicted, axis=0)
                        true_label_max = np.argmax(Y_true.T, axis=0)
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
        print("Avg. missed true neg = ", sum(missed_trueneg)/len(missed_trueneg))
        print("Avg. missed true pos = ", sum(missed_truepos)/len(missed_truepos))
        print()


 
if __name__ == '__main__':
    main()