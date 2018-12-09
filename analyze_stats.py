import pandas as pd
import numpy as np
import os
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


 
def main():
   
    EPOCH = 20
 
    ftdir = '/project/lsrtwitter/mcooley3/output_data/fasttext/stats/'
    wftdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext/stats/'
    wftcfdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext_cf/stats/'
    wftckdir = '/project/lsrtwitter/mcooley3/output_data/wfasttext_ck/stats/'

    #dataset = ['manual', 'train', 'test']
    dirs = [ftdir, wftdir, wftcfdir, wftckdir]

    train_err = pd.DataFrame()
    test_err = pd.DataFrame()
    man_err = pd.DataFrame()
    
    for d in dirs:
        print(d, " ***********************************************")
        num = 0
        
        for filenameft in os.listdir(d):
            print(filenameft)
            if "manual" in filenameft:
                class_error = pd.read_csv(d+filenameft, sep=",", header=None)
                print(class_error)
                man_err.concat(class_error)
                
            elif "test" in filenameft:
                class_error = pd.read_csv(d+filenameft, sep=",", header=None)
                #print(class_error)
                test_err.concat(class_error)
            
            elif "train" in filenameft:
                class_error = pd.read_csv(d+filenameft, sep=",", header=None)
                #print(class_error)
                train_err.concat(class_error)
                    
        # manual
        man_err = man_err.drop(man_err.columns[-1],axis=1)
        man_summary = man_err.describe()
        man_mean = np.array(man_summary.loc[['mean']])
        man_std = np.array(man_summary.loc[['std']])
        man_mean.resize((EPOCH))
        man_std.resize((EPOCH))
        print(man_err)
        
        # test
        test_err = test_err.drop(test_err.columns[-1],axis=1)
        test_summary = test_err.describe()
        test_mean = np.array(test_summary.loc[['mean']])
        test_std = np.array(test_summary.loc[['std']])
        test_mean.resize((EPOCH))
        test_std.resize((EPOCH))
        
        # train
        train_err = train_err.drop(train_err.columns[-1],axis=1)
        train_summary = train_err.describe()
        train_mean = np.array(train_summary.loc[['mean']])
        train_std = np.array(train_summary.loc[['std']])
        train_mean.resize((EPOCH))
        train_std.resize((EPOCH))
                
        
 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
