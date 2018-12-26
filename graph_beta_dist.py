

from matplotlib import pyplot as plt
import os 
import pandas as pd
import numpy as np
import pickle



if __name__ == '__main__':

    wft_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext/'
    wft_ck_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext_ck/'
    wft_cf_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext_cf/'
    
    for filename in os.listdir(wft_betadir):
        if '_19.txt' in filename:
            pkl_file = open(wft_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_betas = dataft['Beta']
            
    for filename in os.listdir(wft_ck_betadir):
        if '_19.txt' in filename:
            pkl_file = open(wft_ck_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_ck_betas = dataft['Beta']
    
    for filename in os.listdir(wft_cf_betadir):
        if '_19.txt' in filename:
            pkl_file = open(wft_cf_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_cf_betas = dataft['Beta']
            
    
    directory = '/Users/madim/Desktop/ML_research/gitfiles/bias_vs_labelefficiency/indices/'
    for filename in os.listdir(directory):
        if '_19.txt' in filename:
            subset = np.loadtxt(directory+filename, dtype=np.object)
    
    subset = subset.astype(int).tolist()
    #sub = [self.file_train[i] for i in subset]
    
    
    labelfile = '/Users/madim/Desktop/ML_research/gitfiles/betas/train_trial10_epoch19.pkl'
    pkl_file = open(labelfile, 'rb')
    dataft = pickle.load(pkl_file)
    Y_true = dataft['Y_true']
    
    true_label_max = np.argmax(Y_true, axis=0)

    #plt.subplot(1, 2, 1)  
    print(wft_betas[true_label_max==0])
    
    index = np.arange(wft_betas[true_label_max==0].shape[0])
    index2 = np.arange(wft_betas[true_label_max==1].shape[0])
    
    print(index2)
    
    #plt.bar(index2, wft_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index, wft_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText")
    #plt.show()
    
    
    #index = np.arange(wft_ck_betas[true_label_max==0].shape[0])
    #index2 = np.arange(wft_ck_betas[true_label_max==1].shape[0])
    
    #plt.bar(index2, wft_ck_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index, wft_ck_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText-ck")
    #plt.show()
    
    
    #index = np.arange(wft_cf_betas[true_label_max==0].shape[0])
    #index2 = np.arange(wft_cf_betas[true_label_max==1].shape[0])
    
    #plt.bar(index2, wft_cf_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index, wft_cf_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText-cf")
    #plt.show()
    
    
    #index = np.arange(wft_betas.shape[0])
    #plt.bar(index, wft_betas.flatten(), color='g', alpha = 0.5)
    
    #index2 = np.arange(wft_ck_betas.shape[0])
    #plt.bar(index2, wft_ck_betas.flatten(), color='r', alpha = 0.5)
    
    #index3 = np.arange(wft_cf_betas.shape[0])
    #plt.bar(index3, wft_cf_betas.flatten(), color='b', alpha = 0.5)
    
    #plt.show()



    ##### METHOD 1 
    #r_female = 0.8
    #r_male = 1.2

    #beta[true_label_max==1] *= r_female
    #beta[true_label_max==0] *= r_male

    #plt.subplot(1, 2, 2)  
    #index = np.arange(beta[true_label_max==0].shape[0])
    #index2 = np.arange(beta[true_label_max==1].shape[0])
    #plt.bar(index, beta[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.bar(index2, beta[true_label_max==1].flatten(), color='g', alpha = 0.5)
    ##plt.xticks(index)
    #plt.xlabel("Male and Females beta weights 2 ")
    #plt.show()

