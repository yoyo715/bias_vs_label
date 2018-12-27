# graph_beta_dist.py

from matplotlib import pyplot as plt
import os 
import pandas as pd
import numpy as np
import pickle


"""
    This script graphs the beta distributions created by
    each method, wFastText, wFastText-ck, and wFastText-cf
    
"""

if __name__ == '__main__':

    # laptop
    #wft_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext/'
    #wft_ck_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext_ck/'
    #wft_cf_betadir = '/Users/madim/Desktop/ML_research/gitfiles/betas/wfasttext_cf/'
    #labelfile = '/Users/madim/Desktop/ML_research/gitfiles/betas/train_trial10_epoch19.pkl'
    
    # desktop
    wft_betadir = '../betas/wfasttext/'
    wft_ck_betadir = '../betas/wfasttext_ck/'
    wft_cf_betadir = '../betas/wfasttext_cf/'
    labelfile = '../betas/train_trial10_epoch19.pkl'
    
    
    for filename in os.listdir(wft_betadir):
        if '_10.txt' in filename:
            pkl_file = open(wft_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_betas = dataft['Beta']
            
    for filename in os.listdir(wft_ck_betadir):
        if '_10.txt' in filename:
            pkl_file = open(wft_ck_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_ck_betas = dataft['Beta']
    
    for filename in os.listdir(wft_cf_betadir):
        if '_10.txt' in filename:
            pkl_file = open(wft_cf_betadir+filename, 'rb')
            dataft = pickle.load(pkl_file)
            wft_cf_betas = dataft['Beta']
    
    
    pkl_file = open(labelfile, 'rb')
    dataft = pickle.load(pkl_file)
    Y_true = dataft['Y_true']
    
    true_label_max = np.argmax(Y_true, axis=0)


    # All methods together
    index = np.arange(wft_betas.shape[0])
    index2 = np.arange(wft_ck_betas.shape[0])
    index3 = np.arange(wft_cf_betas.shape[0])
    
    plt.bar(index, wft_betas.flatten(), color='g', alpha = 0.5)
    plt.bar(index, wft_ck_betas.flatten(), color='r', alpha = 0.5)
    plt.bar(index, wft_cf_betas.flatten(), color='b', alpha = 0.5)
    plt.xlabel("Male and Females beta weights - All methods")
    plt.show()


    index = [i for i,x in enumerate(true_label_max) if x == 1]
    index2 = [i for i,x in enumerate(true_label_max) if x == 0]
    

    # wFastText
    #plt.bar(index, wft_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index2, wft_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText")
    #plt.show()
    
    
    #wFastText-ck
    #plt.bar(index, wft_ck_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index2, wft_ck_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText-ck")
    #plt.show()

    
    #wFastText-cf
    #plt.bar(index, wft_cf_betas[true_label_max==1].flatten(), color='g', alpha = 0.5)
    #plt.bar(index2, wft_cf_betas[true_label_max==0].flatten(), color='b', alpha = 0.5)
    #plt.xlabel("Male and Females beta weights - wFastText-cf")
    #plt.show()
    
    

