# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np

ft_dir = '../slurm_scripts/fasttext/REAL2/'
new_wft_dir = '../slurm_scripts/new_wft/REAL/'
new_wft_cf_dir = '../slurm_scripts/new_wft_cf/REAL/'
new_wft_ck_dir = '../slurm_scripts/new_wft_ck/REAL/'

old_wft_dir = '../slurm_scripts/old_wft/REAL/'
old_wft_cf_dir = '../slurm_scripts/old_wft_cf/REAL/'
old_wft_ck_dir = '../slurm_scripts/old_wft_ck/REAL/'


#dirs = [ft_dir, new_wft_dir, new_wft_cf_dir, new_wft_ck_dir, old_wft_dir, old_wft_cf_dir, old_wft_ck_dir]
#dirs = [ft_dir, new_wft_dir, old_wft_dir, new_wft_ck_dir]
#dirs = [ft_dir, new_wft_dir, new_wft_ck_dir]
dirs = [ft_dir]


epochs = [l for l in range(20)]


def get_avg(d):
    all_train = []
    all_test = []
    all_man = []
    
    for filename in os.listdir(d):
        with open(d+filename,"r") as f:
            train = []
            test = []
            man = []
            
            content = f.readlines()
            for line in content:
                if "SVAL Classification" in line:
                    val = float(line.split()[-1])
                    train.append(val)
                if "STEST Classification" in line:
                    val = float(line.split()[-1])
                    test.append(val)
                if "RVAL Classification" in line:
                    val = float(line.split()[-1])
                    man.append(val)
                    
            train = np.array(train)
            test = np.array(test)
            man = np.array(man)
            
        all_train.append(train)
        all_test.append(test)
        all_man.append(man)
        
    all_train = np.array(all_train)
    all_test = np.array(all_test)
    all_man = np.array(all_man)
    
    return np.mean(all_train, axis=0), np.mean(all_test, axis=0), np.mean(all_man, axis=0)
        

i = 0
markers = ['*', '^', 's']
 
for d in dirs:
    train, test, man = get_avg(d)
    
    if '_wft/' in d:
        m = markers[0]
        lab = 'wft'
    elif '_wft_cf/' in d:
        m = markers[1]
        lab = 'wft-cf'
    elif '_wft_ck/' in d:
        m = markers[2]
        lab = 'wft-ck'
   
    if "old" in d:
        #plt.plot(epochs, train, 'm', label="train", linestyle='--', marker=m)
        #plt.plot(epochs, test, 'c',linestyle='--', marker=m, label = 'old '+lab, markersize=12)
        #plt.plot(epochs, man, 'g', linestyle='--', marker=m, label = 'old '+lab, markersize=12)
        pass
    elif "new" in d:
        #plt.plot(epochs, train[1:], 'm', label= "SL-Train Set "+lab,  marker=m,)
        #plt.plot(epochs, test, 'c', marker=m, label = 'new '+ lab, markersize=12)
        #plt.plot(epochs, man, 'g', marker=m, label = 'new '+ lab, markersize=12)
        
        #plt.plot(epochs, test[1:], 'c', label = "SL-Test Set "+lab, marker=m,)
        plt.plot(epochs, man[1:], 'g', label = "Ran-Test Set "+lab, marker=m)
    
    else:
        #plt.plot(epochs, train, 'm', label="FT SL-Train Set", marker=m)
        #plt.plot(epochs, train[1:], 'm', label="FT SL-Train Set", linestyle='--',)
        #plt.plot(epochs, test, 'c', linestyle='--', label="original")
        plt.plot(epochs, test[1:], 'c', label="FT SL-Test Set", linestyle='--',)
        plt.plot(epochs, man[1:], 'g', label = 'Ran-Test Set ft', linestyle='--',)
    i += 1

        
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylabel('Classification Error', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.legend(loc='upper right', prop={'size': 12})
#plt.title('Classification Error Comparision')
#plt.title("Classification Error of FastText, wFastText, wFastText-ck", fontsize=18)
plt.title("Classification Error of FastText", fontsize=18)
plt.show()
    
                
        
