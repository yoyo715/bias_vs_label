# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np

#directory_new = '../slurm_scripts/newkmm/rbf/'
directory_new = '../slurm_scripts/fasttext/ft/'
#directory_new = '../slurm_scripts/old_wft_cf/hypers/'
#directory_new = '../slurm_scripts/old_wft_ck/hypers/'
#directory_new = '../slurm_scripts/new_wft_ck/hypers/'

directory_old = '../slurm_scripts/oldkmm/rbf/'
#numfiles = len(os.listdir(directory_old))   # number of plots needed 
numfiles = 10

epochs = [l for l in range(21)]
fig, axs = plt.subplots(3, int(numfiles/3), sharex=True, sharey=True)

axs = axs.ravel()
i = 0


def find_bestval():
    best_manual_val = 999
    for filename in os.listdir(directory_new):
        with open(directory_new+filename,"r") as f:
            content = f.readlines()
            for line in content:
                if "RVAL Classification" in line:
                    val = float(line.split()[-1])
                    if val < best_manual_val:
                        best_manual_val = val
    return best_manual_val


best_value = find_bestval()
print("Best manual value = ", best_value)


def get_old_stats(fname):
    for filename in os.listdir(directory_old):        
        new = fname.split('_')
        old = filename.split('_')
             
        if (old[2] in new) and (old[3] in new) :
            with open(directory_old+filename,"r") as f:
                
                train = []
                test = []
                man = []
                
                content = f.readlines()
                for line in content:
                    if "KMM_SVAL" in line:
                        val = float(line.split()[-1])
                        train.append(val)
                    if "KMM_STEST" in line:
                        val = float(line.split()[-1])
                        test.append(val)
                    if "KMM_RVAL" in line:
                        val = float(line.split()[-1])
                        man.append(val)
                        
                train = np.array(train)
                test = np.array(test)
                man = np.array(man)
            return train, test, man


for filename in os.listdir(directory_new):
    #if 'B1.0' not in filename and 'LR0.01' not in filename and 'LR0.02' not in filename:
    with open(directory_new+filename,"r") as f:
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
    
        axs[i].plot(epochs, train, 'm', label="train")
        axs[i].plot(epochs, test, 'c', label="test")
        axs[i].plot(epochs, man, 'g', label="manual")
        
        #old_train, old_test, old_man = get_old_stats(filename)
        
        #axs[i].plot(epochs, old_train, 'm', linestyle='dashed', label="old train")
        #axs[i].plot(epochs, old_test, 'c', linestyle='dashed', label="old test")
        #axs[i].plot(epochs, old_man, 'g', linestyle='dashed', label="old manual")
        
        axs[i].axhline(y = best_value, linewidth=2, color = 'red')
        axs[i].set_title(filename[10:-1])
        
        axs[i].set_ylabel('classification error')
        axs[i].set_xlabel('epoch')
        #axs[i].legend(loc='upper left')
        
        i += 1


plt.suptitle('Both Versions (rbf kernel)')
plt.show()
    
                
        
