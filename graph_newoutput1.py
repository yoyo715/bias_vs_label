# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np

directory = '../slurm_scripts/oldkmm/scripts/rbf/outfiles/'
numfiles = len(os.listdir(directory))   # number of plots needed 

epochs = [l for l in range(20)]
fig, axs = plt.subplots(4, int(numfiles/4)+1, sharex=True, sharey=True)

axs = axs.ravel()
i = 0

for filename in os.listdir(directory):
    with open(directory+filename,"r") as f:
        train = []
        test = []
        man = []
        
        content = f.readlines()
        for line in content:
            if "KMMTRAIN" in line:
                val = float(line.split()[-1])
                train.append(val)
            if "KMMTEST" in line:
                val = float(line.split()[-1])
                test.append(val)
            if "KMMMANUAL" in line:
                val = float(line.split()[-1])
                man.append(val)
                
        train = np.array(train)
        test = np.array(test)
        man = np.array(man)
        
        #print(filename)
        #print(man)
        #print()
    
        axs[i].plot(epochs, train, 'm', label="train")
        axs[i].plot(epochs, test, 'c', label="test")
        axs[i].plot(epochs, man, 'g', label="manual")
        axs[i].set_title(filename)
        
        axs[i].set_ylabel('classification error')
        axs[i].set_xlabel('epoch')
        axs[i].legend(loc='upper left')
        #axs[i].set_ylim(0.69310, 0.69316)
        
        i += 1


plt.show()
    
                
        
