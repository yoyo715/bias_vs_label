# graph_time.py

"""
   
"""

import os
from matplotlib import pyplot as plt
import numpy as np



directory_newverison = '../slurm_scripts/newkmm/normed_lin/outfiles/'

directory = '../slurm_scripts/oldkmm/lin/outfiles/'
numfiles = len(os.listdir(directory))   # number of plots needed 

epochs = [l for l in range(20)]
fig, axs = plt.subplots(4, int(numfiles/4)+1, sharex=True, sharey=True)

axs = axs.ravel()
i = 0


def open_file(fname):
    for filename in os.listdir(directory_newverison):
        old = fname.split('_')
        new = filename.split('_')
        if (old[2] in new) and (old[3] in new) :
            with open(directory_newverison+filename,"r") as f:
                time = []
                full_time = 0
                
                content = f.readlines()
                for line in content:
                    if "Beta took" in line:
                        beta_time = float(line.split()[2])
                        full_time = full_time + beta_time
                            
                    if "~~~~Epoch took" in line:
                        val = float(line.split()[-2])
                        full_time = full_time + val
                        
                        time.append(full_time)
            return time
    

for filename in os.listdir(directory):
    with open(directory+filename,"r") as f:
        time = []
        full_time = 0
        
        new_time = open_file(filename)
        
        content = f.readlines()
        for line in content:
            if "Beta took" in line:
                beta_time = float(line.split()[2])
                full_time = full_time + beta_time
                    
            if "~~~~Epoch took" in line:
                val = float(line.split()[-2])
                full_time = full_time + val
                
                time.append(full_time)
                
        axs[i].plot(epochs, time, 'm', label="time")
        axs[i].plot(epochs, new_time, 'g', label="new_time")

        axs[i].set_title(filename)
        axs[i].set_ylabel('Time (min.)')
        axs[i].set_xlabel('epoch')
        axs[i].legend(loc='upper left')
        
        i += 1


plt.show()
    
                
        
