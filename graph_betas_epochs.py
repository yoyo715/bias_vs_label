# graph_betas_epochs.py

"""
    
"""

import os, math
from matplotlib import pyplot as plt
import numpy as np

directory = '../slurm_scripts/newkmm/not_normed_lin/rbf/'
directory_old = '../slurm_scripts/oldkmm/rbf/outfiles/'
numfiles = len(os.listdir(directory))   # number of plots needed 

epochs = [l for l in range(20)]
fig, axs = plt.subplots(4, int(numfiles/4)+1, sharex=True, sharey=True)

axs = axs.ravel()
i = 0

def get_mean(line):
    for val in line.split():
        if 'mean' in val:
            m = val.split('[')
            m2 = m[1].split(']')
    return float(m2[0])


def get_std(line):
    for val in line.split():
        if 'variance' in val:
            v = val.split('[')
            v2 = v[1].split(']')
    return math.sqrt(float(v2[0]))


def get_old_betastats(fname):
    for filename in os.listdir(directory_old):
        m = 0
        s = 0
        
        new = fname.split('_')
        old = filename.split('_')
        
        if (old[2] in new) and (old[3] in new) :
            with open(directory_old+filename,"r") as f:
                
                content = f.readlines()
                for line in content:
                    if "DescribeResult" in line:
                        m = get_mean(line)
                        s = get_std(line)
                        
            means = [m] * 20 
            stds = [s] * 20
            return np.array(means), np.array(stds)
            
            

for filename in os.listdir(directory):
    with open(directory+filename,"r") as f:
        means = []
        stds = []
        
        content = f.readlines()
        for line in content:
            if "DescribeResult" in line:
                m = get_mean(line)
                means.append(m)
                s = get_std(line)
                stds.append(s)

        means = np.array(means)
        stds = np.array(stds)
        
        old_m, old_s = get_old_betastats(filename)
    
        axs[i].errorbar(epochs, means, stds, linestyle='None', marker='^', color='r')
        axs[i].errorbar(epochs, old_m, old_s, linestyle='None', marker='*', color='g')
        
        axs[i].set_title(filename[20:-1])
        axs[i].set_ylabel('beta mean')
        axs[i].set_xlabel('epoch')
        
        i += 1


plt.suptitle('Both Versions Betas Means + Stds (rbf kernel)')
plt.show()
