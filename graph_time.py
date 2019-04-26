# graph_time.py

"""
   
"""

import os
from matplotlib import pyplot as plt
import numpy as np

ft_dir = '../slurm_scripts/fasttext/REAL/'
new_wft_dir = '../slurm_scripts/new_wft/REAL/'
new_wft_cf_dir = '../slurm_scripts/new_wft_cf/REAL/'
new_wft_ck_dir = '../slurm_scripts/new_wft_ck/REAL/'

old_wft_dir = '../slurm_scripts/old_wft/REAL/'
old_wft_cf_dir = '../slurm_scripts/old_wft_cf/REAL/'
old_wft_ck_dir = '../slurm_scripts/old_wft_ck/REAL/'


dirs = [ft_dir, new_wft_dir, new_wft_cf_dir, new_wft_ck_dir, old_wft_dir, old_wft_cf_dir, old_wft_ck_dir]
epochs = [l for l in range(20)]


def get_avg_per_epoch(d):
    epoch_times = []
    
    for filename in os.listdir(d):
        with open(d+filename,"r") as f:
            times = []
            
            content = f.readlines()
            for line in content:
                if "~~~~Epoch took" in line:
                    val = float(line.split()[2])
                    times.append(val)
            
            times = np.array(times)
            
        epoch_times.append(times)

    
    return np.mean(epoch_times, axis=0)
        

def create_time_per_epoch():
    i = 0
    markers = ['*', '^', 's']
    
    for d in dirs:
        time = get_avg_per_epoch(d)
        
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
            plt.plot(epochs, time, 'g', linestyle='--', marker=m, label = 'old '+lab, markersize=12)
        elif "new" in d:
            plt.plot(epochs, time, 'g', marker=m, label = 'new '+ lab, markersize=12)
        else:
            plt.plot(epochs, time, 'r', label = 'original')
        i += 1

            
    plt.ylabel('Avg. Training Time (min.)')
    plt.xlabel('Epoch')
    #plt.legend(loc='upper left', prop={'size': 18})
    plt.title('Classification Error Comparision')
    plt.show()
        
    
def get_cumulative_time(d):
    epoch_times = []
    
    for filename in os.listdir(d):
        with open(d+filename,"r") as f:
            times = []
            
            content = f.readlines()
            for line in content:
                if "~~~~Epoch took" in line:
                    val = float(line.split()[2])
                    try:
                        val = val + times[-1]
                    except:
                        val = val
                    times.append(val)
            
            times = np.array(times)
            
        epoch_times.append(times)

    
    return np.mean(epoch_times, axis=0)
    
    
def create_cumulative_time():   
    i = 0
    markers = ['*', '^', 's']
    
    for d in dirs:
        time = get_cumulative_time(d)
        
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
            plt.plot(epochs, time, 'g', linestyle='--', marker=m, label = 'old '+lab, markersize=12)
        elif "new" in d:
            plt.plot(epochs, time, 'g', marker=m, label = 'new '+ lab, markersize=12)
        else:
            plt.plot(epochs, time, 'r', label = 'original')
        i += 1

            
    plt.ylabel('Cumulative Training Time (min.)')
    plt.xlabel('Epoch')
    #plt.legend(loc='upper left', prop={'size': 18})
    plt.title('Classification Error Comparision')
    plt.show()
    
    
def main():
    #create_time_per_epoch()
    create_cumulative_time()
    
                
if __name__ == '__main__':
    main()
        
