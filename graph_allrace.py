# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np

#ft_dir = '../slurm_scripts/fasttext/REAL2/'
#new_wft_dir = '../slurm_scripts/new_wft/REAL/'
#new_wft_cf_dir = '../slurm_scripts/new_wft_cf/REAL/'
#new_wft_ck_dir = '../slurm_scripts/new_wft_ck/REAL/'

#old_wft_dir = '../slurm_scripts/old_wft/REAL/'
#old_wft_cf_dir = '../slurm_scripts/old_wft_cf/REAL/'
#old_wft_ck_dir = '../slurm_scripts/old_wft_ck/REAL/'

ft_dir = '../slurm_scripts/RACE/fasttext/real_lr0.15/'
ft2_dir = '../slurm_scripts/RACE/fasttext/real_lr0.015/'
wft_dir = '../slurm_scripts/RACE/new_wft/real/'


#dirs = [ft_dir, new_wft_dir, new_wft_cf_dir, new_wft_ck_dir, old_wft_dir, old_wft_cf_dir, old_wft_ck_dir]
#dirs = [ft_dir, new_wft_dir, old_wft_dir, new_wft_ck_dir]
#dirs = [ft_dir, new_wft_dir, new_wft_ck_dir]
dirs = [ft2_dir, wft_dir]


epochs = [l for l in range(20)]


def get_avg_validation(d):
    all_strain = []
    all_sval = []
    all_stest = []
    all_rval = []
    all_rtest = []
    
    for filename in os.listdir(d):
        with open(d+filename,"r") as f:
            strain = []
            sval = []
            stest = []
            rval = []
            rtest = []
            
            content = f.readlines()
            for line in content:
                if "STRAIN Classification" in line:
                    val = float(line.split()[-1])
                    strain.append(val)
                if "SVAL Classification" in line:
                    val = float(line.split()[-1])
                    sval.append(val)
                if "STEST Classification" in line:
                    val = float(line.split()[-1])
                    stest.append(val)
                if "RVAL Classification" in line:
                    val = float(line.split()[-1])
                    rval.append(val)
                if "RTEST Classification" in line:
                    val = float(line.split()[-1])
                    rtest.append(val)
                    
            strain = np.array(strain)
            sval = np.array(sval)
            stest = np.array(stest)
            rval = np.array(rval)
            rtest = np.array(rtest)
            
        all_strain.append(strain)
        all_sval.append(sval)
        all_stest.append(stest)
        all_rval.append(rval)
        all_rtest.append(rtest)
        
    all_strain = np.array(all_strain)
    all_sval = np.array(all_sval)
    all_stest = np.array(all_stest)
    all_rval = np.array(all_rval)
    all_rtest = np.array(all_rtest)
    
    return np.mean(all_strain, axis=0), np.mean(all_sval, axis=0), np.mean(all_stest, axis=0), \
           np.mean(all_rval, axis=0), np.mean(all_rtest, axis=0), \
           np.std(all_strain, axis=0), np.std(all_sval, axis=0), np.std(all_stest, axis=0), \
           np.std(all_rval, axis=0), np.std(all_rtest, axis=0)
       

def get_stats():
    for d in dirs:
        print(d)
        strain, sval, stest, rval, rtest, strain_std, sval_std, stest_std, rval_std, rtest_std = get_avg_validation(d)
        
        print("Validation: ")
        print("Strain: ", strain[-1], strain_std[-1])
        print("Sval: ", sval[-1], sval_std[-1])
        print("Stest: ", stest[-1], stest_std[-1])
        print("Rval: ", rval[-1], rval_std[-1])
        print("Rtest: ", rtest[-1], rtest_std[-1])
        print()
        
        print()



def plot():
    i = 0
    markers = ['*', '^', 's']
    
    for d in dirs:
        strain, sval, stest, rval, rtest, strain_std, sval_std, stest_std, rval_std, rtest_std = get_avg_validation(d)
        
        if '_wft/' in d:
            m = markers[0]
            #lab = 'wft'
            lab='RoFastText'

        if "_wft" in d: # wft
            #plt.plot(epochs, sval[1:], 'm', label="sval "+lab, marker=m)
            #plt.plot(epochs, stest[1:], 'c', label = 'stest '+lab, marker=m)
            #plt.plot(epochs, rval[1:], 'g', label = "rval "+lab, marker=m)
            #plt.plot(epochs, rtest[1:], 'g', label = "Random Set Test "+lab, marker=m)
            pass
        else: # ft
            plt.plot(epochs, sval[1:], 'm', label="Self-Labeled Set Train FastText")
            plt.plot(epochs, stest[1:], 'c', label = 'Self-Labeled Set Test FastText')
            #plt.plot(epochs, rval[1:], 'g', label = "rval FastText")
            plt.plot(epochs, rtest[1:], 'g', label = "Random Set Test FastText", linestyle='--',)
        i += 1

            
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel('Classification Error', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(loc='upper right', prop={'size': 12})
    #plt.title('Classification Error Comparision')
    #plt.title("Classification Error of FastText, wFastText, wFastText-ck", fontsize=18)
    plt.title("Classification Error on Race Datasets", fontsize=18)
    plt.show()
    

if __name__ == '__main__':
    #get_stats()
    plot()
        
