# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np


ran_gender_dir = '../slurm_scripts/sensitivity_analysis/wft/random_gender/out/'
ran_gen_ss_dirs = [10, 1858, 2783,  3707,  4631,  5093,  6017,  6942,  7866,  8790,
           1396,  2321,  3245,  4169,  472,   5555,  6479,  7404,  8328,  934]

ran_race_dir = '../slurm_scripts/sensitivity_analysis/wft/random_race/out/'
ran_race_ss_dirs  = [10,  112,  133,  153,  174,  194,  215,  235,  256 , 276,  297,  30,  317,  338,  358,  379,  399,  51,  71,  92]

self_gender_dir = '../slurm_scripts/sensitivity_analysis/wft/self_gender/out/'
self_gender_ss_dir  = [10,     1062,  2113,  3165,  4216,  5268,  5794,  6845,  7897,  8948, 10000,  1587,  2639,  3691,  4742,  536,   6319,  7371,  8423,  9474]

self_race_dir = '../slurm_scripts/sensitivity_analysis/wft/self_race/out/'
self_race_ss_dirs  = [10,    1347,  1882,  2416,  277,   3218,  3753,  4288,  4823,  545, 1079,  1614,  2149,  2684,  2951,  3486,  4021,  4555,  5090,  812]

vary_k_race_dir = '../slurm_scripts/sensitivity_analysis/wft/vary_k_race/out/'
vary_k_race_ss_dirs = [10, 20, 30, 40, 50]

vary_k_gender_dir = '../slurm_scripts/sensitivity_analysis/wft/vary_k_gender/out/'
vary_k_gender_ss_dirs = [10, 20, 30, 40, 50]


###################
ss_dirss = sorted(self_gender_ss_dir, key=int)
dir_ = self_gender_dir
###################


epochs = [l for l in range(20)]


def get_avg(d, ss_vals):
    all_train = []
    all_test = []
    all_man = []
    
    for filename in os.listdir(d):
        if 'size'+str(ss_vals)+'.txt' in filename:
        #if 'DIM'+str(ss_vals)+'.txt' in filename:
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
            #print(all_man[:-1])
        
    all_train = np.array(all_train)
    all_test = np.array(all_test)
    all_man = np.array(all_man)
    
    return np.mean(all_test[:-1]), np.mean(all_man[:-1])
        

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
    full_test = []
    full_man = []
        
    ss_dirs = ss_dirss
    
    for ss_vals in ss_dirs:
        
        test, man = get_avg(dir_, ss_vals)
        print(test, man)
        
        full_test.append(test)
        full_man.append(man)
            
    
    plt.plot(ss_dirs, full_test, 'c', label="Self-Reported Test")
    plt.plot(ss_dirs, full_man, 'g', label = 'Random Test')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel('Classification Error', fontsize=15)
    plt.xlabel('Subset Size', fontsize=14)
    #plt.xlabel('Latent Dimension Size', fontsize=14)
    #plt.title('Classification Error vs. Latent Dimension Size', fontsize=14)
    plt.title('Classification Error vs. Subset Size', fontsize=14)
    plt.legend(loc=(0.08,0.15), prop={'size': 12}),
    plt.show()
    
                
if __name__ == '__main__':
    plot()
    #get_stats()
