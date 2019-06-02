# graph_newoutput1.py

"""
    Script to graph the results after applying the new integrated KMM method with
    various learning rates.
"""

import os
from matplotlib import pyplot as plt
import numpy as np


def find_bestval(directory):
    best_manual_val = 999
    for filename in os.listdir(directory):
        with open(directory+filename,"r") as f:
            content = f.readlines()
            for line in content:
                if "RVAL Classification" in line:
                    val = float(line.split()[-1])
                    print(val)
                    if val < best_manual_val:
                        best_manual_val = val
    return best_manual_val






def get_ft_averaged(lr):
    directory = '../slurm_scripts/RACE/fasttext/hypers/'
    
    print("********* LR: ",lr)
    train_all = []
    test_all = []
    man_all = []
    for filename in os.listdir(directory):
        if '_LR'+str(lr)+'_' in filename:
            with open(directory+filename,"r") as f:
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
                train_all.append(train)
                test_all.append(test)
                man_all.append(man)
                        
    train_all = np.array(train_all)
    test_all = np.array(test_all)
    man_all = np.array(man_all)
    return np.mean(train_all, axis=0), np.mean(test_all, axis=0), np.mean(man_all, axis=0)


def plot_ft():
    numfiles = 10

    epochs = [l for l in range(21)]
    fig, axs = plt.subplots(3, int(numfiles/3), sharex=True, sharey=True)
    axs = axs.ravel()

    lrs = [0.01, 0.015, 0.02, 0.04, 0.08, 0.1, 0.15]
    i = 0
    for lr in lrs:
        train, test, man = get_ft_averaged(lr)
        
        axs[i].plot(epochs, train, 'm', label="train")
        axs[i].plot(epochs, test, 'c', label="test")
        axs[i].plot(epochs, man, 'g', label="manual")
    
        #axs[i].axhline(y = best_value, linewidth=2, color = 'red')
        axs[i].set_title(lr)
        
        axs[i].set_ylabel('classification error')
        axs[i].set_xlabel('epoch')
        #axs[i].legend(loc='upper left')
        
        i += 1


    plt.suptitle('race ft')
    plt.show()


def get_wft_averaged(lr, b):
    directory = '../slurm_scripts/RACE/new_wft/hypers/'
    
    print("********* LR: ",lr, " B: ", b)
    train_all = []
    test_all = []
    man_all = []
    for filename in os.listdir(directory):
        if ('_LR'+str(lr)+'_' in filename) and ('_B'+str(b)+'_' in filename) :
            with open(directory+filename,"r") as f:
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
                train_all.append(train)
                test_all.append(test)
                man_all.append(man)
                        
    train_all = np.array(train_all)
    test_all = np.array(test_all)
    man_all = np.array(man_all)
    return np.mean(train_all, axis=0), np.mean(test_all, axis=0), np.mean(man_all, axis=0)


def plot_wft():
    #directory = '../slurm_scripts/RACE/new_wft/hypers/'
    #best_value = find_bestval(directory)
    #print("Best manual value = ", best_value)

    numfiles = 36

    epochs = [l for l in range(21)]
    fig, axs = plt.subplots(3, int(numfiles/3), sharex=True, sharey=True)
    axs = axs.ravel()
    
    lrs = [0.01, 0.02, 0.04, 0.08, 0.1, 0.15]
    B = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    i = 0
    for lr in lrs:
        for b in B:
            train, test, man = get_wft_averaged(lr, b)
            
            axs[i].plot(epochs, train, 'm', label="train")
            axs[i].plot(epochs, test, 'c', label="test")
            axs[i].plot(epochs, man, 'g', label="manual")
        
            #axs[i].axhline(y = best_value, linewidth=2, color = 'red')
            axs[i].axhline(y = 0.45, linewidth=2, color = 'red')
            axs[i].set_title(str(lr)+' '+str(b))
            
            axs[i].set_ylabel('classification error')
            axs[i].set_xlabel('epoch')
            #axs[i].legend(loc='upper left')
            
            i += 1


    plt.suptitle('race ft')
    plt.show()
    
    
    
def get_wft_cf_averaged(r0, r1):
    directory = '../slurm_scripts/RACE/new_wft_cf/hypers/'
    
    print("********* R0: ", r0, " R1: ", r1)
    train_all = []
    test_all = []
    man_all = []
    for filename in os.listdir(directory):
        if ('_R0'+str(r0)+'_' in filename) and ('_R1'+str(r1)+'_' in filename) :
            with open(directory+filename,"r") as f:
                print(filename)
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
                train_all.append(train)
                test_all.append(test)
                man_all.append(man)
                        
    train_all = np.array(train_all)
    test_all = np.array(test_all)
    man_all = np.array(man_all)
    return np.mean(train_all, axis=0), np.mean(test_all, axis=0), np.mean(man_all, axis=0)


def plot_wft_cf():
    numfiles = 9

    epochs = [l for l in range(21)]
    fig, axs = plt.subplots(3, int(numfiles/3), sharex=True, sharey=True)
    axs = axs.ravel()
    
    R0 = [2.0, 4.0, 5.0]
    R1 = [4.0, 8.0, 10.0]
    
    i = 0
    for r0 in R0:
        r1 = R1[i]
        
        train, test, man = get_wft_cf_averaged(r0, r1)
        
        axs[i].plot(epochs, train, 'm', label="train")
        axs[i].plot(epochs, test, 'c', label="test")
        axs[i].plot(epochs, man, 'g', label="manual")
    
        #axs[i].axhline(y = best_value, linewidth=2, color = 'red')
        axs[i].axhline(y = 0.45, linewidth=2, color = 'red')
        axs[i].set_title(str(r0)+' '+str(r1))
        
        axs[i].set_ylabel('classification error')
        axs[i].set_xlabel('epoch')
        #axs[i].legend(loc='upper left')
        
        i += 1


    plt.suptitle('race ft')
    plt.show()


if __name__ == '__main__':
    plot_ft()
    #plot_wft()
    #plot_wft_cf()
    
                
        
