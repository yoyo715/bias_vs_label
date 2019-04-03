# graph_bestout.py

"""

"""

import os
from matplotlib import pyplot as plt
import numpy as np


directory_new = '../slurm_scripts/newkmm/not_normed_lin/lin/batch10/'
directory_old = '../slurm_scripts/oldkmm/lin/outfiles/'

epochs = [l for l in range(21)]
i = 0


old_fn = 'OUT_kmmold_B2.0_LR0.04_lin.txt'
new_fn = 'NOTNormed_OUT_kmmnew_B2.0_LR0.04_lin.txt'


def get_err(filename, directory):
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
        
        return train, test, man


old_train, old_test, old_man = get_err(old_fn, directory_old)
new_train, new_test, new_man = get_err(new_fn, directory_new)

plt.plot(epochs, old_train, 'm', label="old train", marker='^', markersize=14)
plt.plot(epochs, old_test, 'c', label="old test", marker='^', markersize=14)
plt.plot(epochs, old_man, 'g', label="old manual", marker='^', markersize=14)


plt.plot(epochs, new_train, 'm', label="new train", marker='+', markersize=14)
plt.plot(epochs, new_test, 'c', label="new test", marker='+', markersize=14)
plt.plot(epochs, new_man, 'g', label="new manual", marker='+', markersize=14)

plt.axhline(y = 0.40, linewidth=2, color = 'red')

plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()



