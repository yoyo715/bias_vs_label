# run_kmmnew_batch.py

import subprocess, time, sys

           
"""
    TO RUN:
        - python.exe run_bash.py new
        - python.exe run_bash.py old
"""

lr = [0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.15]
B = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

trial_num = 0


def run_new():
    for l in lr:
        for b in B:
            outfilen = 'OUT_kmmnew_B'+str(b)+_'LR'+str(l)+'.txt'
            errorfilen = 'ERR_kmmnew_B'+str(b)+_'LR'+str(l)+'.txt'
            
            cmd = "sbatch new_bash.sh "+outfilen+" "+errorfilen+" "+str(l)+" "+str(b)+" "+str(trial_num)
            rc = subprocess.call(cmd)
            
            print("Ran script")
            time.sleep(5)
            

def run_old():
    for l in lr:
        for b in B:
            outfilen = 'OUT_kmmold_B'+str(b)+_'LR'+str(l)+'.txt'
            errorfilen = 'ERR_kmmold_B'+str(b)+_'LR'+str(l)+'.txt'
    
            cmd = "sbatch old_bash.sh "+outfilen+" "+errorfilen+" "+str(l)+" "+str(b)+" "+str(trial_num)
            rc = subprocess.call(cmd)
            
            print("Ran script")
            time.sleep(5)
            
    
if __name__ == '__main__':
    if sys.argv[1] == 'new':
        run_new()
    elif sys.argv[1] == 'old':
        run_old()
    else:
        print("Error: incorrect arguments")
    
