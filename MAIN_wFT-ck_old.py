# MAIN_wFT-ck_old.py


"""


"""

from CLASS_dictionary2 import Dictionary
from CLASS_wfasttext_ck import wFastText_ck

import argparse, time
import numpy as np
import pandas as pd


# Method to get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Enter trial number and Learning Rate')
    parser.add_argument('-r', "--run", action='store', help="trial number", required=True)
    
    parser.add_argument('-l', "--learning_rate", action='store', help="KMM Learning Rate", required=True)
    
    parser.add_argument('-b', "--b_val", action='store', help="KMM B value", required=True)

    parser.add_argument('-k', '--kernel', action='store', help='KMM Kernel', required=True)
    
    parser.add_argument('-f', '--rfemale', action='store', help='R Female val', required=True)
    
    parser.add_argument('-m', '--rmale', action='store', help='R Male val', required=True)
    args = vars(parser.parse_args())

    return args


# model_version: 'original' or 'kmm;
def create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, run):
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, run)
    end = time.time()
    print("dictionary took ", (end - start)/60.0, " time to create.")
    
    return dictionary


def main():
    print("STARTING")
    
    args = get_args()
    print(args)
    
    run = args['run']
    
    # args from Simple Queries paper
    DIM=30
    WORDGRAMS=2
    MINCOUNT=2  #2 
    MINN=3
    MAXN=3
    BUCKET=1000000

    # adjust these
    EPOCH=20
    KMMLR = float(args['learning_rate'])
    B = float(args['b_val'])
    BATCHSIZE = 10      # number of instances in each batch
    KERNEL = args['kernel']
    
    r_female = float(args['rfemale'])
    r_male = float(args['rmale'])
    
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, run)
    
    wfasttext_ck = wFastText_ck(dictionary, KMMLR, DIM, EPOCH, B, BATCHSIZE, KERNEL, r_female, r_male)
    wfasttext_ck.train_batch()
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
