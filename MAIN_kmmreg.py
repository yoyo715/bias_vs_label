# MAIN_ALL.py


"""


"""

from CLASS_dictionary import Dictionary
from CLASS_wfasttext import wFastText

import argparse, time
import numpy as np
import pandas as pd


# Method to get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Enter trial number and Learning Rate')
    parser.add_argument('-r', "--run", action='store', help="trial number", required=True)
    
    parser.add_argument('-l', "--learning_rate", action='store', help="KMM Learning Rate", required=True)
    
    parser.add_argument('-b', "--b_val", action='store', help="KMM B value", required=True)
    args = vars(parser.parse_args())

    return args


# model_version: 'original' or 'kmm;
def create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, SUBSET_VAL, run):
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, SUBSET_VAL, run)
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
    LR= 0.007       #0.008                 #0.007            # 0.008 good for fasttext
    KMMLR = float(args['learning_rate'])
    B = float(args['b_val'])
    #KMMLR = 0.015   #0.014         #0.015 pretty good
    #KMMLR = 0.0001
    #KMMLR = 0.001

    SUBSET_VAL = 10000   # number of subset instances for self reported dataset
    BATCHSIZE = 50      # number of instances in each batch
    
    KERNEL = 'lin'
    
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, SUBSET_VAL, run)
    
    wfasttext = wFastText(dictionary, KMMLR, DIM, EPOCH, B, BATCHSIZE, KERNEL)
    wfasttext.train()
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
