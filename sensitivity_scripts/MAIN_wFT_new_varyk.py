# MAIN_wFT_new.py


"""


"""

from CLASS_dictionary2 import Dictionary
from CLASS_wfasttext_new import wFastText_new

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
    
    parser.add_argument('-d', '--latent_dim', action='store', help='Dimension', required=True)
    
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
    DIM=args['latent_dim']
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
    
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, run)
    
    wfasttext = wFastText_new(dictionary, KMMLR, DIM, EPOCH, B, BATCHSIZE, KERNEL)
    wfasttext.train_batch()
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
