

# Method to get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Enter trial number')
    parser.add_argument('-r', "--run", action='store', help="trial number", required=True)
    args = vars(parser.parse_args())

    return args


# model_version: 'original' or 'kmm;
def create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model_version):
    print("starting dictionary creation") 

    # dictionary must be recreated each run to get different subsample each time
    start = time.time()
    dictionary = Dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model=model_version)
    end = time.time()
    print("dictionary took ", (end - start)/60.0, " time to create.")
    
    return dictionary


def main():
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
    KMMLR = 0.018   #0.014         #0.015 pretty good

    KERN = 'lin'         # lin or rbf or poly
    NUM_RUNS = 10        # number of test runs
    SUBSET_VAL = 10000   # number of subset instances for self reported dataset
    LIN_C = 1.0          # hyperparameter for linear kernel
    
    BATCHSIZE = 100      # number of instances in each batch
    
    #model = 'kmm'
    model = 'original'   # 'kmm' for kmm implementation
    
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, KERN, SUBSET_VAL, LIN_C, run, model)
    
    
