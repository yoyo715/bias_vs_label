

# Method to get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Enter trial number')
    parser.add_argument('-r', "--run", action='store', help="trial number", required=True)
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

    NUM_RUNS = 10        # number of test runs
    SUBSET_VAL = 10000   # number of subset instances for self reported dataset
    
    BATCHSIZE = 100      # number of instances in each batch
    
    dictionary = create_dictionary(WORDGRAMS, MINCOUNT, BUCKET, SUBSET_VAL, run)
    
    