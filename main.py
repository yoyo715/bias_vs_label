#main.py

from dictionary import Dictionary
import numpy as np


# calculates softmax value
def softmax(x, A, B, nlabels):
    product = np.dot(B.T,A)
    product2 = np.dot(product, x)
 
    print(product2.shape)


# calculates log-likelihood
def log_likelihood():
    return 1


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.1
    WORDGRAMS=2
    MINCOUNT=2
    MINN=3
    MAXN=3
    BUCKET=1 #000000
    EPOCH=20

    train = open('../cleaned_train_withstopwords.txt', 'r')
    dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
    input_ = dictionary.get_bagngram()
    labels = dictionary.get_labels()
    nwords = dictionary.get_nwords()
    nlabels = len(set(labels))
    N = dictionary.get_ninstances()


    ##### instantiations #######################################

    

    # A
    A_n = DIM
    A_m = nwords + BUCKET
    uniform_val = 1.0 / DIM
    A = np.random.uniform(-uniform_val, uniform_val, (A_n,A_m))

    # B
    B_n = DIM
    B_m = nlabels
    B = np.zeros((B_n, B_m))


    #### train ################################################

    # loop through each instance for SGD
    for x in input_:
        #print(instance.shape)
        softmax(x, A, B, nlabels)
    



if __name__ == '__main__':
	main()






