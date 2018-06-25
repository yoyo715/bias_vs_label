#main.py

from dictionary import Dictionary
import numpy as np
from scipy import sparse


# calculates softmax value
def softmax(x, A, B, nlabels):
    temp = sparse.csr_matrix.dot(A, x.T)
    product2 = np.dot(B.T, temp)
    
    exp = np.exp(product2)
    return exp / nlabels * exp
    

# calculates log-likelihood
def log_likelihood(x, A, B, nlabels):
    return np.log(softmax(x, A, B, nlabels))


# finds the loss
def loss_function(label, loglike):
    return np.dot(label, loglike)


def main():

    # args from Simple Queries paper
    DIM=30
    LR=0.1
    WORDGRAMS=2
    MINCOUNT=2
    MINN=3
    MAXN=3
    #BUCKET=1 #000000
    BUCKET = 0
    EPOCH=20

    train = open('../cleaned_train_withstopwords.txt', 'r')
    dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
    input_ = dictionary.get_bagngram()
    labels = dictionary.get_labels()
    nwords = dictionary.get_nwords()
    nlabels = dictionary.get_nlabels()
    N = dictionary.get_ninstances()

    
    ##### instantiations #######################################

    # A
    A_n = DIM               # rows
    A_m = nwords + BUCKET   # cols
    uniform_val = 1.0 / DIM
    A = np.random.uniform(-uniform_val, uniform_val, (A_n, A_m))

    # B
    B_n = DIM               # rows
    B_m = nlabels           # cols
    B = np.zeros((B_n, B_m))


    #### train ################################################

    #for i in range(EPOCH):
    # loop through each instance for SGD
    loss = 0
    l = 0
    for x in input_:
        loglike = log_likelihood(x, A, B, nlabels)
        loss += loss_function(labels[l], loglike)
        
        l += 1
        
    



if __name__ == '__main__':
	main()






