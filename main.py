
from dictionary import Dictionary
from args import Args
import argparse


# Method to get arguments user inputs
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ngrams', action='store', help="Input number of desired ngrams", required=True)
    parser.add_argument('--mincount', action='store', help="Input min number of words", required=True)
    args = vars(parser.parse_args())
    return args


def main():
    args = Args(get_args())
    
    

    train = open('./cleaned_train.txt', 'r')
    dictionary = Dictionary(train, args)


if __name__ == '__main__':
	main()
