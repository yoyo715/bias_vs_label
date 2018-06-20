
from dictionary import Dictionary
from args import Args
import argparse


# Method to get arguments user inputs
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ngrams', action='store', help="Input number of desired ngrams", required=True)
    
    parser.add_argument('--mincount', action='store',
        help=" float in range [0.0, 1.0] or int, default=1 When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.",  required=True)

    parser.add_argument('--maxcount', action='store',
        help="float in range [0.0, 1.0] or int, default=1.0 When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts.", required=True)
    args = vars(parser.parse_args())

    return args


def main():
    args = Args(get_args())

    train = open('../cleaned_train_withstopwords.txt', 'r')
    dictionary = Dictionary(train, args)


if __name__ == '__main__':
	main()
