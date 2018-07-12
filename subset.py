# subset.py

from dictionary import Dictionary


train = open('../cleaned_train_withstopwords_FULL2.txt', 'r')
test = open('../cleaned_test_withstopwords_FULL.txt', 'r')

WORDGRAMS=2
MINCOUNT=2

dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
input_ = dictionary.get_bagngram()


NUMTRAIN_INST = 10
NUMTEST_INST = 5


train_subset = open('../cleaned_train_subset.txt', 'w')
test_subset = open('../cleaned_test_subset.txt', 'w')


i = 0
for inst in train:
    train_subset.write(inst)
    i += 1
    
    if i == NUMTRAIN_INST:
        break
    
