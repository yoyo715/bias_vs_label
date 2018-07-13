# subset.py

from dictionary import Dictionary


train = open('../cleaned_train_withstopwords_FULL2.txt', 'r').readlines() 
test = open('../cleaned_test_withstopwords_FULL.txt', 'r').readlines()

WORDGRAMS=2
MINCOUNT=2

#dictionary = Dictionary(train, WORDGRAMS, MINCOUNT)
#train_inst = dictionary.get_bagngram()


NUMTRAIN_INST = 1000
NUMTEST_INST = 200


train_subset = open('../cleaned_train_subset.txt', 'w')
test_subset = open('../cleaned_test_subset.txt', 'w')


i = 0
print(len(train))
for inst in train:
    print(i)
    train_subset.write(inst)
    i += 1
    
    if i == NUMTRAIN_INST:
        break
    
print()
i = 0
print(len(test))
for inst2 in test:
    print(i)
    test_subset.write(inst2)
    i += 1
    
    if i == NUMTEST_INST:
        break