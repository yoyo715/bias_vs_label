# combine.py

import random 


train = open('../cleaned_train_withstopwords_FULL2.txt', 'r').readlines()
test = open('../cleaned_test_withstopwords_FULL.txt', 'r').readlines()

random.shuffle(train)
random.shuffle(test)

combined = open('../cleaned_combined_FULL.txt', 'w')


for inst in train:
    combined.write(inst)
    
for inst in test:
    combined.write(inst)
