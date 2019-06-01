# subset.py

import random 


NUMINST = 500

dataset = open('../cleaned_combined_FULL.txt', 'r').readlines()

random.shuffle(dataset)

subset = open('../SUBSET.txt', 'w')


i = 0
for inst in dataset:
    subset.write(inst)
    
    i += 1
    if i == NUMINST:
        print(i)
        break

