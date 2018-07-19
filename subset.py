# subset.py

import random 

# used to combine the two datasets
#train = open('../cleaned_train_withstopwords_FULL2.txt', 'r').readlines() 
#test = open('../cleaned_test_withstopwords_FULL.txt', 'r').readlines()

#random.shuffle(train) 
#random.shuffle(test)
#combined = open('../cleaned_combined_FULL.txt', 'w')

#NUMTRAIN_INST = 500
#NUMTEST_INST = 300

NUMINST = 500


#train_subset = open('../cleaned_train_subset.txt', 'w')
#test_subset = open('../cleaned_test_subset.txt', 'w')

#subset = open('../cleaned_subset.txt', 'w')



i = 0
print("Full Train len: ", len(train), " Full Test len: ", len(test))

for inst in train:
    combined.write(inst)
    
    
for inst in test:
    combined.write(inst)




#for inst in train:
    #print(i)
    #train_subset.write(inst)
    #i += 1
    
    #if i == NUMTRAIN_INST:
        #break
    
#print()
#i = 0
#for inst2 in test:
    #print(i)
    #test_subset.write(inst2)
    #i += 1
    
    #if i == NUMTEST_INST:
        #break