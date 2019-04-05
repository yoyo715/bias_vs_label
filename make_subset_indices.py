import os, random

NUMINST = 2000
max_ = 9999

numfiles = 20

#for i in range(numfiles):
    #l = random.sample(range(0, max_), NUMINST)
    #fname = 'sval_TRAIN_'+str(i)+'.txt'
    
    #with open('./indices_Sval/'+fname, 'w') as f:
        #for item in l:
            #f.write("%s\n" % item)
            
            
for i in range(numfiles):
    l = random.sample(range(0, max_), NUMINST)
    fname = 'R_val_'+str(i)+'.txt'
    
    with open('./indices_Rval/'+fname, 'w') as f:
        for item in l:
            f.write("%s\n" % item)
            
            
            
#l = random.sample(range(0, max_), NUMINST)
#fname = 'R_val.txt'
    
#with open('./indices_Rval/'+fname, 'w') as f:
    #for item in l:
        #f.write("%s\n" % item)
