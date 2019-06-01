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
            
            
#for i in range(numfiles):
    #l = random.sample(range(0, max_), NUMINST)
    #fname = 'R_val_'+str(i)+'.txt'
    
    #with open('./indices_Rval/'+fname, 'w') as f:
        #for item in l:
            #f.write("%s\n" % item)
            
            
NUMINST = 1274
max_ = 6370
        
#l = random.sample(range(0, max_), NUMINST)
#fname = 'R_val_RACE.txt'
    
#with open('./indices_Rval_RACE/'+fname, 'w') as f:
    #for item in l:
        #f.write("%s\n" % item)

            
for i in range(numfiles):
    l = random.sample(range(0, max_), NUMINST)
    fname = 'S_val_RACE_'+str(i)+'.txt'
    
    with open('./indices_Sval_RACE/'+fname, 'w') as f:
        for item in l:
            f.write("%s\n" % item)
