import os, random
import numpy as np

#NUMINST = 2000
#max_ = 9999

#numfiles = 20

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
            
            
#NUMINST = 1274
#max_ = 6370
        
#l = random.sample(range(0, max_), NUMINST)
#fname = 'R_val_RACE.txt'
    
#with open('./indices_Rval_RACE/'+fname, 'w') as f:
    #for item in l:
        #f.write("%s\n" % item)

            
#for i in range(numfiles):
    #l = random.sample(range(0, max_), NUMINST)
    #fname = 'S_val_RACE_'+str(i)+'.txt'
    
    #with open('./indices_Sval_RACE/'+fname, 'w') as f:
        #for item in l:
            #f.write("%s\n" % item)


# Sensitivity Analysis indices

ran_gender_dir = './sensitivity_indices/random_gender/'
ran_race_dir = './sensitivity_indices/random_race/'
self_gender_dir = './sensitivity_indices/self_gender/'
self_race_dir = './sensitivity_indices/self_race/'

numfiles = 20
ran_gender_subset_sizes = np.linspace(10, 8790, numfiles).tolist()
ran_gender_subset_sizes = [round(x) for x in ran_gender_subset_sizes]

ran_race_subset_sizes = np.linspace(10, 399, numfiles).tolist()
ran_race_subset_sizes = [round(x) for x in ran_race_subset_sizes]

self_gender_subset_sizes = np.linspace(10, 10000, numfiles).tolist()
self_gender_subset_sizes = [round(x) for x in self_gender_subset_sizes]

self_race_subset_sizes = np.linspace(10, 5090, numfiles).tolist()
self_race_subset_sizes = [round(x) for x in self_race_subset_sizes]


#print(len(ran_gender_subset_sizes))
#print(len(ran_race_subset_sizes))
#print(len(self_gender_subset_sizes))
#print(len(self_race_subset_sizes))


# create the directories
#for i in ran_gender_subset_sizes:
    #os.mkdir(ran_gender_dir+str(i))
    
#for i in ran_race_subset_sizes:
    #os.mkdir(ran_race_dir+str(i))
    
#for i in self_gender_subset_sizes:
    #os.mkdir(self_gender_dir+str(i))
    
#for i in self_race_subset_sizes:
    #os.mkdir(self_race_dir+str(i))


trials = 10

# ran_gender_subset_sizes
#for size in ran_gender_subset_sizes:
    #for i in range(trials):
        #l = random.sample(range(0, 8790), size)
        #fname = 'ran_gender_trial'+str(i)+'_size'+str(size)+'.txt'
        
        #with open(ran_gender_dir+str(size)+'/'+fname, 'w') as f:
            #for item in l:
                #f.write("%s\n" % item)
                
# ran_race_subset_sizes
#for size in ran_race_subset_sizes:
    #for i in range(trials):
        #l = random.sample(range(0, 399), size)
        #fname = 'ran_race_trial'+str(i)+'_size'+str(size)+'.txt'
        
        #with open(ran_race_dir+str(size)+'/'+fname, 'w') as f:
            #for item in l:
                #f.write("%s\n" % item)

# self_gender_subset_sizes
#for size in self_gender_subset_sizes:
    #for i in range(trials):
        #l = random.sample(range(0, 39120), size)
        #fname = 'self_gender_trial'+str(i)+'_size'+str(size)+'.txt'
        
        #with open(self_gender_dir+str(size)+'/'+fname, 'w') as f:
            #for item in l:
                #f.write("%s\n" % item)
                
# self_race_subset_sizes
#for size in self_race_subset_sizes:
    #for i in range(trials):
        #l = random.sample(range(0, 5090), size)
        #fname = 'self_race_trial'+str(i)+'_size'+str(size)+'.txt'
        
        #with open(self_race_dir+str(size)+'/'+fname, 'w') as f:
            #for item in l:
                #f.write("%s\n" % item)







