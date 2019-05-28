#CLASS_dictionary.py

# Dictionary class

"""
    This version of the dictionary creates bag of words with both word ngrams and char ngrams

"""

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import random, re, time, os
from conllu import parse


class Dictionary:
    def __init__(self, ngrams, mincount, bucket, run):
        self.run_number = run
        self.ngrams = ngrams
        self.mincount = mincount
        self.bucket = bucket
        
        TETON = False
        #TETON = True    # WORKING ON TETON OR NOT
        if TETON == True:
            self.raw_file_aa = open('../../TwitterAAE-UD-v1/aa250_gold.conllu', encoding='utf8').read()
            self.raw_file_wh = open('../../TwitterAAE-UD-v1/wh250_gold.conllu', encoding='utf8').read()
        else:
            self.raw_file_aa = open('../../TwitterAAE-UD-v1/aa250_gold.conllu', encoding='utf8').read()
            self.raw_file_wh = open('../../TwitterAAE-UD-v1/wh250_gold.conllu', encoding='utf8').read()
            self.index_dir = './indices_RACE/'  
            self.index_Rval = './indices_Rval_RACE/'
            self.index_Sval = './indices_Sval_RACE/'
            
        raw_aa = self.convert_format(self.raw_file_aa)
        raw_wh = self.convert_format(self.raw_file_wh)
        
        raw_aa_labels = [1.0] * len(raw_aa)  # WARNING double check these values
        raw_wh_labels = [0.0] * len(raw_wh)
        
        
        #print("--------- creating train instances ---------")
        #train_subset = self.split_rand_subset_SFULL()
        #train_instances, train_labels = self.create_instances_and_labels(train_subset)
        #x_strain, x_sval = self.split_Strain_Sval(train_instances)
        #y_strain, y_sval = self.split_Strain_Sval(train_labels)
        #self.n_strain = len(x_strain)
        #self.n_sval = len(x_sval)
        #print("x_strain: ", self.n_strain, " x_sval: ", self.n_sval)
        #print("y_strain: ", len(y_strain), " y_sval: ", len(y_sval))
        #print()
        
        print("---------- creating manual instances ---------")
        manual_instances, y_manual = self.combine_manual_race(raw_aa, raw_wh, raw_aa_labels, raw_wh_labels)
        #manual_instances, y_manual = self.create_instances_and_labels_manset(self.manual_set)
        x_rtest, x_rval = self.split_Rtest_Rval(manual_instances)
        y_rtest, y_rval = self.split_Rtest_Rval(y_manual)
        self.n_rtest = len(x_rtest)
        self.n_rval = len(x_rval)
        print("x_rtest: ", self.n_rtest, " x_rval: ", self.n_rval)
        print("y_rtest: ", len(y_rtest), " y_rval: ", len(y_rval))
        print()
        
        #print("--------- creating testing instances ---------")
        #x_stest, y_stest = self.create_instances_and_labels(self.file_test)
        #self.n_stest = len(x_stest)
        #print("x_stest: ", self.n_stest)
        #print()
    
    
        ## -----------------------------------------------------
        
        #self.nclasses = len(set(train_labels))
        
        #print("Creating bag-of-n-grams")
        #self.X_STRAIN = self.create_initial_bagngrams(x_strain)
        #self.X_SVAL = self.create_bagngrams(x_sval)
        #self.Y_STRAIN = self.create_label_vec(self.n_strain, self.nclasses, y_strain)
        #self.Y_SVAL = self.create_label_vec(self.n_sval, self.nclasses, y_sval)
        
        #self.X_RTEST = self.create_bagngrams(x_rtest)
        #self.X_RVAL = self.create_bagngrams(x_rval)
        #self.Y_RTEST = self.create_label_vec(self.n_rtest, self.nclasses, y_rtest)
        #self.Y_RVAL = self.create_label_vec(self.n_rval, self.nclasses, y_rval)
        
        #self.X_STEST = self.create_bagngrams(x_stest)
        #self.Y_STEST = self.create_label_vec(self.n_stest, self.nclasses, y_stest)
    
        #self.nwords = self.X_STRAIN.shape[1]
    
    
    def convert_format(self, raw_file):
        documents = []
        sentences = parse(raw_file)
        for sent in sentences:
            new_sent = ''
            for word in sent:
                new_sent = new_sent + ' ' + word['form']
            
            documents.append(new_sent)
        return documents
        
        
    def combine_manual_race(self, raw_aa, raw_wh, raw_aa_labels, raw_wh_labels):
        full_man_race = raw_aa + raw_wh
        full_man_race_labels = raw_aa_labels + raw_wh_labels
        
        return full_man_race, full_man_race_labels
        
            
    def split_rand_subset_SFULL(self):
        for filename in os.listdir(self.index_dir):
            if '_'+str(self.run_number)+'.txt' in filename:
                subset = np.loadtxt(self.index_dir+filename, dtype=np.object)
        
        subset = subset.astype(int).tolist()  
        sub = [self.file_train[i] for i in subset]
        return sub
    
        
    def split_Strain_Sval(self, train_set):
        for filename in os.listdir(self.index_Sval):
            if '_'+str(self.run_number)+'.txt' in filename:
                subset = np.loadtxt(self.index_Sval+filename, dtype=np.object)
        
        subset = subset.astype(int).tolist()  
        sval = [train_set[i] for i in subset]
        strain = [element for i, element in enumerate(train_set) if i not in subset]
        return strain, sval
    
    
    def split_Rtest_Rval(self, _set):
        for filename in os.listdir(self.index_Rval):
            if '_'+str(self.run_number)+'.txt' in filename:
                subset = np.loadtxt(self.index_Rval+filename, dtype=np.object)
        
        subset = subset.astype(int).tolist()  
        rval = [_set[i] for i in subset]
        rtest = [element for i, element in enumerate(_set) if i not in subset]
        return rtest, rval
        
        
    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_instances_and_labels(self, subset):
        words =  []
        labels = []
        documents = []
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')
 
        for x in subset[0:-1]:
            inst = ''
            label = x[0:10]
        
            if label[0:9] != '__label__':
                print("ERROR in label creation. label: ", label)
                break
            else:
                labels.append(float(label[-1]))
            
            sent = ''
            word = ''
            for w in x[10:]:
                if w in whitelist:
                    if w == '\t':
                        inst = inst + '\t' + sent
                        sent = ''
                        word = ''
                    elif w != ' ':
                        word = word + w
                    else:
                        if "http" not in word and word != "RT" and word != "rt":
                            sent = sent + ' ' + word
                            word = ''
                        else:
                            word = ''
        
            documents.append(inst)

        print(len(documents), " total instances")
        return documents, labels
        
        
    # this function creates the instances of the manually labeled (Kaggle) dataset
    def create_instances_and_labels_manset(self, manual_set):
        words =  []
        labels = []
        documents = []
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')
        num = 0
        
        # loop through each instance in training data, gets labels
        for x in manual_set[0:-1]:
            if num != 361 and num != 360 and num != 359:
                inst = ''
                label = x[0:10]
                if label[0:9] != '__label__':
                    print("ERROR in manual label creation. Label: ", label)
                    break
                else:
                    labels.append(float(label[-1]))
                    
                sent = ''
                word = ''
                for w in x[10:]:
                    if w in whitelist:
                        if w == '\t':
                            inst = inst + '\t' + sent
                            sent = ''
                            word = ''
                        elif w != ' ':
                            word = word + w
                        else:
                            if "http" not in word and word != "RT" and word != "rt":
                                sent = sent + ' ' + word
                                word = ''
                            else:
                                word = ''
                
                documents.append(inst)
            num += 1
        
        print(len(documents), " total manual instances")
        return documents, labels
    
    
    def words_and_char_ngrams(self, text):
        words = re.findall(r'\w{6,}', text)  # {3,} "3 OR MORE"
        for w in words:
            numgrams = 3
            yield w
            while numgrams > 1:
                for i in range(len(w) - numgrams):
                    yield w[i:i+numgrams]
                numgrams -= 1


    def create_initial_bagngrams(self, x_train): 
        #self.vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount, max_features=self.bucket)
        #data_features = self.vectorizer.fit_transform(x_train) 
        
        #self.vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
        #data_features = self.vectorizer.fit_transform(x_train) 
        
        #********
        self.vectorizer = CountVectorizer(analyzer=self.words_and_char_ngrams, ngram_range=(1,self.ngrams), max_features=self.bucket)
        data_features = self.vectorizer.fit_transform(x_train)
           
        return data_features
        
        
    def create_bagngrams(self, instances):
        return self.vectorizer.transform(instances)    


    def create_label_vec(self, ninstances, nclasses, y):
        labels = np.zeros((ninstances, nclasses))
        
        n_males = 0
        n_females = 0
        
        i = 0
        for label in labels:
            if y[i] == 0:
                label[0] = 1.0
                n_males += 1        #NOTE: need to double check 
            elif y[i] == 1:
                label[1] = 1.0
                n_females += 1      #NOTE: need to double check 
            
            i += 1
            
        return labels #, n_males, n_females
    
    
    
    
    
    
    
    
    
    

        
