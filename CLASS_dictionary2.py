#CLASS_dictionary.py

# Dictionary class

"""
    This version of the dictionary creates bag of words with both word ngrams and char ngrams

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import random, re
import time
import os


class Dictionary:
    def __init__(self, ngrams, mincount, bucket, run):
        self.run_number = run
        self.ngrams = ngrams
        self.mincount = mincount
        self.bucket = bucket
        
        TETON = False    # WORKING ON TETON OR NOT
        if TETON == True:
            self.file_train = open('/project/lsrtwitter/mcooley3/data/query_gender.train', encoding='utf8').readlines()     
            self.file_test = open('/project/lsrtwitter/mcooley3/data/query_gender.test', encoding='utf8').readlines()       
            self.manual_set = open('/project/lsrtwitter/mcooley3/data/FULL_manual_set.txt', encoding='utf8').readlines()                   
            self.index_dir = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/indices/'   
            self.index_Rval = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/indices_Rval/'
            self.index_Sval = '/project/lsrtwitter/mcooley3/bias_vs_labelefficiency/indices_Sval/'
        else:
            self.file_train = open('../../../simple-queries/data/query_gender.train', encoding='utf8').readlines()
            self.file_test = open('../../../simple-queries/data/query_gender.test', encoding='utf8').readlines() 
            self.manual_set = open('../../FULL_manual_set.txt', encoding='utf8').readlines()       
            self.index_dir = './indices/'  
            self.index_Rval = './indices_Rval/'
            self.index_Sval = './indices_Sval/'
            
        del self.file_train[0]  # Blank line screws some things up
        
        print("- creating manual instances")
        Rtest, Rval = self.split_Rtest_Rval(self.manual_set)
        manual_instances, y_manual = self.create_instances_and_labels_manset(maual_subset)

        print("- creating train instances")
        train_instances, train_labels = self.create_instances_and_labels(train_subset)
        
        print("- creating testing instances")
        test_instances, test_labels = self.create_instances_and_labels(test_subset)
    
        self.create_sets()
        self.create_bagngrams()
        self.create_test_bagngrams()
        self.create_manual_bagngrams()
        
        self.nclasses = len(set(self.train_labels))
        self.create_train_labels()
        self.create_test_labels()
        self.create_manual_labels()
    
        self.nwords = self.train_bag_ngrams.shape[1]
    
    
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
        strain = [train_set[i] for i not in subset]
        return strain, sval
    
    
    def split_Rtest_Rval(self, _set):
        for filename in os.listdir(self.index_Rval):
            subset = np.loadtxt(self.index_Rval+filename, dtype=np.object)
        
        subset = subset.astype(int).tolist()  
        rval = [_set[i] for i in subset]
        rtest = [_set[i] for i not in subset]
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

        print("**** ", len(documents))
        #self.train_instances = documents
        #self.train_labels = labels
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
        
        print("**** ", len(documents))
        #self.manual_instances = documents
        #self.y_manual = labels
        #self.n_manual_instances = len(self.manual_instances)
        
        return documents, labels
        
        
    def create_sets(self):
        self.X_train = self.train_instances
        self.X_test = self.test_instances
        self.y_train = self.train_labels
        self.y_test = self.test_labels
        
        self.n_train_instances = len(self.X_train)
        self.n_test_instances = len(self.X_test)
        self.n_train_labels = len(self.y_train)
        self.n_test_labels = len(self.y_test)
    
    
    def words_and_char_ngrams(self, text):
        words = re.findall(r'\w{6,}', text)  # {3,} "3 OR MORE"
        for w in words:
            numgrams = 3
            yield w
            while numgrams > 1:
                for i in range(len(w) - numgrams):
                    yield w[i:i+numgrams]
                numgrams -= 1


    def create_bagngrams(self): 
        #self.vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount, max_features=self.bucket)
        #data_features = self.vectorizer.fit_transform(self.X_train) 
        
        #self.vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
        #data_features = self.vectorizer.fit_transform(self.X_train) 
        
        #********
        self.vectorizer = CountVectorizer(analyzer=self.words_and_char_ngrams, ngram_range=(1,self.ngrams), max_features=self.bucket)
        data_features = self.vectorizer.fit_transform(self.X_train)
           
        self.train_bag_ngrams = data_features
        
        
        
    #creates a bagngrams for testing instances
    def create_test_bagngrams(self): 
        data_features = self.vectorizer.transform(self.X_test)    
        self.test_bag_ngrams = data_features


    # bagngrams for the manually labeled dataset
    def create_manual_bagngrams(self):
        data_features = self.vectorizer.transform(self.manual_instances)    
        self.manual_test_bag_ngrams = data_features


    # index 0: label 0
    # index 1: label 1
    def create_train_labels(self):
        labels = np.zeros((self.n_train_instances, self.nclasses))
        
        self.train_males = 0
        self.train_females = 0
        
        i = 0
        for label in labels:
            if self.y_train[i] == 0:
                label[0] = 1.0
                self.train_males += 1
            elif self.y_train[i] == 1.:
                label[1] = 1.0
                self.train_females += 1
            
            i += 1
            
        self.label_vec = labels
    
    
    # index 0: label 0
    # index 1: label 1
    def create_test_labels(self):
        labels = np.zeros((self.n_test_instances, self.nclasses))
        #print("test labels shape:", labels.shape)
        
        self.test_males = 0
        self.test_females = 0
        
        i = 0
        for label in labels:
            if self.y_test[i] == 0:
                label[0] = 1.0
                self.test_males += 1        #NOTE: need to double check 
            elif self.y_test[i] == 1:
                label[1] = 1.0
                self.test_females += 1      #NOTE: need to double check 
            
            i += 1
            
        self.test_label_vec = labels
        
        
    # index 0: label 0
    # index 1: label 1
    def create_manual_labels(self):
        labels = np.zeros((self.n_manual_instances, self.nclasses))
        #print("manual labels shape:", labels.shape)
        
        self.manual_males = 0
        self.manual_females = 0
        
        i = 0
        for label in labels:
            if self.y_manual[i] == 0:
                label[0] = 1.0
                self.manual_males += 1        #NOTE: need to double check 
            elif self.y_manual[i] == 1:
                label[1] = 1.0
                self.manual_females += 1      #NOTE: need to double check 
            
            i += 1
            
        self.manual_label_vec = labels


    def get_nclasses(self):
        return self.nclasses


    def get_nlabels_eachclass_train(self):
        return self.train_females, self.train_males
    
    
    def get_nlabels_eachclass_test(self):
        return self.test_females, self.test_males
    
    
    def get_nlabels_eachclass_manual(self):
        return self.manual_females, self.manual_males
        
    
    def get_train_and_test(self):
        return self.train_bag_ngrams, self.test_bag_ngrams, self.label_vec, self.test_label_vec
    
    
    def get_trainset(self):
        return self.train_bag_ngrams


    def get_manual_testset(self):
        return self.manual_test_bag_ngrams


    def get_nwords(self):
        return self.nwords


    def get_n_train_instances(self):
        return self.n_train_instances
 
    
    def get_n_test_instances(self):
        return self.n_test_instances
    
    
    def get_n_manual_instances(self):
        return self.n_manual_instances


    def get_manual_set_labels(self):
        return self.manual_label_vec
    
    
    
    
    
    
    
    
    
    

        
