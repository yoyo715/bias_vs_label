# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split
import numpy as np
import random


class Dictionary2:
    def __init__(self, ngrams, mincount, bucket):

        self.subset_value = 1000

        #self.file_train = open('../data/query_gender.train', encoding='utf8').readlines()  
        self.file_train = open('../../simple-queries/data/query_gender.train', encoding='utf8').readlines() 
        del self.file_train[0]

        #self.file_test = open('../data/query_gender.test', encoding='utf8').readlines() 
        self.file_test = open('../../simple-queries/data/query_gender.test', encoding='utf8').readlines() 
        self.file_train.extend(self.file_test)
        self.dataset = self.file_train
        random.shuffle(self.dataset)

        self.ngrams = ngrams
        self.mincount = mincount
        self.bucket = bucket

        self.create_instances_and_labels()
        self.train_and_testsplit()
        self.create_bagngrams()
        self.create_test_bagngrams()
        
        self.nclasses = len(set(self.labels))
        self.create_train_labels()
        self.create_test_labels()
    
        self.nwords = self.train_bag_ngrams.shape[1]
        
        
        

    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_instances_and_labels(self):
        words =  []
        labels = []
        documents = []
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')

        # loop through each instance in training data, gets labels
        for x in self.dataset[0:self.subset_value]:
            i = 0
            inst = ''
            label = x[0:10]
            if label[0:9] != '__label__':
                print("ERROR in label creation")
                break
            else:
                labels.append(float(label[-1]))
                
            sent = ''
            word = ''
            for w in x[10:]:
                if w in whitelist:
                    if w == '\t':
                        #inst.append(sent)
                        inst = inst + '\t' + sent
                        sent = ''
                        word = ''
                        i += 1
                    elif w != ' ':
                        word = word + w
                    else:
                        if "http" not in word and word != "RT" and word != "rt":
                            sent = sent + ' ' + word
                            word = ''
                        else:
                            word = ''
            
            documents.append(inst)
        self.instances = documents
        self.labels = labels
        
        
    def create_ngrams(self, n):
        for inst in self.instances:
            for sentence in inst:
                ngrams = zip(*[sentence[i:] for i in range(n)])
                
                for gram in ngrams:
                    sentence.append(gram)
                
        print()
        print(self.instances[0][0])
                
                
    
    def create_char_ngrams(self, n):
        for inst in self.instances:
            for sentence in inst:
                for word in sentence:
                    #char_ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
                    
                    #for gram in char_ngrams:
                        #sentence.append(gram)
                        
                    print(word)
        print()
        print(self.instances[0][0])
        
        
    def train_and_testsplit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.instances, self.labels, test_size=0.33)
        
        self.n_train_instances = len(self.X_train)
        self.n_test_instances = len(self.X_test)
        self.n_train_labels = len(self.y_train)
        self.n_test_labels = len(self.y_test)
    

    def create_bagngrams(self): 
        #self.vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount, max_features=self.bucket)
        self.vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
        
        #self.vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=self.mincount)
        #self.vectorizer = TfidfVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount, max_features=self.bucket)
        data_features = self.vectorizer.fit_transform(self.X_train)    
        self.train_bag_ngrams = data_features
        
        
    #creates a bagngrams for testing instances
    def create_test_bagngrams(self): 
        data_features = self.vectorizer.transform(self.X_test)    
        self.test_bag_ngrams = data_features



    # index 0: label 0
    # index 1: label 1
    def create_train_labels(self):
        labels = np.zeros((self.n_train_instances, self.nclasses))
        print("train labels shape:", labels.shape)
        
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
        print("test labels shape:", labels.shape)
        
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




    def get_nclasses(self):
        return self.nclasses


    def get_nlabels_eachclass_train(self):
        return self.train_females, self.train_males
    
    
    def get_nlabels_eachclass_test(self):
        return self.test_females, self.test_males
        
    
    def get_train_and_test(self):
        return self.train_bag_ngrams, self.test_bag_ngrams, self.label_vec, self.test_label_vec
    
    
    def get_nwords(self):
        return self.nwords


    def get_n_train_instances(self):
        return self.n_train_instances
    
    
    def get_n_test_instances(self):
        return self.n_test_instances
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
