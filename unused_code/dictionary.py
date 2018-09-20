# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import numpy as np

class Dictionary:
    def __init__(self, file_, ngrams, mincount, bucket):
        self.file_ = file_
        self.ngrams = ngrams
        self.mincount = mincount
        self.bucket = bucket

        self.create_instances()
        self.train_and_testsplit()
        self.create_bagngrams()
        self.create_test_bagngrams()
        
        self.nclasses = len(set(self.labels))
        self.create_train_labels()
        self.create_test_labels()
    
        self.nwords = self.train_bag_ngrams.shape[1]
        
        
        

    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_instances(self):
        combined = []
        self.labels = []
        numsents = 0
        
        for inst in self.file_:
            word = ''
            sentence = ''
            instance = ''
            
            #sentence = []
            #instance = []
            
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
                        if word[-1] == '1' or word[-1] == '0':
                            self.labels.append(int(word[-1]))
                            word = ''
                        else:
                            sentence += word + ' '
                            #sentence.append(word)
                            word = ''
                    else:
                        sentence += word + ' '
                        #sentence.append(word)
                        word = ''
                elif letter == "\t":    
                    instance = instance + "\t" + sentence
                    #instance.append(sentence)
                    sentence = ''
                    #sentence = []
                else:
                    word += letter
            combined.append(instance)  
            
        self.instances = combined
        
        #print(self.instances[0][0])
        #self.create_ngrams(2)
        #self.create_char_ngrams(3)
        
        
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
        
        #print(len(self.X_train), len(self.X_test), len(self.y_train), len(self.y_test))
    

    def create_bagngrams(self): 
        #self.vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount, max_features=self.bucket)
        self.vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
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
                label[0] = 1
                self.train_males += 1
            elif self.y_train[i] == 1:
                label[1] = 1
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
                label[0] = 1
                self.test_males += 1        #NOTE: need to double check 
            elif self.y_test[i] == 1:
                label[1] = 1
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
