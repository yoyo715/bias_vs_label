# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class Dictionary:
    def __init__(self, file_, ngrams, mincount):
        self.file_ = file_
        self.ngrams = ngrams
        self.mincount = mincount

        self.create_instances()
        self.create_bagngrams()
    
        self.nwords = self.bag_ngrams.shape[1]
        self.ninstances = self.bag_ngrams.shape[0]
        self.nlabels = len(set(self.labels))
        

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
            
            #if numsents != 199:
                #print("ERROR (train) instances does not have 200 tweets: ", numsents)
            numsents = 0 
            
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
                        if word[-1] == '1' or word[-1] == '0':
                            self.labels.append(int(word[-1]))
                            word = ''
                        else:
                            sentence += word + ' '
                            word = ''
                    else:
                        sentence += word + ' '
                        word = ''
                elif letter == "\t":    
                    instance = instance + "\t" + sentence
                    sentence = ''
                    numsents += 1
                else:
                    word += letter
            combined.append(instance)  
            
        del combined[0]
        self.instances = combined
        
        
      
      
    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_test_instances(self, test):
        combined = []
        self.test_labels = []
        numsents = 0
        
        for inst in test:
            word = ''
            sentence = ''
            instance = ''
            
            #if numsents != 199:
                #print("ERROR: (test) instances does not have 200 tweets", numsents)
            numsents = 0      
            
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
                        if word[-1] == '1' or word[-1] == '0':
                            self.test_labels.append(int(word[-1]))
                            word = ''
                        else:
                            sentence += word + ' '
                            word = ''
                    else:
                        sentence += word + ' '
                        word = ''
                elif letter == "\t":    
                    instance = instance + "\t" + sentence
                    sentence = ''
                    numsents += 1
                else:
                    word += letter
            combined.append(instance)  
            
        
        del combined[0]
        self.test_instances = combined
    

    def create_bagngrams(self): 
        #vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount)
        self.vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
        data_features = self.vectorizer.fit_transform(self.instances)    
        self.bag_ngrams = data_features
        
        
    #creates a bagngrams for testing instances
    def create_test_bagngrams(self): 
        data_features = self.vectorizer.transform(self.test_instances)    
        self.test_bag_ngrams = data_features
        self.test_ninstances = self.test_bag_ngrams.shape[0]
        return self.test_bag_ngrams


    def get_nwords(self):
        return self.nwords


    def get_ninstances(self):
        return self.ninstances
    
    
    def get_test_ninstances(self):
        return self.test_ninstances


    # index 0: label 0
    # index 1: label 1
    def get_labels(self):
        labels = np.zeros((self.ninstances, self.nlabels))
        print(labels.shape)
        self.train_males = 0
        self.train_females = 0
        
        i = 0
        for label in labels:
            if self.labels[i] == 0:
                label[0] = 1
                self.train_males += 1
            elif self.labels[i] == 1:
                label[1] = 1
                self.train_females += 1
            
            i += 1
            
        self.label_vec = labels
        return self.label_vec
    
    
    # index 0: label 0
    # index 1: label 1
    def get_test_labels(self):
        self.test_nlabels = len(set(self.test_labels))
        labels = np.zeros((self.test_ninstances, self.test_nlabels))
        print(labels.shape)
        self.test_males = 0
        self.test_females = 0
        
        i = 0
        for label in labels:
            if self.test_labels[i] == 0:
                label[0] = 1
                self.test_males += 1        #NOTE: need to double check 
            elif self.test_labels[i] == 1:
                label[1] = 1
                self.test_females += 1      #NOTE: need to double check 
            
            i += 1
            
        self.test_label_vec = labels
        return self.test_label_vec


    def get_bagngram(self):
        return self.bag_ngrams


    def get_nlabels(self):
        return self.nlabels


    def get_nlabels_eachclass_train(self):
        return self.train_females, self.train_males
    
    
    def get_nlabels_eachclass_test(self):
        return self.test_females, self.test_males
        
    
    def train_and_testsplit(self):
        print(type(self.bag_ngrams))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
