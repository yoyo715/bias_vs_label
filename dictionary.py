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
        for inst in self.file_:
            word = ''
            sentence = ''
            instance = ''
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
                else:
                    word += letter
            combined.append(instance)  
        
        del combined[0]
        self.instances = combined
    

    def create_bagngrams(self): 
        #vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount)
        vectorizer = CountVectorizer(ngram_range=(1,1), min_df=self.mincount)
        data_features = vectorizer.fit_transform(self.instances)    
        self.bag_ngrams = data_features


    def get_nwords(self):
        return self.nwords


    def get_ninstances(self):
        return self.ninstances


    # index 0: label 0
    # index 1: label 1
    def get_labels(self):
        labels = np.zeros((self.ninstances, self.nlabels))
        i = 0
        for label in labels:
            if self.labels[i] == 0:
                label[0] = 1
            elif self.labels[i] == 1:
                label[1] = 1
            
            i += 1
            
        self.label_vec = labels
        return self.label_vec


    def get_bagngram(self):
        return self.bag_ngrams


    def get_nlabels(self):
        return self.nlabels


        
