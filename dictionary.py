# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer

class Dictionary:
    def __init__(self, file_, ngrams, mincount):
        self.file_ = file_
        self.ngrams = ngrams
        self.mincount = mincount

        self.create_instances()
        self.create_bagngrams()
    


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
                        self.labels.append(int(word[-1]))
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
        vectorizer = CountVectorizer(ngram_range=(1,self.ngrams), min_df=self.mincount) 
        data_features = vectorizer.fit_transform(self.instances)    
        self.bag_ngrams = data_features


    def get_nwords(self):
        self.nwords = self.bag_ngrams.shape[1]
        return self.nwords


    def get_ninstances(self):
        self.ninstances = self.bag_ngrams.shape[0]
        return self.ninstances


    def get_labels(self):
        return self.labels


    def get_bagngram(self):
        return self.bag_ngrams


        
