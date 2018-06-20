# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer

class Dictionary:
    def __init__(self, file_, args):
        self.file_ = file_
        self.args = args
        self.create_instances()s
        self.create_bagngrams()


    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_instances(self):
        combined = []
        for inst in self.file_:
            word = ''
            sentence = ''
            instance = ''
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
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
        vectorizer = CountVectorizer(ngram_range=(1,self.args.get_ngrams()), min_df=self.args.get_mincount(), max_df=self.args.get_maxcount()) 
        data_features = vectorizer.fit_transform(self.instances)
        #print data_features.shape
        self.bag_ngrams = data_features


    def get_bagngram():
        return self.bag_ngrams


        
