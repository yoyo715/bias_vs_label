# Dictionary class

from sklearn.feature_extraction.text import CountVectorizer

class Dictionary:
    def __init__(self, file_, args):
        self.file_ = file_
        self.args = args
        #self.combine_words()
        self.create_instances()
        #self.create_freqtable()
        self.create_bagngrams()


    # adds each instance a separate element in list
    # each 'tweet' is separated by tab
    def create_instances(self):
        combined = []
        for inst in self.file_:
            word = ''
            sentence = ''
            instance = ''
            #instance = []
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
                        #instance.append(word)
                        #instance = instance + "\t" + word
                        word = ''
                    else:
                        sentence += word + ' '
                        word = ''
                elif letter == "\t":    
                    #instance.append(sentence)
                    instance = instance + "\t" + sentence
                    sentence = ''
                else:
                    word += letter
            combined.append(instance)  
        
        del combined[0]
        self.instances = combined
        #print len(combined)

    
    # adds each instance a separate element in list
    # each tweet is separate element in list of instance
    def combine_words(self):
        combined = []
        for inst in self.file_:
            word = ''
            sentence = ''
            instance = []
            for letter in inst:
                if letter == ' ':
                    if "label" in word:
                        instance.append(word)
                        word = ''
                    else:
                        sentence += word + ' '
                        word = ''
                elif letter == "\t":    
                    instance.append(sentence)
                    sentence = ''
                else:
                    word += letter
            combined.append(instance)
            #self.instances = combined
            #print self.instances
            

    def create_freqtable(self):
        freqtable = dict()
        for inst in self.instances:
            for sent in inst:
                for word in sent.split():
                    if word in freqtable:
                        freqtable[word] += 1
                    else:
                        freqtable[word] = 1
        self.freq_table = freqtable
        self.process_freq_table()


    def process_freq_table(self):
	    #print len(self.freq_table)
        for k, v in self.freq_table.items():
            if int(v) < int(self.args.get_mincount()):
                del self.freq_table[k]
	    #print len(self.freq_table)
        self.vocab_length = len(self.freq_table)
        

    def create_bagngrams(self): 
        vectorizer = CountVectorizer(ngram_range=(1,self.args.get_ngrams()), min_df=self.args.get_mincount(), max_df=self.args.get_maxcount()) 
        data_features = vectorizer.fit_transform(self.instances)
        #print data_features.shape
        self.bag_ngrams = data_features


    def get_bagngram():
        return self.bag_ngrams


        
