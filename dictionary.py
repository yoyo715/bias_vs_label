# Dictionary class

class Dictionary:
    def __init__(self, file_, args):
        self.file_ = file_
        self.args = args
        self.combine_words()
        self.create_freqtable()
        self.bag_words = None
        self.bag_ngrams = None


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
            self.instances = combined
            

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
        for k, v in self.freq_table.items():
            if v <= self.args.get_mincount():
                print v, self.args.get_mincount()
             
        
            

    def create_bagwords():  
        return 1


    def create_bagngram():
        return 1


    def get_freqtable(): 
        return 1


    def get_bagwords():
        return 1


    def get_bagngram():
        return 1


        
