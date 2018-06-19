

class Args:
    def __init__(self, args):
        self.ngrams = args.get('ngrams')
        self.mincount = args.get('mincount')

    
    def get_ngrams(self):
        return self.ngrams

    
    def get_mincount(self):
        return self.mincount
