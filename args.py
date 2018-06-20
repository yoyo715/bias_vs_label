

class Args:
    def __init__(self, args):
        self.ngrams = args.get('ngrams')
        self.mincount = args.get('mincount')
        self.maxcount = args.get('maxcount')

    
    def get_ngrams(self):
        return int(self.ngrams)

    
    def get_mincount(self):
        return int(self.mincount)


    def get_maxcount(self):
        return float(self.maxcount)
