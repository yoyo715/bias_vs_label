#preprocess.py

if __name__ == '__main__':
    
    # read in training and testing files
    #train = open('./data/query_gender.train').readlines()
    #test = open('./data/query_gender.test').readlines()
    
    
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')
    
    # nltk's stop words
    stop_words = ["didn't", 'about', 'ma', 'the', "you've", 'should', 'your', 'so',
                 'yourself', 'why', 'not', "should've", 'above', 'won', "doesn't",
                 'needn', 'and', 'more', 'nor', 'shouldn', "you'd", 'don', 'aren',
                 'during', 'couldn', 'our', 'were', 'them', 'own', 'he', 'by', 'out',
                 'his', 'd', 'himself', 'was', 'in', 'her', 'll', 'hers', 'an', 'hadn',
                 'i', 'where', 'm', 'y', 'will', 'most', 'wouldn', 'are', 'does',
                 "that'll", 'into', 'who', 'over', "it's", "don't", 'these', 'while',
                 'myself', "you're", 'is', "hadn't", 'a', "haven't", 'both', 'herself',
                 'weren', 'as', 'him', 'just', 'am', 're', 'any', 'other', 'didn', 'those',
                 "aren't", 'to', 'of', 'has', 'before', 'further', 've', "hasn't", 'wasn',
                 'some', 'yours', 'did', 'do', 'but', 'for', 's', "you'll", "shouldn't",
                 'this', 'can', "couldn't", "wasn't", 'through', "won't", 'having',
                 'haven', 'hasn', "mightn't", 'ours', 'they', 'me', 'again', "she's",
                 'because', 'below', 'its', 'until', 'ain', 'now', 'theirs', 'on', 'isn',
                 'up', 'such', "needn't", 'between', 'off', 'my', 'then', 'all', 'each',
                 'at', 'no', 'or', 'had', 'yourselves', "isn't", 'whom', 'same', 'that',
                 'than', 'against', "weren't", 'if', 'be', 'down', 'here', 'mightn', 'when',
                 'ourselves', 'under', 'doing', 'too', "mustn't", 'you', 'doesn', 'after',
                 "shan't", 'she', "wouldn't", 'what', 'shan', 't', 'we', 'with', 'from',
                 'it', 'been', 'only', 'have', 'which', 'themselves', 'few', 'once',
                 'there', 'being', 'how', 'very', 'o', 'mustn', 'itself', 'their', 'u', 'ur']
    
    #train = open('/local_d/RESEARCH/fastTextRecreation/data/query_gender_subset_train.txt', encoding='utf8').readlines() 
    train = open('../data/query_gender_subset_train.txt', encoding='utf8').readlines() 
    #train = open('/home/mcooley/Desktop/temp/query_gender_subset_train.txt').readlines() 
    train_cleaned = open('../cleaned_train_withstopwords.txt', 'w')
    
    # gets rid up unknown characters
    cleanedtrain1 = []
    for inst in train:
        answer = ''.join(l for l in inst if l in whitelist)
        cleanedtrain1.append(answer.lower())
            
    # removes links and reformats labels and removes stopwords
    for inst in cleanedtrain1:
        word = ''
        sentence = ''
        for letter in inst:
            if letter == ' ':
                if "label" in word:
                    word = '\n'+"__label__"+word[-1]
                if "http" not in word and word != "RT" and word != "rt": #and word not in stop_words:
                    sentence += word + ' '
                word = ''
            elif letter == "\t":
                sentence = sentence + '\t'
                train_cleaned.write(sentence)
                sentence = ''
            else:
                word += letter 
                
    train_cleaned.close()
    
