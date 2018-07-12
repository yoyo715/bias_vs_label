#preprocess.py

if __name__ == '__main__':
    
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 \t \n')
    
    #train = open('/local_d/RESEARCH/fastTextRecreation/data/query_gender_subset_train.txt', encoding='utf8').readlines() 
    #train = open('../data/query_gender.train', encoding='utf8').readlines() 
    train = open('../data/query_gender.test', encoding='utf8').readlines() 
    #train = open('../data/query_gender.train', encoding='utf8').readlines() # full training dataset
    #train = open('/home/mcooley/Desktop/temp/query_gender_subset_train.txt').readlines() 
    train_cleaned = open('../cleaned_test_withstopwords_FULL.txt', 'w')
    

    # gets rid up unknown characters
    cleanedtrain = []
    for inst in train:
        answer = ''.join(l for l in inst if l in whitelist)
        cleanedtrain.append(answer.lower())
            

    # removes links and reformats labels and removes stopwords
    for inst in cleanedtrain:
        word = ''
        sentence = ''
        for letter in inst:
            if letter == ' ':
                if "label" in word:
                    if word[-1] == '1' or word[-1] == '0':
                        word = '\n'+"__label__"+word[-1]
                    #else:
                        #print("ERROR writing labels: ", word)

                if "http" not in word and word != "RT" and word != "rt":
                    sentence += word + ' '

                word = ''
            elif letter == "\t":
                sentence = sentence + '\t'
                train_cleaned.write(sentence)
                sentence = ''
            else:
                word += letter 
                

    train_cleaned.close()
    
