
import json
import csv



if __name__ == "__main__":
    
    corpus_dir='./corpora/query-gender.json'
    
    try:
        corpus = json.load(open(corpus_dir, 'r'))
    except FileNotFoundError:
        log("Something went wrong")
    
    userd = {}
    
    with open('twitter_ids.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        for idx, info in corpus['annotations'].items():
            userd[idx] = info['query_label2']
            #print(idx)
            writer.writerows([[idx]])
    
    print(len(userd))
        
