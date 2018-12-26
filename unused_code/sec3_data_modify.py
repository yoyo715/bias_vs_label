import json
#from misc_keys import twitter_keys
from ACCESS_TOKENS_DONTSHARE import *
import pandas as pd
from time import localtime
from time import sleep
from time import strftime
import tweepy
import logging as log


#def tw_connect(keys):
def tw_connect(app_pub, app_priv, con_key, con_priv):
    #auth = tweepy.OAuthHandler(keys['app_public'], keys['app_secret'])
    #auth.set_access_token(keys['per_public'], keys['per_secret'])
    
    auth = tweepy.OAuthHandler(app_pub, app_priv)
    auth.set_access_token(con_key, con_priv)
    
    return tweepy.API(auth)

try:
    #assert twitter_keys['app_public']
    app_pub = get_access_token()
    app_priv = get_access_secret()
    con_key = get_consumer_key()
    con_priv = get_consumer_secret()
    
    assert app_pub
    #API = tw_connect(twitter_keys)
    API = tw_connect(app_pub, app_priv, con_key, con_priv)
except AssertionError:
    log("API keys are empty. Please provide them in misc_keys.py...")
    exit()
    
class DB(object):

    def __init__(self, db_name, mode):
        """Open file directory."""
        self.mode = mode
        try:
            self.db = open('/local_madim/Desktop/ML_research/gitfiles/bias_vs_labelefficiency/data/' + db_name + '.db', mode)
        except FileNotFoundError:
            fo = open('/local_madim/Desktop/ML_research/gitfiles/bias_vs_labelefficiency/data/' + db_name + '.db', 'w')
            fo.close()
            self.db = open('/local_madim/Desktop/ML_research/gitfiles/bias_vs_labelefficiency/data/' + db_name + '.db', mode)

    def insert(self, jsonf):
        """Write json line to file."""
        self.db.write(json.dumps(jsonf) + "\n")

    def commit(self):
        """Write changes to disk."""
        self.db.close()

    def fetch_key(self, key):
        """Fetch values for key."""
        for jsf in self.loop():
            yield jsf[key]

    def loop(self):
        """Iterate through db."""
        assert self.mode == 'r'
        for line in self.db:
            jsf = json.loads(line)
            yield jsf

def reconstruct_ids(db_id):
    """Extract query and user information from existing file."""
    uds = DB(db_id + '_fix', 'r')
    user_ids, query_ids = {}, {}
    if not uds.db.read():
        uds = DB(db_id, 'r')
    for line in uds.loop():
        try:
            user_ids[line['user_id']] = line['label']
            query_ids[line['tweet_id']] = line['query']
        except KeyError:
            user_ids[line['id']] = line['label']
            
    print(user_ids)
    return user_ids, query_ids

class DistantCollection(object):

    def __init__(self, query_string, query_words, filters, flip_any,
                 flip_prefix, clean_level='messages', mode='live',
                 db_id='twitter_gender'):
        """Set collection and queries."""
        self.id = db_id
        self.hits = DB(self.id, 'a')
        self.users = DB(self.id + '_usr', 'a')
        self.messages = DB(self.id + '_msg', 'a')
        self.hit_fix = DB(self.id + '_fix', 'a')
        self.msg_fix = DB(self.id + '_msg_fix', 'a')

        self.queries = {query_string.format(k): v for
                        k, v in query_words.items()}
        self.filter = filters
        self.flip_any = flip_any
        self.flip_prefix = flip_prefix

        self.user_ids = dict()
        self.query_ids = dict()

        self.clean_level = clean_level
        self.max = 0 if mode == 'live' else 1

    def remove_query_tweets(self):
        """Remove query hits from tweets."""
        db = DB(self.id + '_msg', 'r')
        for line in db.loop():
            label = self.user_ids[line['user_id']]
            line['distant_label'] = label
            if self.clean_level == 'messages':
                if not any([query in line['tweet_text'].lower()
                            for query in self.queries]):
                    self.msg_fix.insert(line)
            else:
                if line['tweet_id'] not in self.query_ids:
                    self.msg_fix.insert(line)

    def flip_label(self, uid, tid, text):
        """Return flipped label if rules in text, return none if in filter."""
        query_tails = [" ", ".", "!", ",", ":", ";"]  # etc
        if any([it in text for it in self.filter]):  # if illegal
            return
        label = self.user_ids[uid]
        query = self.query_ids[tid]
        if any([query + affix in text for affix in query_tails]):
            if any([f in text for f in self.flip_any]):
                label = 'm' if label == 'f' else 'f'
            elif any([p + query in text for p in self.flip_prefix]):
                label = 'm' if label == 'f' else 'f'
        return label

    def correct_query_tweets(self):
        """Correct the query tweets using heuristics, write to new file."""
        db = DB(self.id, 'r')
        for line in db.loop():
            new_label = self.flip_label(line['user_id'],
                                        line['tweet_id'],
                                        line['tweet_text'].lower())
            if new_label:
                line['label'] = new_label
                self.hit_fix.insert(line)
        self.hit_fix.commit()

    def get_users(self, cursor, label, query):
        """Given a query cursor, store user profile and label."""
        try:
            for page in cursor.pages():
                log("flipping page...")
                for tweet in page:
                    try:
                        self.user_ids[tweet.user.id] = label
                        self.query_ids[tweet.id] = query
                        self.users.insert(tweet.user._json)
                        self.hits.insert({'user_id': tweet.user.id,
                                          'tweet_id': tweet.id,
                                          'tweet_text': tweet.text,
                                          'location': tweet.location,
                                          'label': label,
                                          'query': query})
                    except Exception as e:
                        log("error getting users: " + str(e))
                if self.max:
                    break

        except tweepy.TweepError:
            log("Rate limit hit, going to zzz....")
            sleep(300)
            self.get_users(cursor, label, query)

    def get_queries(self):
        """Search Twitter API for tweets matching queries and fetch users."""
        for query, label in self.queries.items():
            query = '"' + query + '"'
            cursor = tweepy.Cursor(API.search, q=query, include_entities=True,
                                   count=200)
            self.get_users(cursor, label, query)
            if self.max:
                break

    def fetch_query_tweets(self):
        """Given query assignments, collect query tweets with API tokens."""
        self.get_queries()
        self.hits.commit()
        self.users.commit()
        self.correct_query_tweets()

    def get_tweets(self, cursor):
        """Given a timeline cursor, fetch tweets and remove user object."""
        try:
            for page in cursor.pages():
                for tweet in page:
                    tweet = tweet._json
                    del tweet['user']
                    yield tweet
        except tweepy.TweepError:
            log("Rate limit hit, going to zzz....")
            sleep(5)
            self.get_tweets(cursor)

    def get_timelines(self):
        """Given ID assignments, collect timelines with provided API tokens."""
        for user_id in self.user_ids:
            cursor = tweepy.Cursor(API.user_timeline, id=user_id, count=200)
            for tweet in self.get_tweets(cursor):
                tweet['user_id'] = user_id
                assert not tweet.get('user')
                self.messages.insert({'tweet_id': tweet['id'],
                                      'user_id': user_id,
                                      'tweet_text': tweet['text']})
            log("Fetched user...")
            if self.max:
                break

    def get_location(self):
       for user_id in self.user_ids:
           cursor = tweepy.Cursor(API.user_timeline, id=user_id, count=200)
           for tweet in self.get_tweets(cursor):
               tweet['user_id'] = user_id
               assert not tweet.get('user')
               self.messages.insert({'tweet_id': tweet['id'],
                                     'user_id': user_id,
                                     'user_location': tweet['location']})
           log("Fetched user...")
           if self.max:
               break

    def fetch_user_tweets(self):
        """Divide all ids amongst API connections and thread them."""
        try:
            assert self.user_ids
        except (AssertionError, AttributeError):
            self.user_ids, self.query_ids = reconstruct_ids(self.id)

        self.get_timelines()
        self.messages.commit()
        if self.id == 'twitter_gender' or self.id == 'query_gender':
            self.remove_query_tweets()

class QueryCollection(DistantCollection):
    def __init__(self, db_id='query_gender',
                 #corpus_dir='./corpora/query-gender.json',
                 corpus_dir='/local_madim/Desktop/ML_research/gitfiles/bias_vs_labelefficiency/data/query-gender.json',
                 clean_level='messages', mode='live'):
        """Call correct databases corresponding to class, open corpus."""
        self.id = db_id
        self.users = DB(self.id, 'a')
        self.messages = DB(self.id + '_msg', 'a')
        self.msg_fix = DB(self.id + '_msg_fix', 'a')

        # NOTE: these are hardcoded to reproduce the paper
        query_string = 'm a {0}'
        query_words = {'girl': 'f', 'boy': 'm', 'man': 'm', 'woman': 'f',
                       'guy': 'm', 'dude': 'm', 'gal': 'f', 'female': 'f',
                       'male': 'm'}
        self.queries = {query_string.format(k): v for
                        k, v in query_words.items()}

        self.user_ids = dict()

        self.clean_level = clean_level
        self.max = 0 if mode == 'live' else 1

        try:
            self.corpus = json.load(open(corpus_dir, 'r'))
        except FileNotFoundError:
            log("Something went wrong while loading the query corpus. Re-download from http://github.com/cmry/simple-queries and store in corpora")

    def fetch_users(self):
        """Collect the users in the Query corpus."""
        userd = {}
        for idx, info in self.corpus['annotations'].items():
            userd[idx] = info['query_label2']
            if len(userd) == 100:
                userl = list(userd.keys())
                users = API.lookup_users(userl)
                for user in users:
                    user = user._json
                    user['label'] = userd[user['id_str']]
                    self.users.insert(user)
                userd = {}
                if self.max:
                    break
        self.users.commit()
        
tc = QueryCollection(db_id='query_gender', clean_level='messages')
tc.fetch_users()
tc.fetch_user_tweets()
print(tc.user_ids)

