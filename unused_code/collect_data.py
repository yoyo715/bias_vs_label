
# access users' descriptions

import tweepy
import json
from tweepy import OAuthHandler
from ACCESS_TOKENS_DONTSHARE import *   # To get twitter api authentication creds


# This listener will print out all Tweets it receives
class PrintListener(tweepy.StreamListener):
    def on_data(self, data):
        # Decode the JSON data
        tweet = json.loads(data)

        # Print out the Tweet
        print('@%s: %s' % (tweet['user']['screen_name'], tweet['text'].encode('ascii', 'ignore')))

    def on_error(self, status):
        print(status)
        
        
        
# Function to extract tweets
def get_tweets(username, api):
        # 200 tweets to be extracted
        number_of_tweets=200
        tweets = api.user_timeline(screen_name=username)
 
        # Empty Array
        tmp=[] 
 
        # create array of tweet information: username, 
        # tweet id, date/time, text
        tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created 
        for j in tweets_for_csv:
 
            # Appending tweets to the empty array tmp
            tmp.append(j) 
 
        # Printing the tweets
        print(tmp)
        
        
        
if __name__ == '__main__':
    
    CONSUMER_KEY = get_consumer_key()
    CONSUMER_SECRET = get_consumer_secret()
    ACCESS_KEY = get_access_token()
    ACCESS_SECRET = get_access_secret()

    auth = OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    api = tweepy.API(auth)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)

    #search
    api = tweepy.API(auth)
    
    listener = PrintListener()

    # Show system message
    print('I will now print Tweets containing "Python"! ==>')

    
    # Connect the stream to our listener
    stream = tweepy.Stream(auth, listener)
    stream.filter(track=['Python'])
    
    
    
    
    
