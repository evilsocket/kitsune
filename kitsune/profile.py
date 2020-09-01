import os
import glob
import traceback
import json

import kitsune.features as features

def load(profile_path, limit=200, quiet=False, extract_features_vector=True):
    profile_file = os.path.join(profile_path, 'profile.json')
    tweets_path = os.path.join(profile_path, 'tweets')

    if not os.path.exists(profile_file):
        # print("%s does not exist, skipping" % profile_file)
        return None

    if not os.path.exists(tweets_path):
        # print("%s does not exist, skipping" % tweets_path)
        return None

    try:
        profile  = None
        with open(profile_file, 'rt') as fp:
            profile = json.load(fp)

        tweets   = []
        replies  = []
        retweets = []

        if limit > 0:
            tweet_files = list(glob.glob(os.path.join(tweets_path, "*_*.json")))
            tweet_files.sort(key = lambda x: x.split('_').pop(), reverse=True)

            for filename in tweet_files:
                num_tweets = len(tweets)
                num_replies = len(replies)
                num_retweets = len(retweets)
                num_total = num_tweets + num_replies + num_retweets

                if num_total >= limit:
                    break

                with open(filename, 'rt') as fp:
                    tweet = json.load(fp)

                    if 'retweeted_status' in tweet and tweet['retweeted_status'] is not None:
                        retweets.append(tweet)
                    elif 'in_reply_to_status_id' in tweet and tweet['in_reply_to_status_id'] is not None:
                        replies.append(tweet)
                    else:
                        tweets.append(tweet)

            num_tweets = len(tweets)
            num_replies = len(replies)
            num_retweets = len(retweets)
            num_total = num_tweets + num_replies + num_retweets
            if num_total == 0:
                # print("%s does not have tweets, skipping" % tweets_path)
                return None

            data = (profile, tweets, replies, retweets)
            vector = None

            if extract_features_vector:
                if not quiet:
                    print("vectorializing %s : %d tweets, %d replies, %d retweets" % (profile_path, num_tweets, num_replies, num_retweets))        
                vector = features.extract(profile, tweets, replies, retweets)
            elif not quiet:
                print("loaded %s : %d tweets, %d replies, %d retweets" % (profile_path, num_tweets, num_replies, num_retweets))        
        else:
            data = (profile, [], [], [])
            vector = None
            if not quiet:
                print("loaded %s" % profile_path)        
    
        return (data, vector)

    except Exception as e:
        print(traceback.format_exc())
        print("problem loading %s: %s" % (profile_path, e))

    return None