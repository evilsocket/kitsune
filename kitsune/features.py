import time
import math
import re
import numpy as np
from datetime import datetime
from email.utils import parsedate
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

today = datetime.today()

TFIDF_SIMILARITY_THRESHOLD = 0.7

def parse_date(str_date):
    return datetime(*(parsedate(str_date)[:6]))

def days_since(str_date):
    return (today - parse_date(str_date)).days

def ratio(a, b):
    return a / (b + .00000001) # avoid division by zero
 
def one_hot(b):
    return 1.0 if b else 0.0

def entropy(string):
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    return - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

def temporal_distribution(prefix, statuses):
    features = OrderedDict()

    by_hour = Counter({i : 0 for i in range(0, 24)})
    by_week_day = Counter({i : 0 for i in range(0, 7)})
    by_month = Counter({i : 0 for i in range(1, 13)})

    for status in statuses:
        parsed = parse_date(status['created_at'])
        by_hour[parsed.hour] += 1
        by_week_day[parsed.weekday()] += 1
        by_month[parsed.month] +=1

    values = []
    for hour in range(0, 24):
        features['%s_per_hour_of_day_%d' % (prefix, hour)] = by_hour[hour]
     
    for weekday in range(0, 7):
        features['%s_per_weekday_%d' % (prefix, hour)] = by_week_day[weekday]
 
    for month in range(1, 13):
        features['%s_per_month_%d' % (prefix, month)] = by_month[month]

    return features

def has_field(profile, field):
    return one_hot(True if field in profile and profile[field] not in (None, "") else False)

def safe_one_hot(profile, field):
    if field in profile:
        return one_hot(profile[field])
    else:
        return 0.0

def metrics_for_statuses(profile_id, statuses, status_metrics=False, rt_metrics=False, reply_metrics=False):
    num_statuses = len(statuses)
    avg_entropy = 0
    min_entropy = 999999
    max_entropy = 0

    avg_length = 0
    min_length = 999999
    max_length = 0

    avg_retweet_count = 0
    avg_favorite_count = 0

    rt_avg_reaction_time = 0.0
    rt_users = Counter()
    num_self_rt = 0
    num_rts = 0

    num_self_replies = 0

    for status in statuses:
        # size metrics
        text_size = len(status['text'])
        avg_length += text_size
        if text_size < min_length:
            min_length = text_size
        if text_size > max_length:
            max_length = text_size

        # shannon entropy metrics
        text_entropy = entropy(status['text'])
        avg_entropy += text_entropy
        if text_entropy < min_entropy:
            min_entropy = text_entropy
        if text_entropy > max_entropy:
            max_entropy = text_entropy

        # if these are statuses, extract specific metrics
        if status_metrics:
            avg_retweet_count += status['retweet_count']
            avg_favorite_count += status['favorite_count']

        # if these are retweets, extract specific metrics
        if rt_metrics:
            # collect retweet reaction times
            created_at = parse_date(status['retweeted_status']['created_at'])
            retweeted_at = parse_date(status['created_at'])
            rt_avg_reaction_time += (retweeted_at - created_at).seconds
            num_rts += 1

            # collect retweeted users
            rt_users.update([status['retweeted_status']['user']['screen_name'].lower()]) 
            # count self retweets
            if 'retweeted_status' in status and status['retweeted_status'] is not None and status['retweeted_status']['user']['id'] == profile_id:
                num_self_rt += 1

        if reply_metrics:
            # count self replies
            if 'in_reply_to_user_id' in status and status['in_reply_to_user_id'] is not None and status['in_reply_to_user_id'] == profile_id:
                num_self_replies += 1            
        
    if num_statuses:
        avg_length /= num_statuses
        avg_entropy /= num_statuses
        avg_retweet_count /= num_statuses
        avg_favorite_count /= num_statuses

    if num_rts:
        rt_avg_reaction_time /= num_rts

    metrics = [min_length, avg_length, max_length, min_entropy, avg_entropy, max_entropy]

    if status_metrics:
        metrics += [avg_retweet_count, avg_favorite_count]

    if rt_metrics:
        metrics += [rt_avg_reaction_time, num_self_rt, rt_users]

    if reply_metrics:
        metrics += [num_self_replies]

    return metrics

# ref. https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents  
def duplicates_metrics(statuses):
    duplicates = 0
    duplicates_ratio = 0
    group_size = len(statuses)
    corpus = []   

    for r in statuses:
        text = r['text']
        # remove mentions, newlines and make lowercase
        text = re.sub(r'@\w+', '', text).strip().replace('\n', ' ').lower()
        # remove urls   
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        corpus.append(text)

    if len(corpus) > 1 and group_size > 0:
        try:
            vect = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf = vect.fit_transform(corpus)        

            pairwise_similarity = tfidf * tfidf.T 
            pairwise_similarity = pairwise_similarity.toarray()     
            np.fill_diagonal(pairwise_similarity, np.nan)     

            for i, text in enumerate(corpus):
                max_s = 0.0
                max_i = 0

                for j, score in enumerate(pairwise_similarity[i]):
                    if score > max_s:
                        max_s = score
                        max_i = j

                if max_s >= TFIDF_SIMILARITY_THRESHOLD:
                    duplicates += 1
            
            duplicates_ratio = ratio( duplicates, group_size )   
        except ValueError:
            # ValueError: empty vocabulary; perhaps the documents only contain stop words  
            pass    

    return (duplicates, duplicates_ratio)

def extract(profile, tweets, replies, retweets):
    all_statuses = tweets + replies + retweets
    num_tweets = len(tweets)
    num_replies = len(replies)
    num_active_statuses = num_tweets + num_replies
    num_retweets = len(retweets)
    num_total = len(all_statuses)

    unique_languages = Counter()
    unique_hashtags = Counter()
    avg_num_hashtags_per_post = 0

    features = OrderedDict()
    
    # TODO: emoji stats
    # TODO: urls stats
    # TODO: location stats   
    # TODO: source stats, source avg entropy, has custom, known sources histogram 

    # profile data
    features['user_id'] = profile['id']
    
    features['user_screen_name'] = profile['screen_name'].lower()
    features['user_screen_name_length'] = len(features['user_screen_name'])
    features['user_screen_name_entropy'] = entropy(features['user_screen_name'])

    features['days_since_creation'] = days_since( profile['created_at'] )
    features['has_location'] = has_field(profile, 'location')

    features['has_description'] = has_field(profile, 'description')
    if features['has_description']:
        features['description_length'] = len(profile['description'])
        features['description_entropy'] = entropy(profile['description'])
    else:
        features['description_length'] = 0.0
        features['description_entropy'] = 0.0

    features['has_url'] = has_field(profile, 'url')
    features['has_utc_offset'] = has_field(profile, 'utc_offset')
    features['has_time_zone'] = has_field(profile, 'time_zone')
    features['has_lang'] = has_field(profile, 'lang')
    features['contributors_enabled'] = safe_one_hot(profile, 'contributors_enabled')
    features['is_translator'] = safe_one_hot(profile, 'is_translator')
    features['is_translation_enabled'] = safe_one_hot(profile, 'is_translation_enabled')
    features['profile_use_background_image'] = safe_one_hot(profile, 'profile_use_background_image')
    features['has_extended_profile'] = safe_one_hot(profile, 'has_extended_profile')
    features['default_profile'] = safe_one_hot(profile, 'default_profile')
    features['default_profile_image'] = safe_one_hot(profile, 'default_profile_image')
    features['favourites_count'] = profile['favourites_count']
    features['followers_count'] = profile['followers_count']
    features['friends_count'] = profile['friends_count']
    features['followers_to_friends_ratio'] = ratio( profile['followers_count'], profile['friends_count'])
    features['geo_enabled'] = safe_one_hot( profile, 'geo_enabled' )
    features['listed_count'] = profile['listed_count']
    features['protected'] = safe_one_hot( profile, 'protected' )
    features['statuses_count'] = profile['statuses_count']
    features['verified'] = safe_one_hot( profile, 'verified' )

    # process generic features for all statuses
    for status in all_statuses:
        # check for multiple languages
        unique_languages.update([status['lang']])
        # check for hashtags
        if 'entities' in status and 'hashtags' in status['entities']:
            for hashtag in status['entities']['hashtags']:
                unique_hashtags.update([hashtag['text'].lower()])
                avg_num_hashtags_per_post += 1

    avg_num_hashtags_per_post /= num_total

    features['unique_hashtags'] = len(unique_hashtags)
    features['avg_hashtags_per_post'] = avg_num_hashtags_per_post
    features['hashtags_to_tweets_ratio'] = ratio( features['unique_hashtags'], profile['statuses_count'] ) 
    features['unique_languages'] = len(unique_languages)

    # process duplicated statuses and replies
    ( features['duplicate_tweets'], features['duplicate_tweets_ratio'] ) = duplicates_metrics(tweets)
    ( features['duplicate_replies'], features['duplicate_replies_ratio'] ) = duplicates_metrics(replies)

    # process tweets
    ( features['min_tweet_length'],  features['avg_tweet_length'], features['max_tweet_length'], \
      features['min_tweet_entropy'], features['avg_tweet_entropy'] , features['max_tweet_entropy'], \
      features['avg_retweet_count'], features['avg_favorite_count'] ) = metrics_for_statuses(profile['id'], tweets, status_metrics=True)

    # process retweets
    (features['min_retweet_length'], features['avg_retweet_length'], features['max_retweet_length'], \
    features['min_retweet_entropy'] , features['avg_retweet_entropy'], features['max_retweet_entropy'], \
    features['retweets_average_reaction_time'], num_self_retweets, retweed_users ) = metrics_for_statuses(profile['id'], retweets, rt_metrics=True)

    features['retweets_count'] = num_retweets
    features['retweets_to_tweets_ratio'] = ratio( num_retweets, profile['statuses_count'] )
    features['self_retweets_count'] = num_self_retweets
    features['self_retweets_to_tweets_ratio'] = ratio( num_self_retweets, profile['statuses_count'] )

    # process top retweets data
    top_rt_limit = 5
    rt_counters = retweed_users.most_common(top_rt_limit)
    rt_users = [name for name, counter in rt_counters]
    num_rt_users = len(rt_users)
    for i in range(top_rt_limit):
        if num_rt_users >= (i + 1):
            features['top%d_rt_count' % (i + 1)] = rt_counters[i][1]
        else:
            features['top%d_rt_count' % (i + 1)] = 0.0

    # process replies
    (features['min_reply_length'], features['avg_reply_length'] , features['max_reply_length'] , \
    num_self_replies, 
    features['min_reply_entropy'], features['avg_reply_entropy'], features['max_reply_entropy']  ) = metrics_for_statuses(profile['id'], replies, reply_metrics=True)

    features['replies_count'] = num_replies
    features['replies_to_tweets_ratio'] = ratio( num_replies, profile['statuses_count'] )
    features['self_replies_count'] = num_self_replies
    features['self_replies_to_tweets_ratio'] = ratio( num_self_replies, profile['statuses_count'] ) 

    # temporal distribution data
    features.update( temporal_distribution('tweets', tweets) )
    features.update( temporal_distribution('replies', replies) )
    features.update( temporal_distribution('retweets', retweets) )

    return features