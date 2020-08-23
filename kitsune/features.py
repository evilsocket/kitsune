from datetime import datetime
from email.utils import parsedate
import time

from collections import Counter, OrderedDict

today = datetime.today()

def parse_date(str_date):
    return datetime(*(parsedate(str_date)[:6]))

def days_since(str_date):
    return (today - parse_date(str_date)).days

def ratio(a, b):
    return a / (b + .00000001) # avoid division by zero
 
def one_hot(b):
    return 1.0 if b else 0.0

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

def metrics_for_statuses(profile_id, statuses, rt_metrics=False, reply_metrics=False):
    num_statuses = len(statuses)
    avg_length = 0
    min_length = 0
    max_length = 0

    rt_avg_reaction_time = 0.0
    rt_users = Counter()
    num_self_rt = 0
    num_rts = 0

    num_self_replies = 0

    for status in statuses:
        text_size = len(status['text'])
        avg_length += text_size
        if text_size < min_length:
            min_length = text_size
        if text_size > max_length:
            max_length = text_size

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

    if num_rts:
        rt_avg_reaction_time /= num_rts

    metrics = [min_length, avg_length, max_length]

    if rt_metrics:
        metrics += [rt_avg_reaction_time, num_self_rt, rt_users]

    if reply_metrics:
        metrics += [num_self_replies]

    return metrics

def extract(profile, tweets, replies, retweets):
    all_statuses = tweets + replies + retweets
    num_tweets = len(tweets)
    num_replies = len(replies)
    num_retweets = len(retweets)
    num_total = len(all_statuses)

    unique_languages = Counter()
    unique_hashtags = Counter()
    avg_num_hashtags_per_post = 0

    features = OrderedDict()
    
    # TODO: emoji stats

    # profile data
    features['user_id'] = profile['id']
    features['user_screen_name'] = profile['screen_name'].lower()
    features['days_since_creation'] = days_since( profile['created_at'] )
    features['has_location'] = has_field(profile, 'location')
    features['has_description'] = has_field(profile, 'description')
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
    features['tweets_to_hashtags_ratio'] = ratio( profile['statuses_count'], features['unique_hashtags'] ) 
    features['unique_languages'] = len(unique_languages)

    # process tweets
    (min_tweet_length, avg_tweet_length, max_tweet_length) = metrics_for_statuses(profile['id'], tweets)
    
    features['min_tweet_length'] = min_tweet_length
    features['max_tweet_length'] = max_tweet_length
    features['avg_tweet_length'] = avg_tweet_length

    # process retweets
    (min_retweet_length, avg_retweet_length, max_retweet_length, \
    retweet_avg_reaction_time, num_self_retweets, retweed_users ) = metrics_for_statuses(profile['id'], retweets, rt_metrics=True)
    
    features['min_retweet_length'] = min_retweet_length
    features['max_retweet_length'] = max_retweet_length
    features['avg_retweet_length'] = avg_retweet_length
    features['retweets_count'] = num_retweets
    features['retweets_average_reaction_time'] = retweet_avg_reaction_time
    features['tweets_to_retweets_ratio'] = ratio( profile['statuses_count'], num_retweets )
    features['self_retweets_count'] = num_self_retweets
    features['tweets_to_self_retweets_ratio'] = ratio( profile['statuses_count'], num_self_retweets )

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
    (min_reply_length, avg_reply_length, max_reply_length, num_self_replies) = metrics_for_statuses(profile['id'], replies, reply_metrics=True)

    features['min_reply_length'] = min_reply_length
    features['max_reply_length'] = max_reply_length
    features['avg_reply_length'] = avg_reply_length
    features['replies_count'] = num_replies
    features['tweets_to_replies_ratio'] = ratio( profile['statuses_count'], num_replies )
    features['self_replies_count'] = num_self_replies
    features['tweets_to_self_replies_ratio'] = ratio( profile['statuses_count'], num_self_replies ) 

    # temporal distribution data
    features.update( temporal_distribution('tweets', tweets) )
    features.update( temporal_distribution('replies', replies) )
    features.update( temporal_distribution('retweets', retweets) )

    return features