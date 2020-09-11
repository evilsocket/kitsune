import time
import math
import re
import numpy as np
from datetime import datetime
from email.utils import parsedate
from urllib.parse import urlparse, ParseResult
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

today = datetime.today()

EMOJI_RE                   = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
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

    num_statuses = len(statuses)
    by_hour = Counter({i : 0 for i in range(0, 24)})
    by_week_day = Counter({i : 0 for i in range(0, 7)})
    by_month = Counter({i : 0 for i in range(1, 13)})

    prev = None
    avg_delta_t = 0

    for status in statuses:
        parsed = parse_date(status['created_at'])

        if prev is not None:
            avg_delta_t += (parsed - prev).total_seconds()

        prev = parsed
        by_hour[parsed.hour] += 1
        by_week_day[parsed.weekday()] += 1
        by_month[parsed.month] += 1

    features['%s_avg_delta_t' % prefix] = (avg_delta_t / num_statuses) if num_statuses > 0 else 0.0

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

def statuses_metrics(profile_id, statuses, status_metrics=False, rt_metrics=False, reply_metrics=False):
    num_statuses = len(statuses)
    avg_entropy = 0
    min_entropy = 0
    max_entropy = 0

    avg_length = 0
    min_length = 0
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
        if min_length == 0 or text_size < min_length:
            min_length = text_size
        if text_size > max_length:
            max_length = text_size

        # shannon entropy metrics
        text_entropy = entropy(status['text'])
        avg_entropy += text_entropy
        if min_entropy == 0 or text_entropy < min_entropy:
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
            rt_avg_reaction_time += (retweeted_at - created_at).total_seconds()
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
            vect = TfidfVectorizer(min_df=1)
            tfidf = vect.fit_transform(corpus)        

            pairwise_similarity = tfidf * tfidf.T 
            pairwise_similarity = pairwise_similarity.toarray()     
            np.fill_diagonal(pairwise_similarity, np.nan)     

            for i, text in enumerate(corpus):
                for j, score in enumerate(pairwise_similarity[i]):
                    if i != j and score >= TFIDF_SIMILARITY_THRESHOLD:
                        duplicates += 1

            duplicates_ratio = ratio( duplicates, group_size )   
        except ValueError:
            # ValueError: empty vocabulary; perhaps the documents only contain stop words  
            pass    

    return (duplicates, duplicates_ratio)

def source_metrics(user_statuses):
    total = len(user_statuses)
    if total == 0:
        return  (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    adv = 0
    android = 0
    blackberry = 0
    ipad = 0
    iphone = 0
    mac = 0
    websites = 0
    windows = 0
    non_std = 0

    sources = Counter()
    for status in user_statuses:
        src = status['source'].lower()
        sources.update([src])

    for src, count in sources.items():
        if 'twitter for advertisers' in src:
            adv += count
        elif 'twitter for android' in src:
            android += count
        elif 'twitter for blackberry' in src:
            blackberry += count
        elif 'twitter for ipad' in src:
            ipad += count
        elif 'twitter for iphone' in src:
            iphone += count
        elif 'twitter for mac' in src:
            mac += count
        elif 'twitter for websites' in src:
            websites += count
        elif 'twitter for windows' in src:
            windows += count
        elif 'twitter.com' not in src:
            non_std += count

    return ( \
        len(sources), 
        adv / total, 
        android / total, 
        blackberry / total,
        ipad / total,
        iphone / total,
        mac / total,
        websites / total,
        windows / total,
        non_std / total)

def urls_metrics(user_statuses):
    unique_urls = Counter()
    unique_domains = Counter()
    num_http_scheme = 0
    num_https_scheme = 0
    num_other_scheme = 0
    num_statuses_with_urls = 0
    num_not_parsable = 0

    for s in user_statuses:
        if len(s['entities']['urls']) > 0:
            num_statuses_with_urls += 1
            for url in s['entities']['urls']:
                url = url['expanded_url'].lower()
                try:
                    unique_urls.update([url])
                    url = urlparse(url)
                    unique_domains.update([url.netloc])

                    if url.scheme == 'http':
                        num_http_scheme += 1
                    elif url.scheme == 'https':
                        num_https_scheme += 1
                    else:
                        num_other_scheme += 1

                except:
                    num_not_parsable += 1

    top3 = unique_urls.most_common(3)
    ntop = len(top3)
    tops = [0.0, 0.0, 0.0]

    for i in range(ntop):
        tops[i] = top3[i][1]

    return [ len(unique_urls), len(unique_domains), 
             num_statuses_with_urls, num_not_parsable,
             num_http_scheme, num_https_scheme, num_other_scheme ] + tops

def emoji_metrics(user_statuses):
    total = len(user_statuses)
    if total == 0:
        return (0, 0)

    unique = Counter()
    for s in user_statuses:
        emojis = EMOJI_RE.findall(s['text'])
        if emojis:
            for e in emojis:
                unique.update([e])

    return ( len(unique), sum([c for e, c in unique.items()]) / total )

def place_metrics(user_statuses):
    total = len(user_statuses)
    if total == 0:
        return (0, 0, 0, 0, 0)

    unique = Counter()
    for s in user_statuses:
        if 'place' in s and s['place'] is not None:
            unique.update([s['place']['id']])

    tot_statuses_with_place = sum([c for e, c in unique.items()])

    top3 = unique.most_common(3)
    ntop = len(top3)
    tops = [0.0, 0.0, 0.0]

    for i in range(ntop):
        tops[i] = top3[i][1] / tot_statuses_with_place

    return [
        len(unique), 
        tot_statuses_with_place / total
    ] + tops

def extract(profile, tweets, replies, retweets):
    all_statuses = tweets + replies + retweets
    user_statuses = tweets + replies

    # make sure they're properly sorted
    tweets.sort(key=lambda x: parse_date(x['created_at']))
    replies.sort(key=lambda x: parse_date(x['created_at'])) 
    retweets.sort(key=lambda x: parse_date(x['created_at']))
    all_statuses.sort(key=lambda x: parse_date(x['created_at']))
    user_statuses.sort(key=lambda x: parse_date(x['created_at']))

    num_tweets = len(tweets)
    num_replies = len(replies)
    num_retweets = len(retweets)
    num_total = len(all_statuses)

    unique_languages = Counter()
    unique_hashtags = Counter()
    avg_num_hashtags_per_post = 0

    features = OrderedDict()
    
    # TODO: media stats

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

    # process sources   
    ( features['unique_sources'], 
      features['source_adv_ratio'], 
      features['source_android_ratio'],
      features['source_blackberry_ratio'],
      features['source_ipad_ratio'],
      features['source_iphone_ratio'],
      features['source_mac_ratio'],
      features['source_websites_ratio'],
      features['source_windows_ratio'],
      features['source_non_std_ratio'] ) = source_metrics(user_statuses)

    # URLs
    ( features['unique_urls'], features['unique_domains'], 
      features['statuses_with_urls'], features['not_parsable'],
      features['http_scheme'], features['https_scheme'], features['other_scheme'],
      features['top1_url_count'], features['top2_url_count'], features['top3_url_count'] ) = urls_metrics(user_statuses)

    # emojis
    ( features['unique_emojis'], features['emoji_ratio'] ) = emoji_metrics(user_statuses) 

    # places 
    ( features['unique_places'], features['places_ratio'],
      features['top1_place_ratio'], features['top2_place_ratio'], features['top3_place_ratio'] ) = place_metrics(user_statuses) 

    # process duplicated statuses and replies
    ( features['duplicate_tweets'], features['duplicate_tweets_ratio'] ) = duplicates_metrics(tweets)
    ( features['duplicate_replies'], features['duplicate_replies_ratio'] ) = duplicates_metrics(replies)

    # process tweets
    ( features['min_tweet_length'],  features['avg_tweet_length'], features['max_tweet_length'], \
      features['min_tweet_entropy'], features['avg_tweet_entropy'] , features['max_tweet_entropy'], \
      features['avg_retweet_count'], features['avg_favorite_count'] ) = statuses_metrics(profile['id'], tweets, status_metrics=True)

    # process retweets
    (features['min_retweet_length'], features['avg_retweet_length'], features['max_retweet_length'], \
    features['min_retweet_entropy'] , features['avg_retweet_entropy'], features['max_retweet_entropy'], \
    features['retweets_average_reaction_time'], num_self_retweets, retweed_users ) = statuses_metrics(profile['id'], retweets, rt_metrics=True)

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
    features['min_reply_entropy'], features['avg_reply_entropy'], features['max_reply_entropy']  ) = statuses_metrics(profile['id'], replies, reply_metrics=True)

    features['replies_count'] = num_replies
    features['replies_to_tweets_ratio'] = ratio( num_replies, profile['statuses_count'] )
    features['self_replies_count'] = num_self_replies
    features['self_replies_to_tweets_ratio'] = ratio( num_self_replies, profile['statuses_count'] ) 

    # temporal distribution data
    features.update( temporal_distribution('tweets', tweets) )
    features.update( temporal_distribution('replies', replies) )
    features.update( temporal_distribution('retweets', retweets) )

    return features