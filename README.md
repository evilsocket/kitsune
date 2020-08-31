# Kitsune

An artificial neural network designed to detect and correlate Twitter profiles with similar behaviours, originally developed to detect automated Twitter accounts (bots), but that can be used for any custom list of accounts.

**! WORK IN PROGRES, STILL IN BETA STAGE**

## Instructions

Make sure you have python3 and pip3 installed, then proceed to install the requirements:

    cd /path/to/kitsune
    sudo pip3 install -r requirements.txt
    
Then you'll need to create two folders, in this example we'll create a `bots` folder and a `legit` folder. Place in each one a file named `seed.txt` with the list of accounts you want to be classified in that group, so that you'll have:

    /path/to/bots/seed.txt
    /path/to/legit/seed.txt

Ideally the two lists should cointain the same number accounts, at least in the order of one hundred each. The more accounts you'll use and the more accurately they're grouped, the more accurate the model will be.

Now you will have to download the last tweets and profile data for each list:

    /path/to/kitsune/download.py \
        --consumer_key TWITTER_CONSUMER_KEY \
        --consumer_secret TWITTER_CONSUMER_SECRET \
        --access_token TWITTER_ACCESS_TOKEN \
        --access_token_secret TWITTER_ACCESS_SECRET \
        --seed /path/to/bots/seed.txt \
        --output /path/to/bots

and then:

    /path/to/kitsune/download.py \
        --consumer_key TWITTER_CONSUMER_KEY \
        --consumer_secret TWITTER_CONSUMER_SECRET \
        --access_token TWITTER_ACCESS_TOKEN \
        --access_token_secret TWITTER_ACCESS_SECRET \
        --seed /path/to/legit/seed.txt \
        --output /path/to/legit

Now it's time to transform this data into numerical features in a CSV file that kitsune can understand (for the complete features set seet [kitsune/features.py](https://github.com/evilsocket/kitsune/blob/master/kitsune/features.py), keeping in mind this file is changing and improving very fast at this stage):

    /path/to/kitsune/encode.py \
        --label_a bot --path_a /path/to/bots \
        --label_b legit --path_b /path/to/legit \
        --output /path/to/dataset.csv

Once this is done, you can train the model:

     /path/to/kitsune/train.py \
        --dataset /path/to/dataset.csv \
        --output /path/to/model.h5
        
This will start the training, print accuracy metrics and save the model, normalization values and features relevances in the folder you specified.

To test the model predictions on a profile folder or multiple folders at once:

     /path/to/kitsune/test.py \
        --model /path/to/model.h5 \
        --profile /path/to/profile-data-folder

## License

`kitsune` is made with ♥  by [@evilsocket](https://twitter.com/evilsocket) and it is released under the GPL3 license.        