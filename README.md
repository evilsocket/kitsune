# Kitsune

An artificial neural network designed to detect and correlate Twitter profiles with similar behaviours, originally developed to detect automated Twitter accounts (bots), but that can be used for any custom list of accounts.

**! WORK IN PROGRES, STILL IN BETA STAGE !**

## Instructions

Make sure you have python3 and pip3 installed, then proceed to install the requirements:

    cd /path/to/kitsune
    sudo pip3 install -r requirements.txt
    
You'll need to create two folders, in this example we'll create a `bots` folder and a `legit` folder. Place in each one a file named `seed.txt` with the list of accounts you want to be classified in that group, so that you'll have:

    /path/to/bots/seed.txt
    /path/to/legit/seed.txt

Ideally the two lists should cointain the same number accounts, at least in the order of one hundred each. The more accounts you'll use and the more accurately they're grouped, the more accurate the model will be.

Download the last tweets and profile data for each list:

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

    normalizing dataset ...
    data shape: (1797, 198) (197 features)
    bots:541 legit:1256
    generating train, test and validation datasets (test=0.150000 validation=0.150000) ...
    unique labels: 2
    building neural network for: inputs=197 outputs=2
    ...
    training model ...
    Epoch 1/100
    79/79 - 0s - loss: 0.3349 - binary_crossentropy: 0.3349 - binary_accuracy: 0.8568 - val_loss: 0.1520 - val_binary_crossentropy: 0.1520 - val_binary_accuracy: 0.9442
    Epoch 2/100
    79/79 - 0s - loss: 0.1685 - binary_crossentropy: 0.1685 - binary_accuracy: 0.9300 - val_loss: 0.1346 - val_binary_crossentropy: 0.1346 - val_binary_accuracy: 0.9480
    Epoch 3/100
    ...
    79/79 - 0s - loss: 0.0130 - binary_crossentropy: 0.0130 - binary_accuracy: 0.9960 - val_loss: 0.0627 - val_binary_crossentropy: 0.0627 - val_binary_accuracy: 0.9777

To test the model predictions on a profile folder or multiple folders at once:

     /path/to/kitsune/test.py \
        --model /path/to/model.h5 \
        --profile /path/to/profile-data-folder

    writing predictions to /path/to/profile-data-folder/predictions.csv ...

    -------

            screen_name | class | confidence

        someusername   bot     100.000000 %
        someusername   bot     99.893301 %
        someusername   bot     99.999895 %
        someusername   bot     99.993192 %
        someusername   bot     66.441199 %
        someusername   bot     99.981043 %
        someusername   bot     99.999995 %
        someusername   bot     99.999995 %
        someusername   bot     99.760059 %

## License

`kitsune` is made with ♥  by [@evilsocket](https://twitter.com/evilsocket) and it is released under the GPL3 license.        