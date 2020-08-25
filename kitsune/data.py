import pandas as pd
import tensorflow as tf
import json

def load(filename, normalize=True):
    print("normalizing dataset ...")

    dataset = pd.read_csv(filename)
    
    # convert labels to numbers
    dataset['label'] = dataset['label'].apply(lambda x: 1.0 if x == 'bot' else 0.0)
    # drop unrequired columns
    dataset = dataset.drop(columns=['user_id','user_screen_name'], axis = 1)

    if normalize:    
        # normalization
        data_min = dataset.min()
        data_max = dataset.max()

        dataset = ((dataset - data_min) / (data_max - data_min)).fillna(0.0)

        return (data_min, data_max, dataset)
    else:
        return dataset

def split_row(row, n_labels):
    x = row.iloc[:,1:].copy()
    y = tf.keras.utils.to_categorical(row.values[:,0], n_labels)
    return x, y

def split(dataset, p_test, p_val):
    print("generating train, test and validation datasets (test=%f validation=%f) ..." % (p_test, p_val))

    # randomly resample
    dataset = dataset.sample(frac = 1).reset_index(drop = True)
    # count unique labels on first column if no counter is provided externally
    n_labels = len(dataset.iloc[:,0].unique())

    print("unique labels: %d" % n_labels)

    n_tot   = len(dataset)
    n_train = int(n_tot * ( 1 - p_test - p_val))
    n_test  = int(n_tot * p_test)
    n_val   = int(n_tot * p_val)

    train      = dataset.head(n_train)
    test       = dataset.head(n_train + n_test).tail(n_test)
    validation = dataset.tail(n_val)

    X_train, Y_train = split_row(train, n_labels)
    X_test,  Y_test  = split_row(test, n_labels)
    X_val,   Y_val   = split_row(validation, n_labels)

    return (X_train, Y_train, X_test, Y_test, X_val, Y_val)

def save_normalizer(filename, datamin, datamax):
    print("saving normalizing values to %s ..." % filename)

    datamin = datamin.to_dict()
    datamax = datamax.to_dict()

    del datamin['label']
    del datamax['label']

    with open(filename, 'w+t') as fp:
        json.dump({
            'min': datamin,
            'max': datamax
        }, fp, indent=2, sort_keys=True)

def load_normalizer(filename):
    with open('norm.json', 'rt') as fp:
        norm = json.load(fp)

    norm['max'] = pd.DataFrame([norm['max']], columns=norm['max'].keys())
    norm['min'] = pd.DataFrame([norm['min']], columns=norm['min'].keys())

    return norm
        
def normalize(norm, vector):
    # IMPORTANT: use vector columns order!  
    for column in vector:
        v = vector[column]
        min_v = norm['min'][column]
        max_v = norm['max'][column]
        vector[column] = (v - min_v) / (max_v - min_v)
    return vector.fillna(0)

def nomalized_from_dict(norm, v):
    vector = pd.DataFrame([v], columns=v.keys())
    # drop unrequired columns
    vector = vector.drop(columns=['user_id','user_screen_name'], axis = 1)
    # normalization
    return normalize(norm, vector)    