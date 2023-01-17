import os
import pandas as pd
from utils import utils

def path():
    path = 'C:/Users/1142770/Downloads/BERT/data/emotion_analysis_data'
    # os.listdir(path)
    return path

def data_load():
    path = path()

    header = ['default']
    train_df = pd.read_csv(os.path.join(path + 'train.txt'), names=header, encoding='utf-8')
    test_df = pd.read_csv(os.path.join(path + 'test.txt'), names=header, encoding='utf-8')
    val_df = pd.read_csv(os.path.join(path + 'val.txt'), names=header, encoding='utf-8')
    # train_df.head()

    df_list = [train_df, test_df, val_df]
    for df in df_list:
        df['content'] = df.default.str.split(';').str[0]
        df['emotion'] = df.default.str.split(';').str[1]
        df.drop('default', axis=1, inplace=True)
    # train_df.head()

    set(train_df['emotion'])
    change_value_dict = {'anger':0, 'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}

    df_list = [train_df, test_df, val_df]
    for df in df_list:
        df = df.replace({'emotion': change_value_dict}, inplace=True)
    # val_df.head()

    # print(len(train_df))
    # print(len(test_df))
    # print(len(val_df))

    X_train = train_df['content']
    y_train = train_df['emotion']

    X_test = test_df['content']
    y_test = test_df['emotion']

    X_val = val_df['content']
    y_val = val_df['emotion']

    return X_train, y_train, X_test, y_test, X_val, y_val

def preprocess():
    tokenizer = utils.tokenizer_setting()
    X_train, y_train, X_test, y_test, X_val, y_val = data_load()

    train_dataloader = utils.preprocessing(X_train, y_train, tokenizer, process_type='test')
    test_dataloader = utils.preprocessing(X_test, y_test, tokenizer, process_type='test')
    validation_dataloader = utils.preprocessing(X_val, y_val, tokenizer, process_type='test')

    return train_dataloader, test_dataloader, validation_dataloader