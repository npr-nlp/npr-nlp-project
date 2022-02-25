import os
import numpy as np
import pandas as pd
import acquire
import prepare
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


'''
This module uses the acquire and prepare modules to create a usable NPR corpus.

Any further wrangling can be added here.
'''


def get_npr_data():
    '''
    This function reads in the NPR corpus data, cleans, and prepares it then 
    writes data to a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('./npr_corpus.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('./npr_corpus.csv', index_col=0)      
        # print statements for size of each df
        print(f'The df has {df.shape[0]} rows and {df.shape[1]} columns.')

    else:
        
        # Read fresh data from db into a DataFrame
        df = acquire.get_df()

        # prepare data
        df = prepare.prep_npr_data(df)

        # Cache data
        df.to_csv('npr_corpus.csv')

        # print statements for size of each df
        print(f'The df has {df.shape[0]} rows and {df.shape[1]} columns.')
    df.drop(columns='episode_date', inplace = True)
    #change name of a couple columns:
    df = df.rename(columns={'episode_id':"story_id_num", 'episode_order':'utterance_order'})
    
    # regex to adjust speaker names
    df['speaker'] = df.speaker.str.replace(r'\([^)]*\)','', regex=True)
    df['speaker'] = df.speaker.str.replace(r'host','', regex=True)
    df['speaker'] = df.speaker.str.replace(r'[^a-z0-9\s\.]','', regex=True).str.strip()
    df['speaker'] = df.speaker.str.replace('lourdes','lulu')

    ## redo is_host
    # create host_map and list of hosts
    host_map = pd.read_json('host-map.json')
    host_map = host_map.T
    hosts = host_map.name.to_list()
    hosts.append('neal conan')
    df['is_host'] = df.speaker.isin(hosts)
    return df

def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes. 
    '''
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(df.lemmatized)
    # y = df.language
    train, test = train_test_split(df, test_size = .2, stratify = df.is_host, random_state = 222)
    train, validate = train_test_split(train, test_size = .3, stratify = train.is_host,random_state = 222)
    return train, validate, test