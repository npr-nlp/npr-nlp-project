import os
import json
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
        prepare.prep_npr_data(df)

        # Cache data
        df.to_csv('npr_corpus.csv')

        # print statements for size of each df
        print(f'The df has {df.shape[0]} rows and {df.shape[1]} columns.')
    
    # fix is_host column
    df['speaker'] = df.speaker.str.lower()
    df['is_host'] = df.speaker.str.contains(r'\W*(host)\W*')

    return df

def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes. 
    '''
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(df.lemmatized)
    # y = df.language
    train, test = train_test_split(df, test_size = .2, stratify = df.language, random_state = 222)
    train, validate = train_test_split(train, test_size = .3, stratify = train.language, random_state = 222)
    return train, validate, test