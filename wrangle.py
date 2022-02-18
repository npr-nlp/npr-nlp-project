import os
import json
import numpy as np
import pandas as pd
import acquire
import prepare

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
    
    return df