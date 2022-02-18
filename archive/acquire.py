import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_df():
    '''
    Obtains working dataframe from utterances and episodes. Combines the two and 
    labels whether the speaker is a host or not.
    '''
    # read csv's into dataframes
    df = pd.read_csv('./utterances.csv')
    ep_df = pd.read_csv('./episodes.csv')

    # joining utterances df ('df') and episodes on 'id'
    joined_df = pd.merge(df, ep_df, left_on = 'episode', right_on='id', how = 'inner')
    joined_df.drop(columns = ['id'], inplace=True)
    joined_df.rename(columns={'episode':'episode_id'}, inplace = True)
    joined_df['is_host'] = joined_df.speaker.str.contains(r'\W*(host)\W*')
    
    # return joined dataframe with all info
    return joined_df


