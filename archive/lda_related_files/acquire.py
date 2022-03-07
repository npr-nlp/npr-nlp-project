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
    # drop extra columns
    joined_df.drop(columns = ['id'], inplace=True)
    # rename columns
    joined_df.rename(columns={'episode':'episode_id'}, inplace = True)
    # lowercase str columns
    joined_df['speaker'] = joined_df.speaker.str.lower()
    joined_df['program'] = joined_df.program.str.lower()
    joined_df['title'] = joined_df.title.str.lower()

    # create column identifying hosts
    joined_df['is_host'] = joined_df.speaker.str.contains(r'\W*(host)\W*')

    # return joined dataframe with all info
    return joined_df


