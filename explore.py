import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import nltk
import nltk.sentiment
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

import wrangle
import acquire
from prepare import basic_clean, tokenize, lemmatize, stem, remove_stopwords, prep_string_data

# prepare dataset from script functions
df = wrangle.get_npr_data()

# create a question mark count column
df['question_mark_count'] = df.utterance.str.count(r"[\?]")

# extract the total number of question marks for every given speaker, display top ten
df[['speaker','question_mark_count']].groupby(['speaker'])['question_mark_count'] \
                             .count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) \
                             .head(10).T

def npr_host_vs_guest():
    """
    This function produces a visualization comparing the question count of hosts vs guests.
    """
    df['question_mark_count'] = df.utterance.str.count(r"[\?]")

    # create a question mark count df from the above code
    questions = df[['speaker','question_mark_count']].groupby(['speaker'])['question_mark_count'] \
                                 .count() \
                                 .reset_index(name = 'count') \
                                 .sort_values(['count'], ascending = False)

    # merge it with the whole df no speaker for viz purposes below
    questions = questions.merge(df, how ='left', on ='speaker').drop(columns = ['story_id_num', 'utterance_order', 'utterance',
           'title','clean', 'lemmatized', 'vader',
           'question_mark_count'])

    plt.figure(figsize = (10, 6))
    viz = sns.boxplot(data = questions, x = 'is_host', y ='count')
    viz.set_xlabel('Host?', fontsize = 13)
    viz.set_ylabel('Count of Questions', fontsize = 13)
    viz.set_title('Do NPR hosts ask more questions than guests?', fontsize = 17);
    
def avg_sentiment_top_speakers():
    """
    This function produces a visualization of the top 10 speakers average sentiment score.
    """
    # host df
    host_df = df[df.is_host == True]

    # top 10 hosts with the most obseravtions
    hosts_with_the_most = host_df.speaker.value_counts().head(10).index.to_list()

    # limits the overal df to. only thee hosts_with_the_most
    top_hosts_df = df[df.speaker.isin(hosts_with_the_most)]

    # boxplots of the sentiment scores of the top hosts
    plt.figure(figsize = (20,10))
    viz = sns.boxplot(data = top_hosts_df,x = 'speaker', y = 'vader')
    viz.set_title('Average Sentiment Score by Top 10 Speakers')
    viz.set_xlabel('Host Name', fontsize = 13)
    viz.set_ylabel('Vader Score', fontsize = 13)
    plt.xticks(rotation = 45);
    
def sentiment_host_nonhost():
    """
    This function produces a visualization comparing the average sentiment of hosts vs non-hosts.
    """
    # create a non-host df
    non_host_df = df[df.is_host == False]

    # plot vader score for hosts versus not
    plt.figure
    viz = sns.boxplot(data = df, x = 'is_host', y = 'vader')
    viz.set_xlabel('Host?', fontsize = 13)
    viz.set_ylabel('Mean Sentiment Score', fontsize = 13)
    viz.set_title('Mean Sentiment Score for Hosts vs Non-Hosts', fontsize = 17);    
    
def episode_sentiment_year():
    """
    This function changes the date to a datetime object, sets it as the index and 
    produces a visualization of the yearly resampled sentiment score.
    """
    # set date to datetime
    viz_df = df.copy()
    
    viz_df['date'] = pd.to_datetime(df.date)

    # set date to index
    viz_df = viz_df.set_index('date').sort_index()

    # resample vader score by year
    viz_df.resample('Y').vader.mean()

    # create yearly vader df
    vader_yearly = pd.DataFrame(viz_df.resample('Y').vader.mean())

    # and plot
    vader_yearly.plot(figsize = (12, 8), title = 'Episode Sentiment Score - 2005 Through 2019')

    viz_df.reset_index(inplace = True);

def episode_sentiment_weekday():
    """
    This function changes the date to a datetime object, sets it as the index and 
    produces a visualization of the mean sentiment score by day of the week.
    """
    # set date to datetime
    viz_df = df.copy()

    viz_df['date'] = pd.to_datetime(viz_df.date)

    # set date to index
    viz_df = viz_df.set_index('date').sort_index()

    # resample vader score by year
    viz_df.resample('Y').vader.mean()

    # sentiment score by  day of week
    viz_df.groupby(viz_df.index.day_name()).vader.mean()

    # sentiment by day, 0 is monday, 6 is sunday

    # plot and order the avg sentiment score by day of week
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    plt.figure(figsize = (10,6))
    viz = sns.boxplot(data = viz_df, x = viz_df.index.day_name(), y = 'vader', order = order)
    viz.set_xlabel('Week Day', fontsize = 10)
    viz.set_ylabel('Vader Score', fontsize = 10)
    viz.set_title('Mean Vader Score by Week Day', fontsize = 17)

    viz_df.reset_index(inplace = True);
    
    
    
    
    
    