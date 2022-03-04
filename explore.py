import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import nltk
import nltk.sentiment
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.cluster import KMeans

import wrangle
import acquire
from prepare import basic_clean, tokenize, lemmatize, stem, remove_stopwords, prep_string_data

def top_npr_speakers(df):
    """
    This function produces a visualization of the top 10 hosts from NPR.
    """
    df.speaker[df.is_host == True].value_counts().head(10).plot.bar(title = 'Top 10 NPR Speakers', 
                                                                    ec = 'black',
                                                                   figsize = (18, 12))
    plt.rc('font', size = 14)
    plt.xticks(rotation = 45);

def speaker_count(df):
    # create a question mark count column
    df['question_mark_count'] = df.utterance.str.count(r"[\?]")

    # extract the total number of question marks for every given speaker, display top ten
    return df[['speaker','question_mark_count']].groupby(['speaker'])['question_mark_count'] \
                                 .count() \
                                 .reset_index(name='count') \
                                 .sort_values(['count'], ascending=False) \
                                 .head(10).T
    
    
def npr_host_vs_guest(df):
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
    
def avg_sentiment_top_speakers(df):
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
    
def sentiment_host_nonhost(df):
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
    
def episode_sentiment_year(df):
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


def episode_sentiment_month(df):
    """
    This function changes the date to a datetime object, sets it as the index and 
    produces a visualization of the monthly resampled sentiment score.
    """
    # set date to datetime
    viz_df = df.copy()

    viz_df['date'] = pd.to_datetime(df.date)

    # set date to index
    viz_df = viz_df.set_index('date').sort_index()

    # resample vader score by year
    viz_df.resample('Y').vader.mean()

    # create monthly vader df
    vader_monthly = pd.DataFrame(viz_df.resample('M').vader.mean())

    # and plot
    vader_monthly.plot(figsize = (12, 8), title = 'Episode Sentiment Score by Month - 2005 Through 2019')

    viz_df.reset_index(inplace = True);
    
    
def episode_sentiment_weekday(df):
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
    
def program_sentiment(df):
    """
    This function changes the date to a datetime object, sets it as the index and 
    produces a visualization of the resampled average sentiment by program.
    """
    viz_df = df.copy()

    viz_df['date'] = pd.to_datetime(viz_df.date)

    # set date to index
    viz_df = viz_df.set_index('date').sort_index()

    #  making df for each program
    talk_of_the_nation_df = viz_df[viz_df.program == 'talk of the nation']
    morning_edition_df = viz_df[viz_df.program == 'morning edition']
    all_things_considered_df = viz_df[viz_df.program == 'all things considered']
    news_and_notes_df = viz_df[viz_df.program == 'news & notes']
    weekend_edition_saturday_df = viz_df[viz_df.program == 'weekend edition saturday']
    weekend_edition_sunday_df = viz_df[viz_df.program == 'weekend edition sunday']
    day_to_day_df = viz_df[viz_df.program == 'day to day']

    # plot the  sentimen over time of each program
    plt.figure(figsize = (30,10))

    talk_of_the_nation_df.resample("y").vader.mean().plot(alpha = .5)
    all_things_considered_df.resample("y").vader.mean().plot(alpha = .5)
    morning_edition_df.resample("y").vader.mean().plot(alpha = .5)
    news_and_notes_df.resample("y").vader.mean().plot(alpha = .5)
    day_to_day_df.resample("y").vader.mean().plot(alpha = .5)
    weekend_edition_sunday_df.resample("y").vader.mean().plot(alpha = .5)
    weekend_edition_saturday_df.resample("y").vader.mean().plot(alpha = .5)

    plt.title("Average Sentiment Over Time, by Program")
    plt.legend(['Talk of the Nation',
     'All Things Considered',
     'Morning Edition',
     'News & Notes',
     'Day to Day',
     'Weekend Edition Sunday',
     'Weekend Edition Saturday'], prop = {'size': 20});    
    
    
def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range = (0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def elbow_method(df):
    """
    This function produces a curve showing the most optimal k point.
    """
# create continuous variables easily available to us
    df['message_length'] = df.clean.apply(len)
    df['word_count'] = df.clean.apply(str.split).apply(len)
    df['question_mark_count'] = df.utterance.str.count(r"[\?]")

    df2 = df.sample(10000, random_state = 222)

    train, validate, test = wrangle.split_data(df2)
    # split
    scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(train, validate, test)
    # define features for clustering
    X_train_cluster = train_scaled[['vader', 'word_count','question_mark_count']]

    #repeat for validate and test
    X_validate_cluster = validate_scaled[['vader', 'word_count','question_mark_count']]
    X_test_cluster = test_scaled[['vader', 'word_count','question_mark_count']]
    # define cluster object
    kmeans = KMeans(n_clusters=4, random_state = 333)
    # fit cluster object to features
    kmeans.fit(X_train_cluster)
    # use the object
    kmeans.predict(X_train_cluster);
    
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize = (10, 10))
        pd.Series({k: KMeans(k).fit(X_train_cluster).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        
def k_clusters(df):
    """
    This function produces four subplots with different number of clusters on each subplot.
    """
    # create continuous variables easily available to us
    df['message_length'] = df.clean.apply(len)
    df['word_count'] = df.clean.apply(str.split).apply(len)
    df['question_mark_count'] = df.utterance.str.count(r"[\?]")

    df2 = df.sample(10000, random_state = 222)

    train, validate, test = wrangle.split_data(df2)
    # split
    scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(train, validate, test)
    # define features for clustering
    X_train_cluster = train_scaled[['vader', 'word_count','question_mark_count']]

    #repeat for validate and test
    X_validate_cluster = validate_scaled[['vader', 'word_count','question_mark_count']]
    X_test_cluster = test_scaled[['vader', 'word_count','question_mark_count']]
    # define cluster object
    kmeans = KMeans(n_clusters=4, random_state = 333)
    # fit cluster object to features
    kmeans.fit(X_train_cluster)
    # use the object
    kmeans.predict(X_train_cluster);
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)

    for ax, k in zip(axs.ravel(), range(4, 8)):
        clusters = KMeans(k).fit(X_train_cluster).predict(X_train_cluster)
        ax.scatter(X_train_cluster.vader, X_train_cluster.word_count, c=clusters)
        ax.set(title='k = {}'.format(k), xlabel='vader', ylabel='word_count')
    
    