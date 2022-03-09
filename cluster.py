import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import wrangle
import matplotlib as plt
import seaborn as sns

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def get_clusters(df, sample_size = 10_000):
    '''
    Takes in a dataframe and creates clusters based on a sample size
    '''
    df2 = df.sample(sample_size, random_state = 222)

    train, validate, test = wrangle.split_data(df2)
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
    kmeans.predict(X_train_cluster)

    # add cluster features to train and X_train df's
    train_scaled['cluster'] = kmeans.predict(X_train_cluster)
    validate_scaled['cluster'] = kmeans.predict(X_validate_cluster)
    test_scaled['cluster'] = kmeans.predict(X_test_cluster)
    X_train_cluster['cluster'] = kmeans.predict(X_train_cluster)
    return train, validate, test, train_scaled, validate_scaled, test_scaled, X_train_cluster


def plot_clusters(X_train_cluster):
    '''
    Takes in an X_train_cluster and returns a 3d cluster plot
    '''
    fig = plt.figure(figsize = (14,14))
    ax = fig.add_subplot(111, projection = '3d')
    x = X_train_cluster.word_count
    y = X_train_cluster.vader
    z = X_train_cluster.question_mark_count
    ax.scatter(x,y,z, c=X_train_cluster.cluster, s = 40, cmap = 'jet')
    # ax.legend()
    ax.set_xlabel('word_count', fontsize = 15)
    ax.set_ylabel('vader',fontsize = 15)
    ax.set_zlabel('question_mark_count',fontsize = 15)
    plt.title('NPR Clusters by Sentiment, Word Count, and Question Marks')
    return plt.show()